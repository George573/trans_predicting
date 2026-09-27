package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"

	"stand/internal/conditions"
	"stand/internal/features"
	"stand/internal/forecast"
)

const (
	maxBody       = 1 << 20
	maxHourDays   = 31
	maxConditions = 16
	maxRows       = 50000
)

var horizons = []string{"day", "week", "month", "season"}

type forecastRequest struct {
	Model       string             `json:"model"`
	Routes      []int              `json:"routes"`
	From        string             `json:"from"`
	To          string             `json:"to"`
	Granularity string             `json:"granularity"`
	Horizon     string             `json:"horizon"`
	Corridor    *bool              `json:"corridor"`
	Conditions  []conditions.Input `json:"conditions"`
}

type explainRequest struct {
	Model      string             `json:"model"`
	Route      int                `json:"route"`
	Date       string             `json:"date"`
	Hour       *int               `json:"hour"`
	Horizon    string             `json:"horizon"`
	Conditions []conditions.Input `json:"conditions"`
}

type query struct {
	engine     *forecast.Engine
	routes     []int
	from       time.Time
	days       int
	daily      bool
	horizon    string
	corridor   bool
	conditions []conditions.Input
}

func badRequest(field, msg string) *apiError {
	return &apiError{status: http.StatusBadRequest, Code: "bad_request", Field: field, Message: msg}
}

func tooLarge(code, field, msg string) *apiError {
	return &apiError{status: http.StatusRequestEntityTooLarge, Code: code, Field: field, Message: msg}
}

func decode(w http.ResponseWriter, r *http.Request, v any) error {
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxBody))
	err := dec.Decode(v)
	if err == nil {
		if _, err = dec.Token(); err == io.EOF {
			return nil
		} else if err == nil {
			err = errors.New("после JSON в теле есть ещё данные")
		}
	}
	var big *http.MaxBytesError
	var te *json.UnmarshalTypeError
	switch {
	case errors.As(err, &big):
		return tooLarge("bad_request", "", "Тело запроса больше 1 МБ")
	case errors.As(err, &te) && te.Field != "":
		return badRequest(te.Field, fmt.Sprintf("Поле %s имеет неверный тип", te.Field))
	}
	return badRequest("", "Тело запроса - не JSON по контракту")
}

func (s *Server) engine(name string) (*forecast.Engine, error) {
	if name == "" {
		return s.engines[0], nil
	}
	names := make([]string, len(s.engines))
	for i, e := range s.engines {
		if e.Name == name {
			return e, nil
		}
		names[i] = e.Name
	}
	return nil, badRequest("model", "Модель должна быть одной из: "+strings.Join(names, ", "))
}

func checkRoute(field string, r int) error {
	if slices.Contains(features.Routes, r) {
		return nil
	}
	all := make([]string, len(features.Routes))
	for i, x := range features.Routes {
		all[i] = strconv.Itoa(x)
	}
	return badRequest(field, fmt.Sprintf("Неизвестный маршрут %d, доступны: %s", r, strings.Join(all, ", ")))
}

func parseDate(field, s string) (time.Time, error) {
	t, err := time.Parse(time.DateOnly, s)
	if err != nil {
		return t, badRequest(field, "Дата должна быть в формате ГГГГ-ММ-ДД")
	}
	return t, nil
}

func inDomain(e *forecast.Engine, field string, t time.Time) error {
	b := e.Bundle
	msg := ""
	switch {
	case t.Before(b.From):
		msg = "доступно с " + b.From.Format(time.DateOnly)
	case t.After(b.To):
		msg = "доступно до " + b.To.Format(time.DateOnly)
	default:
		return nil
	}
	return &apiError{status: http.StatusBadRequest, Code: "out_of_domain", Field: field,
		Message: fmt.Sprintf("Дата %s вне области определения модели %s, %s", t.Format(time.DateOnly), e.Name, msg)}
}

func checkHorizon(h string) error {
	if slices.Contains(horizons, h) {
		return nil
	}
	return badRequest("horizon", "Горизонт - day, week, month или season")
}

func (s *Server) checkConditions(in []conditions.Input) error {
	if len(in) > maxConditions {
		return tooLarge("too_many_conditions", "conditions", fmt.Sprintf("Условий не больше %d, передано %d", maxConditions, len(in)))
	}
	var fe *conditions.FieldError
	if err := s.catalog.Check(in); errors.As(err, &fe) {
		return badRequest(fe.Field, fe.Message)
	} else if err != nil {
		return err
	}
	return nil
}

func (s *Server) parseForecast(w http.ResponseWriter, r *http.Request) (query, error) {
	var req forecastRequest
	if err := decode(w, r, &req); err != nil {
		return query{}, err
	}
	e, err := s.engine(req.Model)
	if err != nil {
		return query{}, err
	}
	q := query{engine: e, daily: req.Granularity == "day", horizon: req.Horizon, corridor: req.Corridor == nil || *req.Corridor, conditions: req.Conditions}
	for _, rt := range req.Routes {
		if err := checkRoute("routes", rt); err != nil {
			return query{}, err
		}
		if !slices.Contains(q.routes, rt) {
			q.routes = append(q.routes, rt)
		}
	}
	if len(q.routes) == 0 {
		q.routes = slices.Clone(features.Routes)
	}
	if q.from, err = parseDate("from", req.From); err != nil {
		return query{}, err
	}
	to, err := parseDate("to", req.To)
	if err != nil {
		return query{}, err
	}
	if err := inDomain(e, "from", q.from); err != nil {
		return query{}, err
	}
	if err := inDomain(e, "to", to); err != nil {
		return query{}, err
	}
	if to.Before(q.from) {
		return query{}, badRequest("to", "Конец периода раньше начала")
	}
	q.days = int(to.Sub(q.from).Hours()/24) + 1
	if req.Granularity != "hour" && req.Granularity != "day" {
		return query{}, badRequest("granularity", "Квант ряда - hour или day")
	}
	if err := checkHorizon(req.Horizon); err != nil {
		return query{}, err
	}
	if !q.daily && q.days > maxHourDays {
		return query{}, tooLarge("period_too_long", "granularity", fmt.Sprintf("По часам доступен период не больше %d суток, запрошено %d", maxHourDays, q.days))
	}
	if err := s.checkConditions(req.Conditions); err != nil {
		return query{}, err
	}
	if n := e.Rows(q.routes, q.days); n > maxRows {
		return query{}, tooLarge("too_many_rows", "to", fmt.Sprintf("Запрос требует %d строк модели, предел %d: сократите период или число маршрутов", n, maxRows))
	}
	return q, nil
}

func (s *Server) parseExplain(w http.ResponseWriter, r *http.Request) (query, int, error) {
	var req explainRequest
	if err := decode(w, r, &req); err != nil {
		return query{}, 0, err
	}
	e, err := s.engine(req.Model)
	if err != nil {
		return query{}, 0, err
	}
	if err := checkRoute("route", req.Route); err != nil {
		return query{}, 0, err
	}
	d, err := parseDate("date", req.Date)
	if err != nil {
		return query{}, 0, err
	}
	if err := inDomain(e, "date", d); err != nil {
		return query{}, 0, err
	}
	if req.Hour == nil || *req.Hour < 0 || *req.Hour > 23 {
		return query{}, 0, badRequest("hour", "Час - число от 0 до 23")
	}
	if req.Horizon == "" {
		req.Horizon = "day"
	}
	if err := checkHorizon(req.Horizon); err != nil {
		return query{}, 0, err
	}
	if err := s.checkConditions(req.Conditions); err != nil {
		return query{}, 0, err
	}
	return query{engine: e, routes: []int{req.Route}, from: d, days: 1, horizon: req.Horizon, conditions: req.Conditions}, *req.Hour, nil
}
