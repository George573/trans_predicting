package api

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"slices"
	"time"

	"stand/internal/conditions"
	"stand/internal/features"
	"stand/internal/forecast"
)

type computed struct {
	q                  query
	grid               *forecast.Grid
	res                conditions.Result
	lo, hi             [][]float64
	parse, corrections time.Duration
}

type seriesOut struct {
	Route     int     `json:"route"`
	Base      []int64 `json:"base"`
	Value     []int64 `json:"value"`
	Usual     []int64 `json:"usual"`
	Lo        []int64 `json:"lo,omitempty"`
	Hi        []int64 `json:"hi,omitempty"`
	BaseTotal int64   `json:"base_total"`
	Total     int64   `json:"total"`
	PeakIndex int     `json:"peak_index"`
}

type timings struct {
	Parse       float64 `json:"parse"`
	Features    float64 `json:"features"`
	Model       float64 `json:"model"`
	Corrections float64 `json:"corrections"`
}

type metaOut struct {
	Rows    int     `json:"rows"`
	Model   string  `json:"model"`
	Timings timings `json:"timings"`
}

type forecastOut struct {
	Start      string               `json:"start"`
	Step       string               `json:"step"`
	Bundle     string               `json:"bundle"`
	Series     []seriesOut          `json:"series"`
	Conditions []conditions.Applied `json:"conditions"`
	Warnings   []conditions.Warning `json:"warnings"`
	Meta       metaOut              `json:"meta"`
}

type stepOut struct {
	Step   string   `json:"step"`
	Detail string   `json:"detail,omitempty"`
	Factor *float64 `json:"factor"`
	Value  float64  `json:"value"`
}

type explainOut struct {
	Model      string               `json:"model"`
	Route      int                  `json:"route"`
	Date       string               `json:"date"`
	Hour       int                  `json:"hour"`
	Calendar   map[string]any       `json:"calendar"`
	Features   []features.Feature   `json:"features"`
	Steps      []stepOut            `json:"steps"`
	Value      int64                `json:"value"`
	Conditions []conditions.Applied `json:"conditions"`
	Warnings   []conditions.Warning `json:"warnings"`
}

func (s *Server) compute(q query, parse time.Duration) (*computed, error) {
	g, err := q.engine.Grid(q.routes, q.from, q.days)
	if err != nil {
		return nil, err
	}
	t := time.Now()
	c := &computed{q: q, grid: g, parse: parse}
	c.res = s.catalog.Apply(conditions.Frame{Routes: q.routes, From: q.from, Days: q.days, Base: g.Base}, conditions.Context{Horizon: q.horizon, Daily: q.daily}, q.conditions)
	if slices.Contains(q.routes, 5) {
		c.res.Warnings = append(c.res.Warnings, conditions.Warning{Code: "route_without_history", Message: "Маршрут 5: истории нет, значения получены городским профилем"})
	}
	if q.corridor {
		c.lo, c.hi = g.Corridor(&q.engine.Bundle.Corridor, q.horizon, q.daily)
	}
	c.corrections = time.Since(t)
	return c, nil
}

func (c *computed) series() []seriesOut {
	total := func(v []float64) int64 {
		sum := 0.0
		for _, x := range v {
			sum += x
		}
		return int64(math.Round(sum))
	}
	out := make([]seriesOut, len(c.q.routes))
	for i, r := range c.q.routes {
		base, value, usual := c.grid.Base[i], c.res.Value[i], c.grid.Usual[i]
		if c.q.daily {
			base, value, usual = forecast.Daily(base), forecast.Daily(value), forecast.Daily(usual)
		}
		s := seriesOut{Route: r, Base: ints(base), Value: ints(value), Usual: ints(usual), BaseTotal: total(base), Total: total(value)}
		if c.lo != nil {
			s.Lo, s.Hi = ints(c.lo[i]), ints(c.hi[i])
		}
		for j, v := range s.Value {
			if v > s.Value[s.PeakIndex] {
				s.PeakIndex = j
			}
		}
		out[i] = s
	}
	return out
}

func (s *Server) forecast(w http.ResponseWriter, r *http.Request) {
	t0 := time.Now()
	q, err := s.parseForecast(w, r)
	if err != nil {
		s.fail(w, err)
		return
	}
	c, err := s.compute(q, time.Since(t0))
	if err != nil {
		s.fail(w, err)
		return
	}
	step := "1h"
	if q.daily {
		step = "1d"
	}
	tm := timings{ms(c.parse), ms(c.grid.Features), ms(c.grid.Model), ms(c.corrections)}
	out := forecastOut{
		Start: q.from.Format(time.DateOnly) + "T00:00", Step: step, Bundle: q.engine.Bundle.Meta.Version,
		Series: c.series(), Conditions: rounded(c.res.Conditions), Warnings: c.res.Warnings,
		Meta: metaOut{Rows: c.grid.Rows, Model: q.engine.Bundle.Meta.Model, Timings: tm},
	}
	t := time.Now()
	b, err := json.Marshal(out)
	if err != nil {
		s.fail(w, err)
		return
	}
	w.Header().Set("Server-Timing", fmt.Sprintf("parse;dur=%g, features;dur=%g, model;dur=%g, corrections;dur=%g, json;dur=%g",
		tm.Parse, tm.Features, tm.Model, tm.Corrections, ms(time.Since(t))))
	w.Header().Set("Content-Type", jsonType)
	w.Write(b)
	s.stats.record(time.Since(t0), c.grid.Rows)
}

func (s *Server) explain(w http.ResponseWriter, r *http.Request) {
	t0 := time.Now()
	q, hour, err := s.parseExplain(w, r)
	if err != nil {
		s.fail(w, err)
		return
	}
	c, err := s.compute(q, time.Since(t0))
	if err != nil {
		s.fail(w, err)
		return
	}
	e, route, date := q.engine, q.routes[0], q.from
	day, _ := e.Cal.Day(date)
	ny := e.Bundle.Postprocess.NewYearFactor(date, hour)
	feats := []features.Feature{}
	var steps []stepOut
	var running float64
	if route == 5 {
		running = c.grid.Base[0][hour] / ny
		steps = append(steps, stepOut{Step: "маршрут 5", Value: running,
			Detail: fmt.Sprintf("истории нет: городской профиль, доля %.2f%% посадок сети", 100*e.Bundle.Postprocess.Route5Share)})
	} else {
		f, raw, err := e.Model.Explain(features.Cell{Route: route, Day: day}, hour)
		if err != nil {
			s.fail(w, err)
			return
		}
		feats, running = f, max(raw, 0)
		steps = append(steps,
			stepOut{Step: "сырой выход модели", Detail: e.Bundle.Meta.Model + ": " + e.Model.Describe(), Value: raw},
			stepOut{Step: "обрезка нуля", Value: running})
		if e.Bundle.Calibration != nil {
			k := e.Bundle.Calibration.Factor(route, day.Weekday)
			running *= k
			steps = append(steps, stepOut{Step: "калибровка", Detail: "множитель маршрут x день недели", Factor: &k, Value: running})
		}
	}
	if ny != 1 {
		running *= ny
		steps = append(steps, stepOut{Step: "новогодний блок", Detail: "ручной множитель на дату и час", Factor: &ny, Value: running})
	}
	for i, x := range q.conditions {
		a := c.res.Conditions[i]
		st := stepOut{Step: "условие " + x.ID, Value: running}
		switch {
		case !a.Applied:
			for _, wn := range c.res.Warnings {
				if wn.ConditionID != nil && *wn.ConditionID == x.ID {
					st.Detail = wn.Message
				}
			}
		case !x.Covers(route, date, hour):
			st.Detail = "точка вне области условия"
		default:
			running *= a.Factor
			st.Factor, st.Value = &a.Factor, running
			st.Detail = s.catalog.Entry(x.Type).Title + ": " + a.Passport.Source
		}
		steps = append(steps, st)
	}
	if final := c.res.Value[0][hour]; math.Abs(final-running) > 1e-9*max(1, math.Abs(running)) {
		f := final / running
		steps = append(steps, stepOut{Step: "коридор корректировки", Factor: &f, Value: final})
	}
	writeJSON(w, http.StatusOK, explainOut{
		Model: e.Name, Route: route, Date: date.Format(time.DateOnly), Hour: hour,
		Calendar: map[string]any{
			"date": date.Format(time.DateOnly), "weekday": day.Weekday, "season": day.Season,
			"day_of_month": day.DayOfMonth, "days_in_month": day.DaysInMonth, "is_holiday": day.IsHoliday,
			"is_weekend": day.IsWeekend, "is_short_working_day": day.IsShortWorkingDay,
			"days_before_block": day.DaysBeforeBlock, "days_after_block": day.DaysAfterBlock, "in_block": day.InBlock,
		},
		Features: feats, Steps: steps, Value: int64(math.Round(c.res.Value[0][hour])),
		Conditions: rounded(c.res.Conditions), Warnings: c.res.Warnings,
	})
	s.stats.record(time.Since(t0), c.grid.Rows)
}
