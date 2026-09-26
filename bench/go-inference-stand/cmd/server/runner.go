package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

type runner struct {
	base   string
	client *http.Client
}

type prediction struct {
	Values  [][]float64 `json:"values"`
	Rows    int         `json:"rows"`
	ModelMs float64     `json:"model_ms"`
}

type explanation struct {
	Fallback bool            `json:"fallback"`
	Input    json.RawMessage `json:"input"`
	History  *struct {
		Boardings   int64     `json:"boardings"`
		SameWeekday []float64 `json:"same_weekday"`
	} `json:"history"`
	Values  []float64 `json:"values"`
	ModelMs float64   `json:"model_ms"`
}

func newRunner(base string) *runner {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.MaxIdleConnsPerHost = 64
	return &runner{base: strings.TrimRight(base, "/"), client: &http.Client{Transport: t, Timeout: 20 * time.Second}}
}

func (c *runner) get(path string, q url.Values, out any) error {
	resp, err := c.client.Get(c.base + path + "?" + q.Encode())
	if err != nil {
		return fmt.Errorf("раннер недоступен: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		var e struct {
			Error string `json:"error"`
		}
		_ = json.NewDecoder(resp.Body).Decode(&e)
		return fmt.Errorf("раннер ответил %d: %s", resp.StatusCode, e.Error)
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

func (c *runner) predict(routes []int, from time.Time, days int) (prediction, error) {
	s := make([]string, len(routes))
	for i, r := range routes {
		s[i] = strconv.Itoa(r)
	}
	var p prediction
	q := url.Values{"route": {strings.Join(s, ",")}, "from": {from.Format("2006-01-02")}, "days": {strconv.Itoa(days)}}
	if err := c.get("/predict", q, &p); err != nil {
		return p, err
	}
	if len(p.Values) != len(routes) {
		return p, fmt.Errorf("раннер вернул %d маршрутов вместо %d", len(p.Values), len(routes))
	}
	for _, v := range p.Values {
		if len(v) != days*24 {
			return p, fmt.Errorf("раннер вернул %d часов вместо %d", len(v), days*24)
		}
	}
	return p, nil
}

func (c *runner) explain(route int, date time.Time) (explanation, error) {
	var e explanation
	q := url.Values{"route": {strconv.Itoa(route)}, "date": {date.Format("2006-01-02")}}
	if err := c.get("/explain", q, &e); err != nil {
		return e, err
	}
	if len(e.Values) != 24 {
		return e, fmt.Errorf("раннер вернул %d часов вместо 24", len(e.Values))
	}
	return e, nil
}

func (c *runner) model() (map[string]any, error) {
	var m map[string]any
	return m, c.get("/model", nil, &m)
}
