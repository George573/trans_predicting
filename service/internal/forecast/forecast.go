package forecast

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"time"

	"stand/internal/bundle"
	"stand/internal/features"
)

type Model interface {
	Predict(cells []features.Cell, usual bool) ([]float64, error)
	Ordinary(c features.Cell) bool
	Explain(c features.Cell, hour int) ([]features.Feature, float64, error)
	Describe() string
}

type Engine struct {
	Name   string
	Bundle *bundle.Bundle
	Cal    *features.Calendar
	Model  Model
}

type Grid struct {
	Routes   []int
	From     time.Time
	Days     int
	Base     [][]float64
	Usual    [][]float64
	Rows     int
	Features time.Duration
	Model    time.Duration
}

type Check struct {
	Rows     int
	MaxRaw   float64
	MaxFinal float64
}

func New(name string, b *bundle.Bundle, cal *features.Calendar, m Model) (*Engine, error) {
	for _, d := range []time.Time{b.From, b.To} {
		if _, ok := cal.Day(d); !ok {
			return nil, fmt.Errorf("%s: производственный календарь не покрывает %s", name, d.Format(time.DateOnly))
		}
	}
	return &Engine{Name: name, Bundle: b, Cal: cal, Model: m}, nil
}

func (e *Engine) Rows(routes []int, days int) int {
	return len(modeled(routes)) * days * 24
}

func (e *Engine) Grid(routes []int, from time.Time, days int) (*Grid, error) {
	t0 := time.Now()
	mod := modeled(routes)
	info := make([]features.Day, days)
	for d := range days {
		day, ok := e.Cal.Day(from.AddDate(0, 0, d))
		if !ok {
			return nil, fmt.Errorf("нет производственного календаря на %s", from.AddDate(0, 0, d).Format(time.DateOnly))
		}
		info[d] = day
	}
	cells := make([]features.Cell, 0, len(mod)*days)
	for _, r := range mod {
		for d := range days {
			cells = append(cells, features.Cell{Route: r, Day: info[d]})
		}
	}
	var odd []int
	var oddCells []features.Cell
	for i, c := range cells {
		if !e.Model.Ordinary(c) {
			odd, oddCells = append(odd, i), append(oddCells, c)
		}
	}
	t1 := time.Now()
	raw, err := e.Model.Predict(cells, false)
	if err != nil {
		return nil, err
	}
	usualRaw, err := e.Model.Predict(oddCells, true)
	if err != nil {
		return nil, err
	}
	t2 := time.Now()
	cal, usual := map[int][]float64{}, map[int][]float64{}
	for i, r := range mod {
		c := make([]float64, days*24)
		for d := range days {
			k := e.Bundle.Calibration.Factor(r, info[d].Weekday)
			for h := range 24 {
				c[d*24+h] = max(raw[(i*days+d)*24+h], 0) * k
			}
		}
		cal[r], usual[r] = c, slices.Clone(c)
	}
	for j, ci := range odd {
		r, d := mod[ci/days], ci%days
		k := e.Bundle.Calibration.Factor(r, info[d].Weekday)
		for h := range 24 {
			usual[r][d*24+h] = max(usualRaw[j*24+h], 0) * k
		}
	}
	if slices.Contains(routes, 5) {
		cal[5], usual[5] = e.city(cal, days), e.city(usual, days)
	}
	g := &Grid{Routes: routes, From: from, Days: days, Rows: 24 * (len(cells) + len(oddCells)), Features: t1.Sub(t0), Model: t2.Sub(t1)}
	for _, r := range routes {
		b := slices.Clone(cal[r])
		for d := range days {
			for h := range 24 {
				b[d*24+h] *= e.Bundle.Postprocess.NewYearFactor(info[d].Date, h)
			}
		}
		g.Base, g.Usual = append(g.Base, b), append(g.Usual, usual[r])
	}
	return g, nil
}

func (e *Engine) city(byRoute map[int][]float64, days int) []float64 {
	s := e.Bundle.Postprocess.Route5Share
	out := make([]float64, days*24)
	for _, r := range features.Routes {
		if v, ok := byRoute[r]; ok && r != 5 {
			for i, x := range v {
				out[i] += x
			}
		}
	}
	for i := range out {
		out[i] *= s / (1 - s)
	}
	return out
}

func (g *Grid) Corridor(c *bundle.Corridor, horizon string, daily bool) ([][]float64, [][]float64) {
	lo, hi := make([][]float64, len(g.Routes)), make([][]float64, len(g.Routes))
	for i, r := range g.Routes {
		if daily {
			for _, b := range Daily(g.Base[i]) {
				l, h := c.Bounds(c.DayQ(r), horizon, b)
				lo[i], hi[i] = append(lo[i], l), append(hi[i], h)
			}
			continue
		}
		for cell, b := range g.Base[i] {
			l, h := c.Bounds(c.HourQ(r, cell%24), horizon, b)
			lo[i], hi[i] = append(lo[i], l), append(hi[i], h)
		}
	}
	return lo, hi
}

func Daily(v []float64) []float64 {
	out := make([]float64, len(v)/24)
	for i, x := range v {
		out[i/24] += x
	}
	return out
}

func modeled(routes []int) []int {
	var out []int
	for _, r := range routes {
		if r != 5 && !slices.Contains(out, r) {
			out = append(out, r)
		}
	}
	if slices.Contains(routes, 5) {
		for _, r := range features.Routes {
			if r != 5 && !slices.Contains(out, r) {
				out = append(out, r)
			}
		}
	}
	return out
}

func (e *Engine) Verify() (Check, error) {
	path := filepath.Join(e.Bundle.Dir, "reference.csv")
	f, err := os.Open(path)
	if err != nil {
		return Check{}, err
	}
	defer f.Close()
	recs, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return Check{}, fmt.Errorf("%s: %w", path, err)
	}
	if len(recs) == 0 || !slices.Equal(recs[0], []string{"route", "date", "hour", "raw", "final"}) {
		return Check{}, fmt.Errorf("%s: заголовок должен быть route,date,hour,raw,final", path)
	}
	recs = recs[1:]
	if len(recs) == 0 || len(recs)%24 != 0 {
		return Check{}, fmt.Errorf("%s: %d строк, нужно кратное 24", path, len(recs))
	}
	type ref struct {
		route, hour int
		date        time.Time
		raw         string
		final       float64
	}
	rows := make([]ref, len(recs))
	var from, to time.Time
	for i, r := range recs {
		x := &rows[i]
		x.route, err = strconv.Atoi(r[0])
		if err == nil {
			x.date, err = time.Parse(time.DateOnly, r[1])
		}
		if err == nil {
			x.hour, err = strconv.Atoi(r[2])
		}
		if err == nil {
			x.final, err = strconv.ParseFloat(r[4], 64)
		}
		if err != nil {
			return Check{}, fmt.Errorf("%s, строка %d: %w", path, i+2, err)
		}
		x.raw = r[3]
		if !slices.Contains(features.Routes, x.route) {
			return Check{}, fmt.Errorf("%s, строка %d: неизвестный маршрут %d", path, i+2, x.route)
		}
		if x.hour != i%24 || i%24 > 0 && (x.route != rows[i-1].route || !x.date.Equal(rows[i-1].date)) {
			return Check{}, fmt.Errorf("%s, строка %d: строки должны идти по маршруту и дате, 24 часа подряд", path, i+2)
		}
		if from.IsZero() || x.date.Before(from) {
			from = x.date
		}
		if x.date.After(to) {
			to = x.date
		}
	}
	var cells []features.Cell
	var rawRows []ref
	for i := 0; i < len(rows); i += 24 {
		if rows[i].raw == "" {
			continue
		}
		day, ok := e.Cal.Day(rows[i].date)
		if !ok {
			return Check{}, fmt.Errorf("нет производственного календаря на %s", rows[i].date.Format(time.DateOnly))
		}
		cells = append(cells, features.Cell{Route: rows[i].route, Day: day})
		rawRows = append(rawRows, rows[i:i+24]...)
	}
	var c Check
	raw, err := e.Model.Predict(cells, false)
	if err != nil {
		return Check{}, err
	}
	tol := e.Bundle.Meta.Reference.RawTolerance
	for i, r := range rawRows {
		want, err := strconv.ParseFloat(r.raw, 64)
		if err != nil {
			return Check{}, fmt.Errorf("%s: маршрут %d, %s %d ч: %w", path, r.route, r.date.Format(time.DateOnly), r.hour, err)
		}
		diff := math.Abs(raw[i] - want)
		if diff > tol*max(1, math.Abs(want)) {
			return Check{}, fmt.Errorf("сырой выход модели расходится с эталоном: маршрут %d, %s %d ч: Go %.9g, эталон %.9g", r.route, r.date.Format(time.DateOnly), r.hour, raw[i], want)
		}
		c.MaxRaw = max(c.MaxRaw, diff)
	}
	var routes []int
	for _, r := range features.Routes {
		if slices.ContainsFunc(rows, func(x ref) bool { return x.route == r }) {
			routes = append(routes, r)
		}
	}
	days := int(to.Sub(from).Hours()/24) + 1
	g, err := e.Grid(routes, from, days)
	if err != nil {
		return Check{}, err
	}
	for _, r := range rows {
		got := g.Base[slices.Index(routes, r.route)][int(r.date.Sub(from).Hours()/24)*24+r.hour]
		if math.Abs(math.Round(got)-math.Round(r.final)) > 1 {
			return Check{}, fmt.Errorf("прогноз после постобработки расходится с эталоном: маршрут %d, %s %d ч: Go %.9g, эталон %.9g", r.route, r.date.Format(time.DateOnly), r.hour, got, r.final)
		}
		c.MaxFinal = max(c.MaxFinal, math.Abs(got-r.final))
		c.Rows++
	}
	return c, nil
}
