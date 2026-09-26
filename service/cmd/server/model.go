package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"slices"
	"time"
)

type network struct {
	ModelKind    string        `json:"model_kind"`
	Cutoff       string        `json:"cutoff"`
	ForecastDays int           `json:"forecast_days"`
	Routes       []int         `json:"routes"`
	Parameters   int           `json:"parameters"`
	Scale        float64       `json:"scale"`
	Checkpoint   string        `json:"checkpoint_sha256"`
	History      [][]float64   `json:"history"`
	Base         [][]float32   `json:"hidden_base"`
	Calendar     [][4]float32  `json:"hidden_calendar"`
	Lead         []float32     `json:"hidden_lead"`
	Output       [24][]float32 `json:"output_weight"`
	Bias         [24]float32   `json:"output_bias"`
	Check        []checkedDay  `json:"check"`
	first        time.Time
	cutoff       time.Time
}

type checkedDay struct {
	Route  int         `json:"route"`
	Date   string      `json:"date"`
	Values [24]float64 `json:"values"`
}

type headInput struct {
	RouteIndex    int     `json:"route_index"`
	Lead          int     `json:"lead"`
	WeekdaySin    float32 `json:"weekday_sin"`
	WeekdayCos    float32 `json:"weekday_cos"`
	DayOfMonthSin float32 `json:"day_of_month_sin"`
	DayOfMonthCos float32 `json:"day_of_month_cos"`
}

func loadNetwork(path string) (*network, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var n network
	if err := json.Unmarshal(b, &n); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if n.cutoff, err = time.Parse("2006-01-02", n.Cutoff); err != nil {
		return nil, fmt.Errorf("%s: cutoff: %w", path, err)
	}
	n.first = n.cutoff.AddDate(0, 0, -21)
	width := len(n.Lead)
	valid := len(n.Calendar) == width && len(n.Base) == len(n.Routes) && len(n.History) == len(n.Routes)
	for _, w := range n.Output {
		valid = valid && len(w) == width
	}
	for i := range n.Base {
		valid = valid && len(n.Base[i]) == width && len(n.History[i]) == 21*24
	}
	if !valid {
		return nil, fmt.Errorf("%s: размеры весов не сходятся", path)
	}
	all := n.predict(n.Routes, n.cutoff, n.ForecastDays)
	for _, c := range n.Check {
		d, err := time.Parse("2006-01-02", c.Date)
		i, day := slices.Index(n.Routes, c.Route), int(d.Sub(n.cutoff).Hours()/24)
		if err != nil || i < 0 || day < 0 || day >= n.ForecastDays {
			return nil, fmt.Errorf("%s: check: маршрут %d, %s вне модели", path, c.Route, c.Date)
		}
		got := all[i][day*24:]
		for h, want := range c.Values {
			if math.Abs(got[h]-want) > 1e-4*math.Abs(want)+1e-3 {
				return nil, fmt.Errorf("маршрут %d, %s, час %d: Go %.4f, torch %.4f", c.Route, c.Date, h, got[h], want)
			}
		}
	}
	return &n, nil
}

func (n *network) features(d time.Time) headInput {
	weekday := float64((int(d.Weekday())+6)%7) / 7 * 2 * math.Pi
	day := float64(d.Day()-1) / 31 * 2 * math.Pi
	return headInput{
		Lead:       int(d.Sub(n.cutoff).Hours()/24) + 1,
		WeekdaySin: float32(math.Sin(weekday)), WeekdayCos: float32(math.Cos(weekday)),
		DayOfMonthSin: float32(math.Sin(day)), DayOfMonthCos: float32(math.Cos(day)),
	}
}

func erf32(x float32) float32 {
	x = min(max(x, -4), 4)
	x2 := x * x
	p := x2*-2.72614225801306e-10 + 2.77068142495902e-08
	p = x2*p + -2.10102402082508e-06
	p = x2*p + -5.69250639462346e-05
	p = x2*p + -7.34990630326855e-04
	p = x2*p + -2.95459980854025e-03
	p = x2*p + -1.60960333262415e-02
	q := x2*-1.45660718464996e-05 + -2.13374055278905e-04
	q = x2*q + -1.68282697438203e-03
	q = x2*q + -7.37332916720468e-03
	q = x2*q + -1.42647390514189e-02
	return x * p / q
}

type row struct {
	dst []float64
	i   int
	d   time.Time
}

func (n *network) predict(routes []int, from time.Time, days int) [][]float64 {
	values := make([][]float64, len(routes))
	var rows []row
	for r, route := range routes {
		values[r] = make([]float64, days*24)
		if i := slices.Index(n.Routes, route); i >= 0 {
			for d := range days {
				rows = append(rows, row{values[r][d*24 : d*24+24], i, from.AddDate(0, 0, d)})
			}
		}
	}
	n.run(rows)
	return values
}

func (n *network) run(rows []row) {
	width := len(n.Lead)
	g := make([]float32, min(len(rows), 4)*width)
	for ; len(rows) >= 4; rows = rows[4:] {
		for b, r := range rows[:4] {
			n.hidden(g[b*width:(b+1)*width], r.i, r.d)
		}
		n.output4(rows[:4], g)
	}
	for _, r := range rows {
		n.hidden(g[:width], r.i, r.d)
		n.output(r.dst, g[:width])
	}
}

func (n *network) hidden(g []float32, i int, d time.Time) {
	in := n.features(d)
	step := float32(in.Lead-1) / 60
	base := n.Base[i][:len(g)]
	for j := range g {
		c := &n.Calendar[j]
		h := base[j] + c[0]*in.WeekdaySin + c[1]*in.WeekdayCos + c[2]*in.DayOfMonthSin + c[3]*in.DayOfMonthCos + n.Lead[j]*step
		g[j] = 0.5 * h * (1 + erf32(h*math.Sqrt2/2))
	}
}

func (n *network) softplus(v float32) float64 {
	x := float64(v)
	t := math.Exp(-math.Abs(x))
	p := t*-0.0031760570360347628 + 0.019542527353769396
	p = t*p + -0.056373613033429151
	p = t*p + 0.10543623819571624
	p = t*p + -0.15269667091305339
	p = t*p + 0.1966327426436095
	p = t*p + -0.24951616264643695
	p = t*p + 0.33329710499153697
	p = t*p + -0.49999892648076932
	p = t*p + 0.99999999465629363
	return (max(x, 0) + t*p) * n.Scale
}

func (n *network) output(dst []float64, g []float32) {
	for k, w := range n.Output {
		w = w[:len(g)]
		var a0, a1, a2, a3 float32
		j := 0
		for ; j+4 <= len(g); j += 4 {
			a0 += g[j] * w[j]
			a1 += g[j+1] * w[j+1]
			a2 += g[j+2] * w[j+2]
			a3 += g[j+3] * w[j+3]
		}
		for ; j < len(g); j++ {
			a0 += g[j] * w[j]
		}
		dst[k] = n.softplus(n.Bias[k] + (a0 + a1) + (a2 + a3))
	}
}

func (n *network) output4(rows []row, g []float32) {
	width := len(n.Lead)
	g0, g1, g2, g3 := g[:width], g[width:2*width], g[2*width:3*width], g[3*width:4*width]
	for k, w := range n.Output {
		w = w[:width]
		g0, g1, g2, g3 := g0[:len(w)], g1[:len(w)], g2[:len(w)], g3[:len(w)]
		var a0, a1, a2, a3, b0, b1, b2, b3 float32
		j := 0
		for ; j+2 <= len(w); j += 2 {
			w0, w1 := w[j], w[j+1]
			a0 += g0[j] * w0
			a1 += g1[j] * w0
			a2 += g2[j] * w0
			a3 += g3[j] * w0
			b0 += g0[j+1] * w1
			b1 += g1[j+1] * w1
			b2 += g2[j+1] * w1
			b3 += g3[j+1] * w1
		}
		for ; j < len(w); j++ {
			a0 += g0[j] * w[j]
			a1 += g1[j] * w[j]
			a2 += g2[j] * w[j]
			a3 += g3[j] * w[j]
		}
		b := n.Bias[k]
		rows[0].dst[k] = n.softplus(b + a0 + b0)
		rows[1].dst[k] = n.softplus(b + a1 + b1)
		rows[2].dst[k] = n.softplus(b + a2 + b2)
		rows[3].dst[k] = n.softplus(b + a3 + b3)
	}
}

func (n *network) rows(routes []int, days int) int {
	rows := 0
	for _, route := range routes {
		if slices.Contains(n.Routes, route) {
			rows += days
		}
	}
	return rows
}

func (n *network) history(i int, d time.Time, hour int) (boardings, sameWeekday float64) {
	same := 0
	for k, v := range n.History[i] {
		boardings += v
		if k%24 == hour && n.first.AddDate(0, 0, k/24).Weekday() == d.Weekday() {
			sameWeekday += v
			same++
		}
	}
	return boardings, sameWeekday / float64(same)
}
