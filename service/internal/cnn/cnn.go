package cnn

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"time"

	"stand/internal/features"
)

type Net struct {
	Model         string        `json:"model"`
	Cutoff        string        `json:"cutoff"`
	Days          int           `json:"days"`
	HistoryDays   int           `json:"history_days"`
	Routes        []int         `json:"routes"`
	Scale         float64       `json:"scale"`
	Parameters    int           `json:"parameters"`
	Features      []string      `json:"features"`
	HiddenBase    [][]float64   `json:"hidden_base"`
	HiddenRequest [][]float64   `json:"hidden_request"`
	Hidden2Weight [][]float64   `json:"hidden2_weight"`
	Hidden2Bias   []float64     `json:"hidden2_bias"`
	OutputWeight  [][]float64   `json:"output_weight"`
	OutputBias    []float64     `json:"output_bias"`
	Inputs        [][][]float64 `json:"inputs"`
	UsualInputs   [][][]float64 `json:"usual_inputs"`
	cutoff        time.Time
}

func Load(dir string) (*Net, error) {
	b, err := os.ReadFile(filepath.Join(dir, "head.json"))
	if err != nil {
		return nil, err
	}
	n := &Net{}
	if err := json.Unmarshal(b, n); err != nil {
		return nil, fmt.Errorf("head.json: %w", err)
	}
	if n.cutoff, err = time.Parse(time.DateOnly, n.Cutoff); err != nil {
		return nil, fmt.Errorf("head.json: отсечка: %w", err)
	}
	if err := n.check(); err != nil {
		return nil, fmt.Errorf("head.json: %w", err)
	}
	return n, nil
}

func (n *Net) Predict(cells []features.Cell, usual bool) ([]float64, error) {
	out := make([]float64, len(cells)*24)
	h1, h2 := make([]float64, len(n.Hidden2Bias)), make([]float64, len(n.Hidden2Bias))
	for i, c := range cells {
		r, d, err := n.index(c)
		if err != nil {
			return nil, err
		}
		in := n.Inputs[r][d]
		if usual {
			in = n.UsualInputs[r][d]
		}
		n.day(r, d, in, h1, h2, out[i*24:(i+1)*24])
	}
	return out, nil
}

func (n *Net) Ordinary(c features.Cell) bool {
	r, d, err := n.index(c)
	return err == nil && slices.Equal(n.Inputs[r][d], n.UsualInputs[r][d])
}

func (n *Net) Explain(c features.Cell, hour int) ([]features.Feature, float64, error) {
	r, d, err := n.index(c)
	if err != nil {
		return nil, 0, err
	}
	var out []features.Feature
	for i, v := range n.Inputs[r][d] {
		out = append(out, features.Feature{Name: n.Features[i], Kind: "float", Value: v})
	}
	out = append(out, features.Feature{Name: n.Features[len(n.Features)-1], Kind: "float", Value: float64(d + 1)})
	raw, err := n.Predict([]features.Cell{c}, false)
	if err != nil {
		return nil, 0, err
	}
	return out, raw[hour], nil
}

func (n *Net) Describe() string {
	return fmt.Sprintf("CNN, история %d суток до %s, %d параметров", n.HistoryDays, n.Cutoff, n.Parameters)
}

func (n *Net) index(c features.Cell) (int, int, error) {
	r := slices.Index(n.Routes, c.Route)
	if r < 0 {
		return 0, 0, fmt.Errorf("CNN не прогнозирует маршрут %d", c.Route)
	}
	d := int(c.Date.Sub(n.cutoff).Hours() / 24)
	if d < 0 || d >= n.Days {
		return 0, 0, fmt.Errorf("CNN прогнозирует %d суток от %s, запрошено %s", n.Days, n.Cutoff, c.Date.Format(time.DateOnly))
	}
	return r, d, nil
}

func (n *Net) day(r, d int, in, h1, h2, out []float64) {
	lead := float64(d) / 60
	for j, w := range n.HiddenRequest {
		s := n.HiddenBase[r][j] + w[len(in)]*lead
		for k, v := range in {
			s += w[k] * v
		}
		h1[j] = gelu(s)
	}
	for j, w := range n.Hidden2Weight {
		s := n.Hidden2Bias[j]
		for k, v := range h1 {
			s += w[k] * v
		}
		h2[j] = gelu(s)
	}
	for o, w := range n.OutputWeight {
		s := n.OutputBias[o]
		for k, v := range h2 {
			s += w[k] * v
		}
		out[o] = softplus(s) * n.Scale
	}
}

func gelu(x float64) float64 { return 0.5 * x * (1 + math.Erf(x/math.Sqrt2)) }

func softplus(x float64) float64 {
	if x > 20 {
		return x
	}
	return math.Log1p(math.Exp(x))
}

func (n *Net) check() error {
	w, k := len(n.Hidden2Bias), len(n.Features)-1
	bad := func(what string) error {
		return fmt.Errorf("размеры весов не сходятся: %s", what)
	}
	rows := func(m [][]float64, count, width int) bool {
		return len(m) == count && !slices.ContainsFunc(m, func(v []float64) bool { return len(v) != width })
	}
	cube := func(m [][][]float64) bool {
		return len(m) == len(n.Routes) && !slices.ContainsFunc(m, func(v [][]float64) bool { return !rows(v, n.Days, k) })
	}
	switch {
	case w == 0 || k < 1:
		return bad("нет скрытого слоя или входных признаков")
	case !rows(n.HiddenBase, len(n.Routes), w):
		return bad("hidden_base")
	case !rows(n.HiddenRequest, w, k+1):
		return bad("hidden_request")
	case !rows(n.Hidden2Weight, w, w):
		return bad("hidden2_weight")
	case !rows(n.OutputWeight, 24, w):
		return bad("output_weight")
	case len(n.OutputBias) != 24:
		return bad("output_bias")
	case !cube(n.Inputs):
		return bad("inputs")
	case !cube(n.UsualInputs):
		return bad("usual_inputs")
	case n.Scale <= 0:
		return bad("scale")
	}
	return nil
}
