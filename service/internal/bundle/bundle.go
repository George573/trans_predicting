package bundle

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"time"
)

var horizons = []string{"day", "week", "month", "season"}

type Reference struct {
	From         string  `json:"from"`
	To           string  `json:"to"`
	Rows         int     `json:"rows"`
	RawTolerance float64 `json:"raw_tolerance"`
}

type Meta struct {
	Version     string    `json:"version"`
	Model       string    `json:"model"`
	Mode        string    `json:"mode"`
	TrainPeriod [2]string `json:"train_period"`
	Horizon     [2]string `json:"horizon"`
	WAPEScore   float64   `json:"wape_score"`
	Reference   Reference `json:"reference"`
}

type Calibration struct {
	Keys   []string                      `json:"keys"`
	Shrink float64                       `json:"shrink"`
	Clip   [2]float64                    `json:"clip"`
	K      map[string]map[string]float64 `json:"k"`
}

type Corridor struct {
	Quantiles [2]float64                       `json:"quantiles"`
	Bands     map[string][2]int                `json:"bands"`
	Hourly    map[string]map[string][2]float64 `json:"hourly"`
	Daily     map[string][2]float64            `json:"daily"`
	Widen     map[string]float64               `json:"widen"`
	band      [24]string
}

type Postprocess struct {
	Route5Share  float64                `json:"route5_share"`
	NewYearTable map[string][24]float64 `json:"new_year"`
}

type File struct {
	Name   string `json:"name"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
}

type Bundle struct {
	Dir         string
	Meta        Meta
	MetaRaw     map[string]any
	Calibration *Calibration
	Corridor    Corridor
	Postprocess Postprocess
	Files       []File
	From, To    time.Time
}

func Load(dir string) (*Bundle, error) {
	b := &Bundle{Dir: dir}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	for _, e := range entries {
		if !e.Type().IsRegular() {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			return nil, err
		}
		sum := sha256.Sum256(data)
		b.Files = append(b.Files, File{Name: e.Name(), Size: int64(len(data)), SHA256: hex.EncodeToString(sum[:])})
	}
	for _, name := range []string{"meta.json", "corridor.json", "postprocess.json", "reference.csv"} {
		if !slices.ContainsFunc(b.Files, func(f File) bool { return f.Name == name }) {
			return nil, fmt.Errorf("бандл %s неполный: нет %s", dir, name)
		}
	}
	for name, v := range map[string]any{"meta.json": &b.Meta, "corridor.json": &b.Corridor, "postprocess.json": &b.Postprocess} {
		if err := readJSON(filepath.Join(dir, name), v); err != nil {
			return nil, err
		}
	}
	if err := readJSON(filepath.Join(dir, "meta.json"), &b.MetaRaw); err != nil {
		return nil, err
	}
	if _, err := os.Stat(filepath.Join(dir, "calibration.json")); err == nil {
		b.Calibration = &Calibration{}
		if err := readJSON(filepath.Join(dir, "calibration.json"), b.Calibration); err != nil {
			return nil, err
		}
	}
	if err := b.check(); err != nil {
		return nil, fmt.Errorf("бандл %s: %w", dir, err)
	}
	return b, nil
}

func readJSON(path string, v any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, v); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}

func (b *Bundle) check() error {
	m := b.Meta
	if m.Version == "" {
		return errors.New("в meta.json нет версии")
	}
	if m.Mode != "service" {
		return fmt.Errorf("meta.json: режим %q, нужен service", m.Mode)
	}
	var err error
	if b.From, err = time.Parse(time.DateOnly, m.Horizon[0]); err != nil {
		return fmt.Errorf("meta.json: начало горизонта: %w", err)
	}
	if b.To, err = time.Parse(time.DateOnly, m.Horizon[1]); err != nil {
		return fmt.Errorf("meta.json: конец горизонта: %w", err)
	}
	if b.To.Before(b.From) {
		return fmt.Errorf("meta.json: горизонт %s - %s пуст", m.Horizon[0], m.Horizon[1])
	}
	if m.Reference.RawTolerance <= 0 {
		return errors.New("meta.json: допуск сверки с эталоном должен быть больше нуля")
	}
	if c := b.Calibration; c != nil {
		for route, byDay := range c.K {
			for day, k := range byDay {
				if k < c.Clip[0] || k > c.Clip[1] {
					return fmt.Errorf("calibration.json: коэффициент %g для маршрута %s, день %s вне [%g, %g]", k, route, day, c.Clip[0], c.Clip[1])
				}
			}
		}
	}
	if err := b.Corridor.check(); err != nil {
		return fmt.Errorf("corridor.json: %w", err)
	}
	if s := b.Postprocess.Route5Share; s <= 0 || s >= 1 {
		return fmt.Errorf("postprocess.json: доля маршрута 5 %g вне (0, 1)", s)
	}
	return nil
}

func (c *Corridor) check() error {
	for name, r := range c.Bands {
		if r[0] < 0 || r[1] > 23 || r[0] > r[1] {
			return fmt.Errorf("полоса %s: часы %d-%d вне 0-23", name, r[0], r[1])
		}
		for h := r[0]; h <= r[1]; h++ {
			if c.band[h] != "" {
				return fmt.Errorf("час %d входит в полосы %s и %s", h, c.band[h], name)
			}
			c.band[h] = name
		}
		if _, ok := c.Hourly["*"][name]; !ok {
			return fmt.Errorf("в hourly[\"*\"] нет полосы %s", name)
		}
	}
	for h, name := range c.band {
		if name == "" {
			return fmt.Errorf("час %d не входит ни в одну полосу", h)
		}
	}
	if _, ok := c.Daily["*"]; !ok {
		return errors.New("нет daily[\"*\"]")
	}
	for _, h := range horizons {
		if c.Widen[h] <= 0 {
			return fmt.Errorf("нет расширения коридора для горизонта %s", h)
		}
	}
	return nil
}

func (c *Calibration) Factor(route, weekday int) float64 {
	if c == nil {
		return 1
	}
	if k, ok := c.K[strconv.Itoa(route)][strconv.Itoa(weekday)]; ok {
		return k
	}
	return 1
}

func (c *Corridor) HourQ(route, hour int) [2]float64 {
	if q, ok := c.Hourly[strconv.Itoa(route)][c.band[hour]]; ok {
		return q
	}
	return c.Hourly["*"][c.band[hour]]
}

func (c *Corridor) DayQ(route int) [2]float64 {
	if q, ok := c.Daily[strconv.Itoa(route)]; ok {
		return q
	}
	return c.Daily["*"]
}

func (c *Corridor) Bounds(q [2]float64, horizon string, base float64) (float64, float64) {
	w := c.Widen[horizon]
	return max(0, base*(1-(1-q[0])*w)), base * (1 + (q[1]-1)*w)
}

func (p *Postprocess) NewYearFactor(d time.Time, hour int) float64 {
	if f, ok := p.NewYearTable[d.Format(time.DateOnly)]; ok {
		return f[hour]
	}
	return 1
}
