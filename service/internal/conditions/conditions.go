package conditions

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"stand/internal/features"
)

var (
	types        = []string{"rain", "temperature", "event", "delay", "closure", "route_change"}
	classes      = []string{"operational", "structural"}
	horizons     = []string{"day", "week", "month", "season"}
	horizonNames = map[string]string{"day": "день", "week": "неделя", "month": "месяц", "season": "сезон"}
)

type Passport struct {
	Unit      string     `json:"unit"`
	Step      float64    `json:"step,omitempty"`
	Source    string     `json:"source"`
	SourceURL string     `json:"source_url,omitempty"`
	EffectPct float64    `json:"effect_pct"`
	CIPct     [2]float64 `json:"ci_pct"`
	Sample    int        `json:"sample,omitempty"`
	Horizons  []string   `json:"horizons"`
	Class     string     `json:"class"`
}

type Range struct {
	Min  float64 `json:"min"`
	Max  float64 `json:"max"`
	Step float64 `json:"step,omitempty"`
}

type Entry struct {
	Type     string              `json:"type"`
	Class    string              `json:"class"`
	Title    string              `json:"title"`
	Passport Passport            `json:"passport"`
	ByRoute  map[string]Passport `json:"by_route,omitempty"`
	Range    Range               `json:"range"`
	Auto     bool                `json:"auto"`
	Curve    [][2]float64        `json:"curve"`
}

type Catalog struct {
	Clamp      [2]float64 `json:"clamp"`
	Conditions []Entry    `json:"conditions"`
}

type Scope struct {
	Routes   []int    `json:"routes"`
	Dates    []string `json:"dates"`
	Weekdays []int    `json:"weekdays"`
	Hours    []int    `json:"hours"`
}

type Input struct {
	ID    string  `json:"id"`
	Type  string  `json:"type"`
	Value float64 `json:"value"`
	Scope Scope   `json:"scope"`
}

type FieldError struct {
	Field   string
	Message string
}

type Frame struct {
	Routes []int
	From   time.Time
	Days   int
	Base   [][]float64
}

type Context struct {
	Horizon string
	Daily   bool
}

type Applied struct {
	ID              string    `json:"id"`
	Type            string    `json:"type"`
	Applied         bool      `json:"applied"`
	Factor          float64   `json:"factor"`
	ContributionPct float64   `json:"contribution_pct"`
	Points          int       `json:"points"`
	Profile         []float64 `json:"profile"`
	Passport        Passport  `json:"passport"`
}

type Warning struct {
	Code        string  `json:"code"`
	ConditionID *string `json:"condition_id"`
	Message     string  `json:"message"`
}

type Result struct {
	Value      [][]float64
	Conditions []Applied
	Warnings   []Warning
}

type active struct {
	i       int
	factor  float64
	routeOK []bool
	dayOK   []bool
	h0, h1  int
}

type routeEffect struct {
	EffectPct float64    `json:"effect_pct"`
	CIPct     [2]float64 `json:"ci_pct"`
	Sample    int        `json:"sample"`
}

func (e *FieldError) Error() string { return e.Message }

func LoadCatalog(path string) (*Catalog, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var raw struct {
		Clamp      [2]float64 `json:"clamp"`
		Conditions []struct {
			Entry
			ByRoute map[string]routeEffect `json:"by_route"`
		} `json:"conditions"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	c := &Catalog{Clamp: raw.Clamp}
	for _, r := range raw.Conditions {
		e := r.Entry
		if len(r.ByRoute) > 0 {
			e.ByRoute = map[string]Passport{}
			for k, v := range r.ByRoute {
				p := e.Passport
				p.EffectPct, p.CIPct, p.Sample = v.EffectPct, v.CIPct, v.Sample
				e.ByRoute[k] = p
			}
		}
		c.Conditions = append(c.Conditions, e)
	}
	if err := c.Validate(); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return c, nil
}

func (c *Catalog) Validate() error {
	if c.Clamp[0] < 0 || c.Clamp[0] >= 1 || c.Clamp[1] <= 1 {
		return fmt.Errorf("коридор корректировки %v должен содержать 1 и не уходить ниже 0", c.Clamp)
	}
	seen := map[string]bool{}
	for _, e := range c.Conditions {
		if !slices.Contains(types, e.Type) || seen[e.Type] {
			return fmt.Errorf("тип условия %q неизвестен или повторяется", e.Type)
		}
		seen[e.Type] = true
	}
	if len(seen) != len(types) {
		return fmt.Errorf("в каталоге %d типов условий из %d", len(seen), len(types))
	}
	return c.validateEntries()
}

func (c *Catalog) validateEntries() error {
	for i := range c.Conditions {
		if err := c.Conditions[i].validate(); err != nil {
			return fmt.Errorf("%s: %w", c.Conditions[i].Type, err)
		}
	}
	return nil
}

func (e *Entry) validate() error {
	if !slices.Contains(classes, e.Class) || e.Passport.Class != e.Class {
		return fmt.Errorf("класс %q не совпадает с паспортом %q", e.Class, e.Passport.Class)
	}
	if e.Range.Min >= e.Range.Max {
		return fmt.Errorf("пустой диапазон %v", e.Range)
	}
	if len(e.Curve) < 2 {
		return errors.New("в кривой эффекта меньше двух точек")
	}
	for i, pt := range e.Curve {
		if i > 0 && pt[0] < e.Curve[i-1][0] {
			return errors.New("точки кривой должны идти по возрастанию значения")
		}
		if pt[1] < -100 {
			return errors.New("эффект кривой ниже -100%")
		}
	}
	if err := e.Passport.validate(e.Auto); err != nil {
		return err
	}
	for k, p := range e.ByRoute {
		if r, err := strconv.Atoi(k); err != nil || !slices.Contains(features.Routes, r) {
			return fmt.Errorf("неизвестный маршрут %q в by_route", k)
		}
		if err := p.validate(e.Auto); err != nil {
			return fmt.Errorf("маршрут %s: %w", k, err)
		}
	}
	return nil
}

func (p Passport) validate(auto bool) error {
	if p.Source == "" || p.Unit == "" {
		return errors.New("у паспорта нет источника или единицы")
	}
	if p.CIPct[0] > p.EffectPct || p.EffectPct > p.CIPct[1] {
		return fmt.Errorf("эффект %g вне интервала %v", p.EffectPct, p.CIPct)
	}
	if auto && p.CIPct[0] <= 0 && p.CIPct[1] >= 0 {
		return errors.New("условие с интервалом, включающим ноль, не может применяться автоматически")
	}
	if len(p.Horizons) == 0 {
		return errors.New("у паспорта нет допустимых горизонтов")
	}
	for _, h := range p.Horizons {
		if !slices.Contains(horizons, h) {
			return fmt.Errorf("неизвестный горизонт %q", h)
		}
	}
	return nil
}

func (c *Catalog) Entry(typ string) *Entry {
	for i := range c.Conditions {
		if c.Conditions[i].Type == typ {
			return &c.Conditions[i]
		}
	}
	return nil
}

func (e *Entry) Resolve(value float64, scopeRoutes []int) (float64, Passport) {
	p, scale := e.Passport, 1.0
	if len(scopeRoutes) == 1 && e.Passport.EffectPct != 0 {
		if rp, ok := e.ByRoute[strconv.Itoa(scopeRoutes[0])]; ok {
			p, scale = rp, rp.EffectPct/e.Passport.EffectPct
		}
	}
	return max(0, 1+interp(e.Curve, value)*scale/100), p
}

func (c *Catalog) Check(in []Input) error {
	ids := map[string]bool{}
	for i, x := range in {
		field := func(name string) string { return fmt.Sprintf("conditions[%d].%s", i, name) }
		if x.ID == "" || utf8.RuneCountInString(x.ID) > 32 {
			return &FieldError{field("id"), "У условия должен быть id длиной до 32 символов"}
		}
		if ids[x.ID] {
			return &FieldError{field("id"), fmt.Sprintf("Условие с id %q передано дважды", x.ID)}
		}
		ids[x.ID] = true
		e := c.Entry(x.Type)
		if e == nil {
			return &FieldError{field("type"), fmt.Sprintf("Неизвестный тип условия %.32q", x.Type)}
		}
		if x.Value < e.Range.Min || x.Value > e.Range.Max {
			return &FieldError{field("value"), fmt.Sprintf("Значение условия «%s» должно быть от %g до %g, передано %g", e.Title, e.Range.Min, e.Range.Max, x.Value)}
		}
		if n := len(x.Scope.Routes); n > 10 {
			return &FieldError{field("scope.routes"), fmt.Sprintf("В области условия не больше 10 маршрутов, передано %d", n)}
		}
		if n := len(x.Scope.Dates); n > 366 {
			return &FieldError{field("scope.dates"), fmt.Sprintf("В области условия не больше 366 дат, передано %d", n)}
		}
		if n := len(x.Scope.Weekdays); n > 7 {
			return &FieldError{field("scope.weekdays"), fmt.Sprintf("В области условия не больше 7 дней недели, передано %d", n)}
		}
		for _, r := range x.Scope.Routes {
			if !slices.Contains(features.Routes, r) {
				return &FieldError{field("scope.routes"), fmt.Sprintf("Неизвестный маршрут %d в области условия", r)}
			}
		}
		for _, d := range x.Scope.Dates {
			if _, err := time.Parse(time.DateOnly, d); err != nil {
				return &FieldError{field("scope.dates"), fmt.Sprintf("Дата %q в области условия не в формате ГГГГ-ММ-ДД", d)}
			}
		}
		for _, w := range x.Scope.Weekdays {
			if w < 1 || w > 7 {
				return &FieldError{field("scope.weekdays"), fmt.Sprintf("День недели %d вне 1...7", w)}
			}
		}
		if h := x.Scope.Hours; h != nil && (len(h) != 2 || h[0] < 0 || h[1] > 23 || h[0] > h[1]) {
			return &FieldError{field("scope.hours"), "Часы условия - пара [от, до] в пределах 0...23, где от не больше до"}
		}
	}
	return nil
}

func (x Input) Covers(route int, d time.Time, hour int) bool {
	h0, h1 := x.hourRange()
	return x.matchRoute(route) && x.matchDate(d) && hour >= h0 && hour <= h1
}

func (x Input) matchRoute(route int) bool {
	return x.Scope.Routes == nil || slices.Contains(x.Scope.Routes, route)
}

func (x Input) matchDate(d time.Time) bool {
	wd := int(d.Weekday())
	if wd == 0 {
		wd = 7
	}
	return (x.Scope.Dates == nil || slices.Contains(x.Scope.Dates, d.Format(time.DateOnly))) &&
		(x.Scope.Weekdays == nil || slices.Contains(x.Scope.Weekdays, wd))
}

func (x Input) hourRange() (int, int) {
	if x.Scope.Hours == nil {
		return 0, 23
	}
	return x.Scope.Hours[0], x.Scope.Hours[1]
}

func (c *Catalog) Apply(f Frame, ctx Context, in []Input) Result {
	res := Result{Value: make([][]float64, len(f.Routes)), Conditions: make([]Applied, len(in)), Warnings: []Warning{}}
	for r := range f.Routes {
		res.Value[r] = slices.Clone(f.Base[r])
	}
	var act []active
	for i, x := range in {
		e := c.Entry(x.Type)
		factor, p := e.Resolve(x.Value, x.Scope.Routes)
		res.Conditions[i] = Applied{ID: x.ID, Type: x.Type, Factor: 1, Passport: p}
		if !slices.Contains(e.Passport.Horizons, ctx.Horizon) {
			res.Warnings = append(res.Warnings, warning("condition_not_applicable_on_horizon", x.ID, e.notApplicable(ctx.Horizon)))
			continue
		}
		a := active{i: i, factor: factor, routeOK: make([]bool, len(f.Routes)), dayOK: make([]bool, f.Days)}
		a.h0, a.h1 = x.hourRange()
		for r, route := range f.Routes {
			a.routeOK[r] = x.matchRoute(route)
		}
		for d := range f.Days {
			a.dayOK[d] = x.matchDate(f.From.AddDate(0, 0, d))
		}
		points := 0
		for r := range f.Routes {
			for d := range f.Days {
				if !a.routeOK[r] || !a.dayOK[d] {
					continue
				}
				if ctx.Daily {
					points++
				} else {
					points += a.h1 - a.h0 + 1
				}
			}
		}
		if points == 0 {
			res.Warnings = append(res.Warnings, warning("condition_scope_empty", x.ID, fmt.Sprintf("Условие «%s» не попало ни в одну точку выбранного периода", e.Title)))
			continue
		}
		res.Conditions[i].Applied, res.Conditions[i].Factor, res.Conditions[i].Points = true, factor, points
		act = append(act, a)
	}
	if len(act) == 0 {
		return res
	}
	share := make([]float64, len(in))
	total, clamped := 0.0, 0
	var cover []int
	for r := range f.Routes {
		for d := range f.Days {
			for h := range 24 {
				cell := d*24 + h
				b := f.Base[r][cell]
				total += b
				cover = cover[:0]
				p := 1.0
				for k, a := range act {
					if a.routeOK[r] && a.dayOK[d] && h >= a.h0 && h <= a.h1 {
						cover = append(cover, k)
						p *= a.factor
					}
				}
				if len(cover) == 0 {
					continue
				}
				if q := min(max(p, c.Clamp[0]), c.Clamp[1]); q != p {
					clamped++
					p = q
				}
				res.Value[r][cell] = b * p
				allocate(share, act, cover, b*(p-1))
			}
		}
	}
	for _, a := range act {
		if total > 0 {
			res.Conditions[a.i].ContributionPct = 100 * share[a.i] / total
		}
		if len(f.Routes) == 1 && f.Days == 1 {
			prof := make([]float64, 24)
			for h := a.h0; h <= a.h1; h++ {
				prof[h] = 100 * (a.factor - 1)
			}
			res.Conditions[a.i].Profile = prof
		}
	}
	if clamped > 0 {
		res.Warnings = append(res.Warnings, warning("correction_clamped", "", fmt.Sprintf(
			"Произведение условий вышло за коридор корректировки [%.2f; %.2f] и ограничено в %d часах", c.Clamp[0], c.Clamp[1], clamped)))
	}
	return res
}

func (e *Entry) notApplicable(horizon string) string {
	if e.Class == "operational" && (horizon == "month" || horizon == "season") {
		return fmt.Sprintf("Оперативные условия на горизонте %s не применяются: измеренный эффект меньше ширины коридора", horizonNames[horizon])
	}
	names := make([]string, len(e.Passport.Horizons))
	for i, h := range e.Passport.Horizons {
		names[i] = horizonNames[h]
	}
	return fmt.Sprintf("Условие «%s» на горизонте %s не применяется, допустимые горизонты: %s", e.Title, horizonNames[horizon], strings.Join(names, ", "))
}

func allocate(share []float64, act []active, cover []int, delta float64) {
	if delta == 0 {
		return
	}
	var zeros []int
	logs := 0.0
	for _, k := range cover {
		if act[k].factor == 0 {
			zeros = append(zeros, k)
		} else {
			logs += math.Log(act[k].factor)
		}
	}
	switch {
	case len(zeros) > 0:
		for _, k := range zeros {
			share[act[k].i] += delta / float64(len(zeros))
		}
	case logs == 0:
		for _, k := range cover {
			share[act[k].i] += delta / float64(len(cover))
		}
	default:
		for _, k := range cover {
			share[act[k].i] += delta * math.Log(act[k].factor) / logs
		}
	}
}

func interp(c [][2]float64, v float64) float64 {
	if v <= c[0][0] {
		return c[0][1]
	}
	for i := 1; i < len(c); i++ {
		if v < c[i][0] {
			x0, y0, x1, y1 := c[i-1][0], c[i-1][1], c[i][0], c[i][1]
			return y0 + (y1-y0)*(v-x0)/(x1-x0)
		}
	}
	return c[len(c)-1][1]
}

func warning(code, id, msg string) Warning {
	w := Warning{Code: code, Message: msg}
	if id != "" {
		w.ConditionID = &id
	}
	return w
}
