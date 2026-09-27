package features

import (
	"fmt"
	"math"
	"slices"
	"strconv"
)

type Row struct {
	Route int
	Hour  int
	Day
}

type Cell struct {
	Route int
	Day
}

type Feature struct {
	Name  string `json:"name"`
	Value any    `json:"value"`
	Kind  string `json:"kind"`
	Hash  *int32 `json:"hash"`
}

type Spec struct {
	Names  []string
	IsCat  []bool
	floats []func(Row) float32
	cats   []func(Row) string
}

var floatFeatures = map[string]func(Row) float32{
	"hour":              func(r Row) float32 { return float32(r.Hour) },
	"hour_sin":          func(r Row) float32 { return float32(math.Sin(r.hourAngle())) },
	"hour_cos":          func(r Row) float32 { return float32(math.Cos(r.hourAngle())) },
	"weekday_sin":       func(r Row) float32 { return float32(math.Sin(r.weekdayAngle())) },
	"weekday_cos":       func(r Row) float32 { return float32(math.Cos(r.weekdayAngle())) },
	"day_of_month_sin":  func(r Row) float32 { return float32(math.Sin(r.monthAngle())) },
	"day_of_month_cos":  func(r Row) float32 { return float32(math.Cos(r.monthAngle())) },
	"days_before_block": func(r Row) float32 { return float32(r.DaysBeforeBlock) },
	"days_after_block":  func(r Row) float32 { return float32(r.DaysAfterBlock) },
	"in_block":          func(r Row) float32 { return float32(b01(r.InBlock)) },
}

var catFeatures = map[string]func(Row) string{
	"route":                func(r Row) string { return strconv.Itoa(r.Route) },
	"is_holiday":           func(r Row) string { return strconv.Itoa(b01(r.IsHoliday)) },
	"is_weekend":           func(r Row) string { return strconv.Itoa(b01(r.IsWeekend)) },
	"is_short_working_day": func(r Row) string { return strconv.Itoa(b01(r.IsShortWorkingDay)) },
	"season":               func(r Row) string { return r.Season },
	"weekday":              func(r Row) string { return strconv.Itoa(r.Weekday) },
}

func NewSpec(names, catNames []string) (*Spec, error) {
	for _, n := range catNames {
		if !slices.Contains(names, n) {
			return nil, fmt.Errorf("категориальный признак %q не входит в список признаков", n)
		}
	}
	s := &Spec{Names: names}
	for _, n := range names {
		isCat := slices.Contains(catNames, n)
		s.IsCat = append(s.IsCat, isCat)
		if isCat {
			f, ok := catFeatures[n]
			if !ok {
				return nil, fmt.Errorf("неизвестный категориальный признак %q", n)
			}
			s.cats = append(s.cats, f)
			continue
		}
		f, ok := floatFeatures[n]
		if !ok {
			return nil, fmt.Errorf("неизвестный числовой признак %q", n)
		}
		s.floats = append(s.floats, f)
	}
	return s, nil
}

func (s *Spec) NumFloat() int { return len(s.floats) }

func (s *Spec) NumCat() int { return len(s.cats) }

func (s *Spec) Encode(r Row, fl []float32, cat []string) {
	for i, f := range s.floats {
		fl[i] = f(r)
	}
	for i, f := range s.cats {
		cat[i] = f(r)
	}
}

func (s *Spec) Values(r Row) []Feature {
	out := make([]Feature, 0, len(s.Names))
	fi, ci := 0, 0
	for i, n := range s.Names {
		if s.IsCat[i] {
			out = append(out, Feature{Name: n, Kind: "categorical", Value: s.cats[ci](r)})
			ci++
			continue
		}
		out = append(out, Feature{Name: n, Kind: "float", Value: float64(s.floats[fi](r))})
		fi++
	}
	return out
}

func (r Row) hourAngle() float64 { return 2 * math.Pi * float64(r.Hour) / 24 }

func (d Day) weekdayAngle() float64 { return 2 * math.Pi * float64(d.Weekday-1) / 7 }

func (d Day) monthAngle() float64 {
	return 2 * math.Pi * float64(d.DayOfMonth-1) / float64(d.DaysInMonth)
}

func b01(v bool) int {
	if v {
		return 1
	}
	return 0
}
