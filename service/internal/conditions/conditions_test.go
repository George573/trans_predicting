package conditions

import (
	"errors"
	"math"
	"slices"
	"strings"
	"testing"
	"time"
)

var day0 = time.Date(2025, 11, 10, 0, 0, 0, 0, time.UTC)

func catalog() *Catalog {
	op := Passport{Unit: "мм/ч", Source: "тест", EffectPct: -10, CIPct: [2]float64{-20, -5}, Horizons: []string{"day", "week"}, Class: "operational"}
	ev := Passport{Unit: "%", Source: "тест", EffectPct: 0, CIPct: [2]float64{0, 0}, Horizons: []string{"day", "week"}, Class: "operational"}
	st := Passport{Unit: "доля", Source: "тест", EffectPct: -100, CIPct: [2]float64{-100, -100}, Horizons: []string{"day", "week", "month", "season"}, Class: "structural"}
	route7 := op
	route7.EffectPct, route7.CIPct = -5, [2]float64{-8, -2}
	return &Catalog{Clamp: [2]float64{0, 1.5}, Conditions: []Entry{
		{Type: "rain", Class: "operational", Title: "Дождь", Passport: op, ByRoute: map[string]Passport{"7": route7}, Range: Range{Min: 0, Max: 5}, Curve: [][2]float64{{0, 0}, {1, -10}}},
		{Type: "event", Class: "operational", Title: "Событие", Passport: ev, Range: Range{Min: -50, Max: 50}, Curve: [][2]float64{{-50, -50}, {50, 50}}},
		{Type: "closure", Class: "structural", Title: "Отмена", Passport: st, Range: Range{Min: 0, Max: 1}, Curve: [][2]float64{{0, 0}, {1, -100}}},
	}}
}

func frame(routes []int, days int) Frame {
	f := Frame{Routes: routes, From: day0, Days: days}
	for r := range routes {
		row := make([]float64, days*24)
		for i := range row {
			row[i] = float64(10 + 100*r + i%24)
		}
		f.Base = append(f.Base, row)
	}
	return f
}

func hours(a, b int) Scope { return Scope{Hours: []int{a, b}} }

func sum(v [][]float64) float64 {
	s := 0.0
	for _, r := range v {
		for _, x := range r {
			s += x
		}
	}
	return s
}

func near(a, b float64) bool { return math.Abs(a-b) <= 1e-9*max(1, math.Abs(b)) }

func TestEmptyScenarioKeepsBase(t *testing.T) {
	f := frame([]int{7, 11}, 2)
	res := catalog().Apply(f, Context{Horizon: "day"}, nil)
	if !slices.EqualFunc(res.Value, f.Base, slices.Equal) || len(res.Conditions) != 0 || len(res.Warnings) != 0 {
		t.Fatalf("empty scenario changed the forecast: %+v", res)
	}
	if res.Conditions == nil || res.Warnings == nil {
		t.Fatal("conditions and warnings must be empty arrays, not nil")
	}
}

func TestOutOfScopeChangesNothing(t *testing.T) {
	f := frame([]int{1, 11}, 1)
	res := catalog().Apply(f, Context{Horizon: "day"}, []Input{{ID: "c1", Type: "rain", Value: 1, Scope: Scope{Routes: []int{7}}}})
	a := res.Conditions[0]
	if a.Applied || a.Points != 0 || a.Factor != 1 || !slices.EqualFunc(res.Value, f.Base, slices.Equal) {
		t.Fatalf("out of scope condition applied: %+v", a)
	}
	if len(res.Warnings) != 1 || res.Warnings[0].Code != "condition_scope_empty" || *res.Warnings[0].ConditionID != "c1" {
		t.Fatalf("warnings %+v", res.Warnings)
	}
}

func TestScopeLimitsCells(t *testing.T) {
	f := frame([]int{7, 11}, 2)
	s := hours(16, 22)
	s.Routes, s.Dates = []int{7}, []string{"2025-11-11"}
	res := catalog().Apply(f, Context{Horizon: "day"}, []Input{{ID: "c1", Type: "rain", Value: 1, Scope: s}})
	a := res.Conditions[0]
	if !a.Applied || a.Points != 7 || !near(a.Factor, 0.95) {
		t.Fatalf("applied %+v", a)
	}
	for r := range f.Routes {
		for cell, b := range f.Base[r] {
			want := b
			if r == 0 && cell/24 == 1 && cell%24 >= 16 && cell%24 <= 22 {
				want = b * 0.95
			}
			if !near(res.Value[r][cell], want) {
				t.Fatalf("route %d cell %d: %v want %v", r, cell, res.Value[r][cell], want)
			}
		}
	}
}

func TestProductDoesNotDependOnOrder(t *testing.T) {
	f := frame([]int{7}, 1)
	a := Input{ID: "a", Type: "rain", Value: 0.5, Scope: hours(8, 20)}
	b := Input{ID: "b", Type: "event", Value: 20, Scope: hours(12, 23)}
	ab := catalog().Apply(f, Context{Horizon: "week"}, []Input{a, b})
	ba := catalog().Apply(f, Context{Horizon: "week"}, []Input{b, a})
	for cell := range f.Base[0] {
		if !near(ab.Value[0][cell], ba.Value[0][cell]) {
			t.Fatalf("cell %d: %v vs %v", cell, ab.Value[0][cell], ba.Value[0][cell])
		}
	}
	if !near(ab.Value[0][15], f.Base[0][15]*0.95*1.2) {
		t.Fatalf("cell 15 is not the product: %v", ab.Value[0][15])
	}
}

func TestInapplicableHorizonKeepsForecast(t *testing.T) {
	f := frame([]int{7}, 3)
	in := []Input{{ID: "c1", Type: "rain", Value: 1}, {ID: "c2", Type: "closure", Value: 0.5, Scope: hours(22, 23)}}
	res := catalog().Apply(f, Context{Horizon: "month", Daily: true}, in)
	if res.Conditions[0].Applied || res.Conditions[0].Factor != 1 {
		t.Fatalf("operational condition applied on month: %+v", res.Conditions[0])
	}
	w := res.Warnings[0]
	if w.Code != "condition_not_applicable_on_horizon" || *w.ConditionID != "c1" || w.Message != "Оперативные условия на горизонте месяц не применяются: измеренный эффект меньше ширины коридора" {
		t.Fatalf("warning %+v", w)
	}
	if !res.Conditions[1].Applied || res.Conditions[1].Points != 3 {
		t.Fatalf("structural condition must apply on month, daily points = days: %+v", res.Conditions[1])
	}
	if !near(res.Value[0][10], f.Base[0][10]) || !near(res.Value[0][22], f.Base[0][22]*0.5) {
		t.Fatal("only the closure hours may change")
	}
}

func TestContributionsAddUpToTotal(t *testing.T) {
	f := frame([]int{7, 11}, 2)
	in := []Input{
		{ID: "r", Type: "rain", Value: 2, Scope: hours(6, 20)},
		{ID: "e", Type: "event", Value: 30, Scope: hours(10, 23)},
		{ID: "x", Type: "closure", Value: 0.4, Scope: Scope{Routes: []int{11}, Hours: []int{18, 23}}},
	}
	res := catalog().Apply(f, Context{Horizon: "day"}, in)
	total := 0.0
	for _, a := range res.Conditions {
		total += a.ContributionPct
	}
	if want := 100 * (sum(res.Value)/sum(f.Base) - 1); !near(total, want) {
		t.Fatalf("contributions %v, total change %v", total, want)
	}
}

func TestClampIsWarnedNotSilent(t *testing.T) {
	f := frame([]int{7}, 1)
	in := []Input{{ID: "a", Type: "event", Value: 50, Scope: hours(8, 9)}, {ID: "b", Type: "event", Value: 50, Scope: hours(8, 9)}}
	res := catalog().Apply(f, Context{Horizon: "day"}, in)
	if !near(res.Value[0][8], f.Base[0][8]*1.5) {
		t.Fatalf("product not clamped: %v", res.Value[0][8])
	}
	w := res.Warnings[len(res.Warnings)-1]
	if w.Code != "correction_clamped" || w.ConditionID != nil || !strings.HasSuffix(w.Message, "ограничено в 2 часах") {
		t.Fatalf("warnings %+v", res.Warnings)
	}
	if total := res.Conditions[0].ContributionPct + res.Conditions[1].ContributionPct; !near(total, 100*(sum(res.Value)/sum(f.Base)-1)) {
		t.Fatal("contributions ignore the clamp")
	}
}

func TestFullClosureZeroesScope(t *testing.T) {
	f := frame([]int{7}, 1)
	res := catalog().Apply(f, Context{Horizon: "season"}, []Input{{ID: "x", Type: "closure", Value: 1}})
	if sum(res.Value) != 0 || !near(res.Conditions[0].ContributionPct, -100) {
		t.Fatalf("full closure: sum %v contribution %v", sum(res.Value), res.Conditions[0].ContributionPct)
	}
}

func TestProfileOnlyForOneRouteOneDay(t *testing.T) {
	in := []Input{{ID: "c1", Type: "rain", Value: 1, Scope: hours(16, 22)}}
	p := catalog().Apply(frame([]int{11}, 1), Context{Horizon: "day"}, in).Conditions[0].Profile
	if len(p) != 24 || !near(p[16], -10) || p[15] != 0 || p[23] != 0 {
		t.Fatalf("profile %v", p)
	}
	if two := catalog().Apply(frame([]int{7, 11}, 1), Context{Horizon: "day"}, in); two.Conditions[0].Profile != nil {
		t.Fatal("profile must be null for several routes")
	}
}

func TestRouteEffectOnlyForSingleRouteScope(t *testing.T) {
	e := catalog().Entry("rain")
	if f, p := e.Resolve(1, []int{7}); !near(f, 0.95) || p.EffectPct != -5 {
		t.Fatalf("route 7: %v %+v", f, p)
	}
	if f, p := e.Resolve(1, []int{7, 11}); !near(f, 0.9) || p.EffectPct != -10 {
		t.Fatalf("two routes: %v %+v", f, p)
	}
}

func TestCheckRejectsBadInput(t *testing.T) {
	cases := []struct {
		in    Input
		field string
	}{
		{Input{ID: "", Type: "rain"}, "conditions[0].id"},
		{Input{ID: "c1", Type: "snow"}, "conditions[0].type"},
		{Input{ID: "c1", Type: "rain", Value: 9}, "conditions[0].value"},
		{Input{ID: "c1", Type: "rain", Scope: Scope{Routes: []int{3}}}, "conditions[0].scope.routes"},
		{Input{ID: "c1", Type: "rain", Scope: Scope{Dates: []string{"11.11.2025"}}}, "conditions[0].scope.dates"},
		{Input{ID: "c1", Type: "rain", Scope: Scope{Weekdays: []int{0}}}, "conditions[0].scope.weekdays"},
		{Input{ID: "c1", Type: "rain", Scope: Scope{Hours: []int{22, 2}}}, "conditions[0].scope.hours"},
		{Input{ID: "c1", Type: "rain", Scope: Scope{Routes: slices.Repeat([]int{7}, 11)}}, "conditions[0].scope.routes"},
		{Input{ID: "c1", Type: "rain", Scope: Scope{Dates: slices.Repeat([]string{"2025-11-11"}, 367)}}, "conditions[0].scope.dates"},
		{Input{ID: "c1", Type: "rain", Scope: Scope{Weekdays: slices.Repeat([]int{1}, 8)}}, "conditions[0].scope.weekdays"},
	}
	for _, tc := range cases {
		var fe *FieldError
		if err := catalog().Check([]Input{tc.in}); !errors.As(err, &fe) || fe.Field != tc.field {
			t.Errorf("%+v: got %v, want field %s", tc.in, err, tc.field)
		}
	}
	var fe *FieldError
	if err := catalog().Check([]Input{{ID: "a", Type: "rain"}, {ID: "a", Type: "event"}}); !errors.As(err, &fe) || fe.Field != "conditions[1].id" {
		t.Fatalf("duplicate id: %v", err)
	}
}

func TestBundleCatalogIsValid(t *testing.T) {
	c, err := LoadCatalog("../../../artifacts/bundle/conditions.json")
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Conditions) != 6 {
		t.Fatalf("%d entries", len(c.Conditions))
	}
	for _, e := range c.Conditions {
		if e.Auto {
			t.Errorf("%s must not be automatic", e.Type)
		}
	}
	rain := c.Entry("rain")
	if len(rain.ByRoute) != 9 || rain.ByRoute["7"].Unit != "мм/ч" || rain.ByRoute["7"].Horizons[0] != "day" {
		t.Fatalf("route passports not filled from the overall one: %+v", rain.ByRoute["7"])
	}
	for v, want := range map[float64]float64{0: 0, 0.2: -2.2, 0.3: -6.4, 1.2: -7.6, 3: -9.7, 5: -9.7} {
		if f, _ := rain.Resolve(v, nil); !near(f, 1+want/100) {
			t.Errorf("rain %v: factor %v, want %v", v, f, 1+want/100)
		}
	}
}

func TestValidateRejectsBadCatalog(t *testing.T) {
	c := catalog()
	c.Conditions = append(c.Conditions, catalog().Conditions...)
	if err := c.Validate(); err == nil {
		t.Fatal("duplicate types accepted")
	}
	c = catalog()
	c.Conditions[0].Auto = true
	c.Conditions[0].Passport.CIPct = [2]float64{-10, 1}
	if err := c.validateEntries(); err == nil {
		t.Fatal("automatic condition with a zero-crossing interval accepted")
	}
}

func TestUnknownTypeMessageIsBounded(t *testing.T) {
	var fe *FieldError
	err := catalog().Check([]Input{{ID: "c1", Type: strings.Repeat("x", 1000)}})
	if !errors.As(err, &fe) || fe.Field != "conditions[0].type" {
		t.Fatalf("got %v", err)
	}
	if strings.Count(fe.Message, "x") != 32 {
		t.Fatalf("message echoes %d characters of the type: %s", strings.Count(fe.Message, "x"), fe.Message)
	}
}
