package forecast_test

import (
	"math"
	"os"
	"slices"
	"testing"
	"time"

	"stand/internal/bundle"
	"stand/internal/catboost"
	"stand/internal/features"
	"stand/internal/forecast"
)

var engine *forecast.Engine

func TestMain(m *testing.M) {
	b, err := bundle.Load("../../../artifacts/bundle/catboost")
	if err != nil {
		panic(err)
	}
	cal, err := features.LoadCalendars("../../../input/calendar")
	if err != nil {
		panic(err)
	}
	model, err := catboost.Load(b.Dir)
	if err != nil {
		panic(err)
	}
	if engine, err = forecast.New("catboost", b, cal, model); err != nil {
		panic(err)
	}
	os.Exit(m.Run())
}

func date(s string) time.Time {
	t, _ := time.Parse(time.DateOnly, s)
	return t
}

func TestReferenceGridMatchesPython(t *testing.T) {
	c, err := engine.Verify()
	if err != nil {
		t.Fatal(err)
	}
	if c.Rows != 14640 {
		t.Fatalf("checked %d rows", c.Rows)
	}
	t.Logf("max |raw| %.3g, max |final| %.3g", c.MaxRaw, c.MaxFinal)
}

func TestUsualEqualsBaseOnOrdinaryDay(t *testing.T) {
	g, err := engine.Grid([]int{7}, date("2025-11-12"), 1)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(g.Base[0], g.Usual[0]) || g.Rows != 24 {
		t.Fatalf("ordinary day: rows %d", g.Rows)
	}
}

func TestUsualIgnoresHoliday(t *testing.T) {
	g, err := engine.Grid([]int{17}, date("2025-11-04"), 1)
	if err != nil {
		t.Fatal(err)
	}
	var base, usual float64
	for i := range g.Base[0] {
		base += g.Base[0][i]
		usual += g.Usual[0][i]
	}
	if usual <= base || g.Rows != 48 {
		t.Fatalf("holiday: base %.0f usual %.0f rows %d", base, usual, g.Rows)
	}
}

func TestRoute5IsCityShare(t *testing.T) {
	d := date("2025-11-12")
	g5, err := engine.Grid([]int{5}, d, 1)
	if err != nil {
		t.Fatal(err)
	}
	var others []int
	for _, r := range features.Routes {
		if r != 5 {
			others = append(others, r)
		}
	}
	g, err := engine.Grid(others, d, 1)
	if err != nil {
		t.Fatal(err)
	}
	s := engine.Bundle.Postprocess.Route5Share
	for cell := range 24 {
		city := 0.0
		for i := range others {
			city += g.Base[i][cell]
		}
		if want := city * s / (1 - s); math.Abs(g5.Base[0][cell]-want) > 1e-9*max(1, want) {
			t.Fatalf("cell %d: %v want %v", cell, g5.Base[0][cell], want)
		}
	}
	if engine.Rows([]int{5}, 2) != 9*48 || engine.Rows([]int{7, 5}, 1) != 9*24 || engine.Rows([]int{7, 11}, 1) != 48 {
		t.Fatal("row count for route 5 must include the nine modeled routes")
	}
}

func TestCorridorBracketsBase(t *testing.T) {
	g, err := engine.Grid([]int{7, 5}, date("2025-11-11"), 3)
	if err != nil {
		t.Fatal(err)
	}
	for _, daily := range []bool{false, true} {
		lo, hi := g.Corridor(&engine.Bundle.Corridor, "week", daily)
		for i := range g.Routes {
			base := g.Base[i]
			if daily {
				base = forecast.Daily(base)
			}
			if len(lo[i]) != len(base) {
				t.Fatalf("daily=%v: %d bounds for %d points", daily, len(lo[i]), len(base))
			}
			for j, b := range base {
				if lo[i][j] > b || hi[i][j] < b || lo[i][j] < 0 {
					t.Fatalf("daily=%v route %d point %d: %v not in [%v, %v]", daily, g.Routes[i], j, b, lo[i][j], hi[i][j])
				}
			}
		}
	}
}
