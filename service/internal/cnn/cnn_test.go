package cnn_test

import (
	"os"
	"testing"
	"time"

	"stand/internal/bundle"
	"stand/internal/cnn"
	"stand/internal/features"
	"stand/internal/forecast"
)

var engine *forecast.Engine

func TestMain(m *testing.M) {
	b, err := bundle.Load("../../../artifacts/bundle/cnn")
	if err != nil {
		panic(err)
	}
	cal, err := features.LoadCalendars("../../../input/calendar")
	if err != nil {
		panic(err)
	}
	net, err := cnn.Load(b.Dir)
	if err != nil {
		panic(err)
	}
	if engine, err = forecast.New("cnn", b, cal, net); err != nil {
		panic(err)
	}
	os.Exit(m.Run())
}

func TestReferenceGridMatchesTorch(t *testing.T) {
	c, err := engine.Verify()
	if err != nil {
		t.Fatal(err)
	}
	if c.Rows != 43440 {
		t.Fatalf("checked %d rows", c.Rows)
	}
	t.Logf("max |raw| %.3g, max |final| %.3g", c.MaxRaw, c.MaxFinal)
	nine := []int{1, 7, 11, 12, 17, 25, 26, 28, 50}
	from := time.Date(2025, 11, 1, 0, 0, 0, 0, time.UTC)
	var took []time.Duration
	for _, q := range []struct {
		routes []int
		days   int
	}{{[]int{7}, 61}, {nine, 31}, {nine, 61}} {
		t0 := time.Now()
		if _, err := engine.Grid(q.routes, from, q.days); err != nil {
			t.Fatal(err)
		}
		took = append(took, time.Since(t0))
	}
	t.Logf("grid 1 route x 61 days %v, 9 routes x 31 days %v, 9 routes x 61 days %v", took[0], took[1], took[2])
}

func TestOutsideContextHorizonFails(t *testing.T) {
	if _, err := engine.Grid([]int{7}, time.Date(2026, 5, 1, 0, 0, 0, 0, time.UTC), 1); err == nil {
		t.Fatal("CNN answered after 2026-04-30")
	}
}

func TestHolidayIsNotOrdinary(t *testing.T) {
	g, err := engine.Grid([]int{7}, time.Date(2025, 11, 4, 0, 0, 0, 0, time.UTC), 1)
	if err != nil {
		t.Fatal(err)
	}
	if g.Rows != 48 {
		t.Fatalf("holiday must get a separate usual run, rows %d", g.Rows)
	}
}
