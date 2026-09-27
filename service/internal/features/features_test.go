package features

import (
	"math"
	"slices"
	"testing"
	"time"
)

func load(t *testing.T) *Calendar {
	t.Helper()
	c, err := LoadCalendars("../../../input/calendar")
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func day(t *testing.T, c *Calendar, s string) Day {
	t.Helper()
	d, err := time.Parse(time.DateOnly, s)
	if err != nil {
		t.Fatal(err)
	}
	info, ok := c.Day(d)
	if !ok {
		t.Fatalf("%s outside calendar", s)
	}
	return info
}

func TestCalendarFlagsAndBlocks(t *testing.T) {
	c := load(t)
	cases := []struct {
		date                    string
		weekend, holiday, short bool
		before, after           int
		inBlock                 bool
	}{
		{"2025-11-01", false, false, true, 1, 7, false},
		{"2025-11-04", true, true, false, 0, 0, true},
		{"2025-11-05", false, false, false, 7, 1, false},
		{"2025-11-10", false, false, false, 7, 6, false},
		{"2025-12-24", false, false, false, 7, 7, false},
		{"2025-12-29", false, false, false, 2, 7, false},
		{"2025-12-31", true, false, false, 0, 0, true},
		{"2026-01-11", true, false, false, 0, 0, true},
		{"2026-01-12", false, false, false, 7, 1, false},
		{"2026-02-23", true, true, false, 0, 0, true},
		{"2026-03-09", true, false, false, 0, 0, true},
		{"2026-04-30", false, false, true, 1, 7, false},
	}
	for _, tc := range cases {
		d := day(t, c, tc.date)
		got := []any{d.IsWeekend, d.IsHoliday, d.IsShortWorkingDay, d.DaysBeforeBlock, d.DaysAfterBlock, d.InBlock}
		want := []any{tc.weekend, tc.holiday, tc.short, tc.before, tc.after, tc.inBlock}
		if !slices.Equal(got, want) {
			t.Errorf("%s: got %v, want %v", tc.date, got, want)
		}
	}
}

func TestOrdinaryDropsCalendarAnomalies(t *testing.T) {
	c := load(t)
	d := day(t, c, "2025-12-31").Ordinary()
	if d.IsWeekend || d.IsHoliday || d.InBlock || d.DaysBeforeBlock != BlockHorizon || d.DaysAfterBlock != BlockHorizon || d.Weekday != 3 {
		t.Fatalf("unexpected ordinary day %+v", d)
	}
	if plain := day(t, c, "2025-11-12"); plain != plain.Ordinary() {
		t.Fatalf("2025-11-12 should already be ordinary: %+v", plain)
	}
}

var serviceFeatures = []string{
	"hour", "route", "is_holiday", "is_weekend", "is_short_working_day", "season", "weekday",
	"hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "day_of_month_sin", "day_of_month_cos",
	"days_before_block", "days_after_block", "in_block",
}
var serviceCats = []string{"route", "is_holiday", "is_weekend", "is_short_working_day", "season", "weekday"}

func TestSpecEncodesInModelOrder(t *testing.T) {
	c := load(t)
	s, err := NewSpec(serviceFeatures, serviceCats)
	if err != nil {
		t.Fatal(err)
	}
	if s.NumFloat() != 10 || s.NumCat() != 6 {
		t.Fatalf("floats %d cats %d", s.NumFloat(), s.NumCat())
	}
	r := Row{Route: 7, Hour: 8, Day: day(t, c, "2025-11-10")}
	fl, cat := make([]float32, 10), make([]string, 6)
	s.Encode(r, fl, cat)
	want := []float32{
		8, float32(math.Sin(2 * math.Pi * 8 / 24)), float32(math.Cos(2 * math.Pi * 8 / 24)), 0, 1,
		float32(math.Sin(2 * math.Pi * 9 / 30)), float32(math.Cos(2 * math.Pi * 9 / 30)), 7, 6, 0,
	}
	if !slices.Equal(fl, want) {
		t.Fatalf("floats %v, want %v", fl, want)
	}
	if want := []string{"7", "0", "0", "0", "autumn", "1"}; !slices.Equal(cat, want) {
		t.Fatalf("cats %v, want %v", cat, want)
	}
	v := s.Values(r)
	if len(v) != 16 || v[1].Name != "route" || v[1].Kind != "categorical" || v[1].Value != "7" || v[0].Kind != "float" || v[0].Value != 8.0 {
		t.Fatalf("values %+v", v)
	}
}

func TestSpecRejectsUnknownFeature(t *testing.T) {
	if _, err := NewSpec([]string{"hour", "temperature_c"}, nil); err == nil {
		t.Fatal("unknown feature accepted")
	}
	if _, err := NewSpec([]string{"hour"}, []string{"route"}); err == nil {
		t.Fatal("categorical feature outside the list accepted")
	}
}
