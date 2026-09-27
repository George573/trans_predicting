package features

import (
	"encoding/xml"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"time"
)

const (
	BlockHorizon = 7
	minBlockLen  = 3
)

type Day struct {
	Date              time.Time
	Weekday           int
	Season            string
	DayOfMonth        int
	DaysInMonth       int
	IsHoliday         bool
	IsWeekend         bool
	IsShortWorkingDay bool
	DaysBeforeBlock   int
	DaysAfterBlock    int
	InBlock           bool
}

type Calendar struct {
	first time.Time
	days  []Day
}

type override struct {
	t       int
	holiday bool
}

type xmlCalendar struct {
	Year string `xml:"year,attr"`
	Days []struct {
		D string `xml:"d,attr"`
		T string `xml:"t,attr"`
		H string `xml:"h,attr"`
	} `xml:"days>day"`
}

func LoadCalendars(dir string) (*Calendar, error) {
	paths, err := filepath.Glob(filepath.Join(dir, "*.xml"))
	if err != nil {
		return nil, err
	}
	if len(paths) == 0 {
		return nil, fmt.Errorf("в %s нет производственных календарей *.xml", dir)
	}
	overrides := map[time.Time]override{}
	var years []int
	for _, p := range paths {
		y, err := readCalendar(p, overrides)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", p, err)
		}
		years = append(years, y)
	}
	slices.Sort(years)
	for i := 1; i < len(years); i++ {
		if years[i] != years[i-1]+1 {
			return nil, fmt.Errorf("календари должны идти подряд, есть %v", years)
		}
	}
	c := &Calendar{first: date(years[0], 1, 1)}
	for d := c.first; d.Year() <= years[len(years)-1]; d = d.AddDate(0, 0, 1) {
		c.days = append(c.days, newDay(d, overrides))
	}
	c.markBlocks(years[len(years)-1] + 1)
	return c, nil
}

func readCalendar(path string, overrides map[time.Time]override) (int, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	var x xmlCalendar
	if err := xml.Unmarshal(b, &x); err != nil {
		return 0, err
	}
	year, err := strconv.Atoi(x.Year)
	if err != nil {
		return 0, fmt.Errorf("год календаря: %w", err)
	}
	for _, d := range x.Days {
		t, err := strconv.Atoi(d.T)
		if err != nil {
			return 0, fmt.Errorf("тип дня %q: %w", d.D, err)
		}
		dt, err := time.Parse("2006.01.02", x.Year+"."+d.D)
		if err != nil {
			return 0, err
		}
		overrides[dt] = override{t: t, holiday: t == 1 && d.H != ""}
	}
	return year, nil
}

func newDay(d time.Time, overrides map[time.Time]override) Day {
	wd := int(d.Weekday())
	if wd == 0 {
		wd = 7
	}
	day := Day{
		Date: d, Weekday: wd, Season: season(d.Month()), DayOfMonth: d.Day(),
		DaysInMonth: date(d.Year(), d.Month()+1, 0).Day(), IsWeekend: wd >= 6,
		DaysBeforeBlock: BlockHorizon, DaysAfterBlock: BlockHorizon,
	}
	if o, ok := overrides[d]; ok {
		day.IsWeekend, day.IsShortWorkingDay, day.IsHoliday = o.t == 1, o.t == 2, o.holiday
	}
	return day
}

func (c *Calendar) markBlocks(nextYear int) {
	var off []time.Time
	for _, d := range c.days {
		if d.IsWeekend {
			off = append(off, d.Date)
		}
	}
	for i := 1; i <= 8; i++ {
		off = append(off, date(nextYear, 1, i))
	}
	var blocks [][2]time.Time
	start := off[0]
	for i := 1; i <= len(off); i++ {
		if i < len(off) && daysBetween(off[i-1], off[i]) == 1 {
			continue
		}
		if daysBetween(start, off[i-1])+1 >= minBlockLen {
			blocks = append(blocks, [2]time.Time{start, off[i-1]})
		}
		if i < len(off) {
			start = off[i]
		}
	}
	for i := range c.days {
		d := &c.days[i]
		for _, b := range blocks {
			if !d.Date.Before(b[0]) && !d.Date.After(b[1]) {
				d.InBlock, d.DaysBeforeBlock, d.DaysAfterBlock = true, 0, 0
				break
			}
			if d.Date.Before(b[0]) {
				d.DaysBeforeBlock = min(d.DaysBeforeBlock, daysBetween(d.Date, b[0]))
			} else {
				d.DaysAfterBlock = min(d.DaysAfterBlock, daysBetween(b[1], d.Date))
			}
		}
	}
}

func (c *Calendar) Day(t time.Time) (Day, bool) {
	i := daysBetween(c.first, t)
	if t.Before(c.first) || i >= len(c.days) {
		return Day{}, false
	}
	return c.days[i], true
}

func (d Day) Ordinary() Day {
	d.IsHoliday, d.IsShortWorkingDay, d.InBlock = false, false, false
	d.IsWeekend = d.Weekday >= 6
	d.DaysBeforeBlock, d.DaysAfterBlock = BlockHorizon, BlockHorizon
	return d
}

func season(m time.Month) string {
	switch m {
	case 12, 1, 2:
		return "winter"
	case 3, 4, 5:
		return "spring"
	case 6, 7, 8:
		return "summer"
	}
	return "autumn"
}

func date(y int, m time.Month, d int) time.Time {
	return time.Date(y, m, d, 0, 0, 0, 0, time.UTC)
}

func daysBetween(a, b time.Time) int {
	return int(b.Sub(a).Hours() / 24)
}
