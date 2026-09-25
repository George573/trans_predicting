// Package features строит признаки модели по (route, date, hour).
// Логика повторяет data_preparation.enrich_features и input/calendar/transform.py.
package features

import (
	"encoding/xml"
	"fmt"
	"math"
	"os"
	"strconv"
	"time"
)

type DayFlags struct {
	IsHoliday         bool
	IsWeekend         bool
	IsShortWorkingDay bool
}

type Calendar struct {
	year      int
	overrides map[string]DayFlags // "MM.DD"
}

type xmlCalendar struct {
	Year string `xml:"year,attr"`
	Days []struct {
		D string `xml:"d,attr"`
		T string `xml:"t,attr"`
		H string `xml:"h,attr"`
	} `xml:"days>day"`
}

// LoadCalendar читает производственный календарь consultant.ru (xmlcalendar).
func LoadCalendar(path string) (*Calendar, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var x xmlCalendar
	if err := xml.Unmarshal(b, &x); err != nil {
		return nil, err
	}
	y, err := strconv.Atoi(x.Year)
	if err != nil {
		return nil, fmt.Errorf("calendar year: %w", err)
	}
	c := &Calendar{year: y, overrides: map[string]DayFlags{}}
	for _, d := range x.Days {
		t, _ := strconv.Atoi(d.T)
		c.overrides[d.D] = DayFlags{
			IsWeekend:         t == 1,
			IsShortWorkingDay: t == 2,
			IsHoliday:         t == 1 && d.H != "",
		}
	}
	return c, nil
}

func (c *Calendar) Flags(d time.Time) DayFlags {
	if d.Year() == c.year {
		if f, ok := c.overrides[d.Format("01.02")]; ok {
			return f
		}
	}
	wd := d.Weekday()
	return DayFlags{IsWeekend: wd == time.Saturday || wd == time.Sunday}
}

// Row - признаки одной ячейки route x date x hour.
type Row struct {
	Route   int
	Hour    int
	Weekday int // 1=пн .. 7=вс, как в polars
	Season  string
	// для sin/cos дня месяца
	DayOfMonth  int
	DaysInMonth int
	DayFlags
}

func Season(m time.Month) string {
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

func ISOWeekday(d time.Time) int {
	wd := int(d.Weekday())
	if wd == 0 {
		return 7
	}
	return wd
}

func (c *Calendar) Build(route int, d time.Time, hour int) Row {
	return Row{Route: route, Hour: hour, Weekday: ISOWeekday(d), Season: Season(d.Month()),
		DayOfMonth: d.Day(), DaysInMonth: time.Date(d.Year(), d.Month()+1, 0, 0, 0, 0, 0, time.UTC).Day(), DayFlags: c.Flags(d)}
}

func b01(b bool) string {
	if b {
		return "1"
	}
	return "0"
}

// CatStrings - категориальные признаки в том порядке и той строковой форме, в которой
// модель видела их при обучении из pandas: bool -> "0"/"1", int -> "7". Если передать
// "True"/"False", модель молча выдаст другой прогноз (проверено: до 3177 посадок на ячейку).
func (r Row) CatStrings() []string {
	return []string{
		strconv.Itoa(r.Route), b01(r.IsHoliday), b01(r.IsWeekend), b01(r.IsShortWorkingDay),
		r.Season, strconv.Itoa(r.Weekday),
	}
}

// Floats - числовые признаки cyclic-модели в порядке обучения:
// hour, hour_sin, hour_cos, weekday_sin, weekday_cos, day_of_month_sin, day_of_month_cos.
// Формулы как в data_preparation.add_cyclic_features.
func (r Row) Floats() [7]float32 {
	ah := 2 * math.Pi * float64(r.Hour) / 24
	aw := 2 * math.Pi * float64(r.Weekday-1) / 7
	ad := 2 * math.Pi * float64(r.DayOfMonth-1) / float64(r.DaysInMonth)
	return [7]float32{
		float32(r.Hour),
		float32(math.Sin(ah)), float32(math.Cos(ah)),
		float32(math.Sin(aw)), float32(math.Cos(aw)),
		float32(math.Sin(ad)), float32(math.Cos(ad)),
	}
}
