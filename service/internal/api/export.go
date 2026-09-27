package api

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"

	"stand/internal/features"
	"stand/internal/forecast"
)

func (s *Server) export(w http.ResponseWriter, r *http.Request) {
	t0 := time.Now()
	format := r.URL.Query().Get("format")
	if format != "csv" && format != "xlsx" {
		s.fail(w, badRequest("format", "Формат выгрузки - csv или xlsx"))
		return
	}
	q, err := s.parseForecast(w, r)
	if err != nil {
		s.fail(w, err)
		return
	}
	c, err := s.compute(q, time.Since(t0))
	if err != nil {
		s.fail(w, err)
		return
	}
	forecastSheet := c.table()
	var buf bytes.Buffer
	contentType := "text/csv; charset=utf-8"
	if format == "csv" {
		err = forecastSheet.csv(&buf)
	} else {
		contentType = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
		sheets := []sheet{forecastSheet}
		if len(c.res.Conditions) > 0 {
			sheets = append(sheets, c.audit())
		}
		err = writeXLSX(&buf, sheets)
	}
	if err != nil {
		s.fail(w, err)
		return
	}
	w.Header().Set("Content-Type", contentType)
	w.Header().Set("Content-Disposition", fmt.Sprintf(`attachment; filename="%s"`, c.filename(format)))
	w.Write(buf.Bytes())
	s.stats.record(time.Since(t0), c.grid.Rows)
}

func (c *computed) table() sheet {
	sh := sheet{name: "Прогноз", header: []string{"route", "date", "hour", "base", "factor", "prediction"}}
	if c.q.daily {
		sh.header = slices.Delete(sh.header, 2, 3)
	}
	if c.lo != nil {
		sh.header = append(sh.header, "lo", "hi")
	}
	for i, s := range c.series() {
		base, value := c.grid.Base[i], c.res.Value[i]
		if c.q.daily {
			base, value = forecast.Daily(base), forecast.Daily(value)
		}
		for j := range s.Value {
			row := []any{s.Route}
			if c.q.daily {
				row = append(row, c.q.from.AddDate(0, 0, j).Format(time.DateOnly))
			} else {
				row = append(row, c.q.from.AddDate(0, 0, j/24).Format(time.DateOnly), j%24)
			}
			factor := 1.0
			if base[j] != 0 {
				factor = roundTo(value[j]/base[j], 4)
			}
			row = append(row, s.Base[j], factor, s.Value[j])
			if c.lo != nil {
				row = append(row, s.Lo[j], s.Hi[j])
			}
			sh.rows = append(sh.rows, row)
		}
	}
	return sh
}

func (c *computed) audit() sheet {
	sh := sheet{name: "Условия", header: []string{"id", "тип", "применено", "множитель", "вклад_%", "точек"}}
	for _, a := range rounded(c.res.Conditions) {
		applied := "нет"
		if a.Applied {
			applied = "да"
		}
		sh.rows = append(sh.rows, []any{a.ID, a.Type, applied, a.Factor, a.ContributionPct, a.Points})
	}
	return sh
}

func (c *computed) filename(ext string) string {
	name := "forecast_" + c.q.engine.Name + "_" + c.q.from.Format(time.DateOnly)
	if c.q.days > 1 {
		name += "_" + c.q.from.AddDate(0, 0, c.q.days-1).Format(time.DateOnly)
	}
	routes := "all"
	if len(c.q.routes) != len(features.Routes) {
		parts := make([]string, len(c.q.routes))
		for i, r := range c.q.routes {
			parts[i] = strconv.Itoa(r)
		}
		routes = "r" + strings.Join(parts, "-")
	}
	return name + "_" + routes + "." + ext
}

func (sh sheet) csv(buf *bytes.Buffer) error {
	cw := csv.NewWriter(buf)
	cw.Comma = ';'
	if err := cw.Write(sh.header); err != nil {
		return err
	}
	rec := make([]string, len(sh.header))
	for _, row := range sh.rows {
		for i, v := range row {
			rec[i] = fmt.Sprint(v)
		}
		if err := cw.Write(rec); err != nil {
			return err
		}
	}
	cw.Flush()
	return cw.Error()
}
