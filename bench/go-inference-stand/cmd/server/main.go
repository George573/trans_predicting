// Стенд прогнозного API: CNN считается раннером на ONNX Runtime на каждый запрос, без кэша.
// На "/" отдаётся интерактивный UI (web/index.html), который ходит в этот же API.
package main

import (
	"crypto/subtle"
	"embed"
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"math"
	"net/http"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"
)

//go:embed web
var webFS embed.FS

var routesAll = []int{1, 5, 7, 11, 12, 17, 25, 26, 28, 50}

var (
	horizonFrom = time.Date(2025, 11, 1, 0, 0, 0, 0, time.UTC)
	horizonTo   = time.Date(2025, 12, 31, 0, 0, 0, 0, time.UTC)
)

type stats struct {
	requests atomic.Int64
	errors   atomic.Int64
	rows     atomic.Int64
	modelNs  atomic.Int64
	totalNs  atomic.Int64
}

type app struct {
	runner  *runner
	info    map[string]any
	name    string
	version string
	user    string
	pass    string
	started time.Time
	st      stats
}

type apiError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func (a *app) fail(w http.ResponseWriter, status int, code, msg string) {
	a.st.errors.Add(1)
	writeJSON(w, status, map[string]apiError{"error": {code, msg}})
}

func (a *app) auth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		u, p, ok := r.BasicAuth()
		if a.user != "" && (!ok || subtle.ConstantTimeCompare([]byte(u), []byte(a.user)) != 1 ||
			subtle.ConstantTimeCompare([]byte(p), []byte(a.pass)) != 1) {
			w.Header().Set("WWW-Authenticate", `Basic realm="forecast"`)
			a.fail(w, http.StatusUnauthorized, "unauthorized", "нужен логин и пароль")
			return
		}
		next.ServeHTTP(w, r)
	})
}

type params struct {
	routes   []int
	from, to time.Time
	gran     string
	k        float64
	days     int
}

func parseDate(q string, name string) (time.Time, error) {
	t, err := time.Parse("2006-01-02", q)
	if err != nil {
		return t, fmt.Errorf("%s: дата в формате YYYY-MM-DD", name)
	}
	if t.Before(horizonFrom) || t.After(horizonTo) {
		return t, fmt.Errorf("%s: прогноз есть на %s..%s", name, horizonFrom.Format("2006-01-02"), horizonTo.Format("2006-01-02"))
	}
	return t, nil
}

func parseK(q map[string][]string) (float64, error) {
	k := 1.0
	// корректирующие коэффициенты перемножаются: k_weather * k_event * k_season
	for _, name := range []string{"k_weather", "k_event", "k_season"} {
		if v := q[name]; len(v) > 0 && v[0] != "" {
			x, err := strconv.ParseFloat(v[0], 64)
			if err != nil || x < 0 || x > 5 {
				return 0, fmt.Errorf("%s: число в диапазоне 0..5", name)
			}
			k *= x
		}
	}
	return k, nil
}

func parseRoute(s string) (int, error) {
	n, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil || !slices.Contains(routesAll, n) {
		return 0, fmt.Errorf("неизвестный маршрут %q, есть %v", s, routesAll)
	}
	return n, nil
}

// parse: route=1,7 (пусто = все), from/to=YYYY-MM-DD, granularity=hour|day, k_*=коэффициенты.
func parse(r *http.Request) (params, error) {
	q := r.URL.Query()
	p := params{gran: q.Get("granularity")}
	if p.gran == "" {
		p.gran = "hour"
	}
	if p.gran != "hour" && p.gran != "day" {
		return p, errors.New("granularity: hour или day")
	}
	if rs := q.Get("route"); rs != "" {
		for _, s := range strings.Split(rs, ",") {
			n, err := parseRoute(s)
			if err != nil {
				return p, err
			}
			p.routes = append(p.routes, n)
		}
	} else {
		p.routes = routesAll
	}
	var err error
	if p.from, err = parseDate(q.Get("from"), "from"); err != nil {
		return p, err
	}
	p.to = p.from
	if t := q.Get("to"); t != "" {
		if p.to, err = parseDate(t, "to"); err != nil {
			return p, err
		}
	}
	if p.to.Before(p.from) {
		return p, errors.New("to раньше from")
	}
	p.days = int(p.to.Sub(p.from).Hours()/24) + 1
	if p.gran == "hour" && p.days > 31 {
		return p, errors.New("почасовая детализация - не больше 31 дня, для длинных интервалов granularity=day")
	}
	p.k, err = parseK(q)
	return p, err
}

type series struct {
	Route  int     `json:"route"`
	Total  int64   `json:"total"`
	Peak   int     `json:"peak_index"`
	Values []int64 `json:"values"`
}

type timings struct {
	Parse  float64 `json:"parse_ms"`
	Runner float64 `json:"runner_ms"`
	Model  float64 `json:"model_ms"`
	Rollup float64 `json:"rollup_ms"`
}

func ms(d time.Duration) float64 { return math.Round(float64(d.Microseconds())) / 1000 }

func (a *app) handleForecast(w http.ResponseWriter, r *http.Request) {
	t0 := time.Now()
	p, err := parse(r)
	if err != nil {
		a.fail(w, http.StatusBadRequest, "bad_request", err.Error())
		return
	}
	t1 := time.Now()

	// 1. один запрос в раннер на весь прогноз: маршрут x день x 24 часа
	pr, err := a.runner.predict(p.routes, p.from, p.days)
	if err != nil {
		a.fail(w, http.StatusBadGateway, "runner", err.Error())
		return
	}
	t2 := time.Now()
	model := time.Duration(pr.ModelMs * float64(time.Millisecond))

	// 2. коэффициент, свёртка до часа/дня, округление
	out := make([]series, len(p.routes))
	for ri, rt := range p.routes {
		s := series{Route: rt}
		for d := 0; d < p.days; d++ {
			var day float64
			for h := 0; h < 24; h++ {
				v := pr.Values[ri][d*24+h] * p.k
				day += v
				if p.gran == "hour" {
					s.Values = append(s.Values, int64(math.Round(v)))
				}
			}
			if p.gran == "day" {
				s.Values = append(s.Values, int64(math.Round(day)))
			}
			s.Total += int64(math.Round(day))
		}
		for i, v := range s.Values {
			if v > s.Values[s.Peak] {
				s.Peak = i
			}
		}
		out[ri] = s
	}
	t3 := time.Now()

	step := "1h"
	if p.gran == "day" {
		step = "1d"
	}
	body, _ := json.Marshal(map[string]any{
		"model_version": a.version, "start": p.from.Format("2006-01-02") + "T00:00", "step": step, "k": p.k,
		"series": out,
		"meta": map[string]any{
			"rows": pr.Rows, "model": a.name,
			"timings": timings{ms(t1.Sub(t0)), ms(t2.Sub(t1) - model), ms(model), ms(t3.Sub(t2))},
		},
	})
	t4 := time.Now()
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Server-Timing", fmt.Sprintf("parse;dur=%.3f, runner;dur=%.3f, model;dur=%.3f, rollup;dur=%.3f, json;dur=%.3f",
		ms(t1.Sub(t0)), ms(t2.Sub(t1)-model), ms(model), ms(t3.Sub(t2)), ms(t4.Sub(t3))))
	w.Header().Set("X-Model-Rows", strconv.Itoa(pr.Rows))
	_, _ = w.Write(body)

	a.st.requests.Add(1)
	a.st.rows.Add(int64(pr.Rows))
	a.st.modelNs.Add(int64(model))
	a.st.totalNs.Add(int64(time.Since(t0)))
}

// explain - разбор одной ячейки: что ушло в модель и как получилось число в ответе.
func (a *app) handleExplain(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	route, err := parseRoute(q.Get("route"))
	if err != nil {
		a.fail(w, http.StatusBadRequest, "bad_request", err.Error())
		return
	}
	date, err := parseDate(q.Get("date"), "date")
	if err != nil {
		a.fail(w, http.StatusBadRequest, "bad_request", err.Error())
		return
	}
	hour, err := strconv.Atoi(q.Get("hour"))
	if err != nil || hour < 0 || hour > 23 {
		a.fail(w, http.StatusBadRequest, "bad_request", "hour: 0..23")
		return
	}
	k, err := parseK(q)
	if err != nil {
		a.fail(w, http.StatusBadRequest, "bad_request", err.Error())
		return
	}
	ex, err := a.runner.explain(route, date)
	if err != nil {
		a.fail(w, http.StatusBadGateway, "runner", err.Error())
		return
	}
	var history map[string]any
	if ex.History != nil {
		window := a.info["history"].(map[string]any)
		history = map[string]any{"from": window["from"], "to": window["to"], "boardings": ex.History.Boardings,
			"same_weekday_hour": ex.History.SameWeekday[hour]}
	}
	weekday := int(date.Weekday())
	if weekday == 0 {
		weekday = 7
	}
	raw := ex.Values[hour]
	writeJSON(w, http.StatusOK, map[string]any{
		"route": route, "date": date.Format("2006-01-02"), "hour": hour,
		"calendar": map[string]any{
			"weekday": weekday, "day_of_month": date.Day(), "days_in_month": time.Date(date.Year(), date.Month()+1, 0, 0, 0, 0, 0, time.UTC).Day(),
		},
		"fallback": ex.Fallback, "model_input": ex.Input, "history": history, "model": a.name, "parameters": a.info["parameters"],
		"raw": raw, "k": k, "value": int64(math.Round(raw * k)), "model_us": int64(ex.ModelMs * 1000),
	})
}

func (a *app) handleExport(w http.ResponseWriter, r *http.Request) {
	p, err := parse(r)
	if err != nil {
		a.fail(w, http.StatusBadRequest, "bad_request", err.Error())
		return
	}
	pr, err := a.runner.predict(p.routes, p.from, p.days)
	if err != nil {
		a.fail(w, http.StatusBadGateway, "runner", err.Error())
		return
	}
	w.Header().Set("Content-Type", "text/csv; charset=utf-8")
	w.Header().Set("Content-Disposition", `attachment; filename="forecast.csv"`)
	cw := csv.NewWriter(w)
	cw.Comma = ';'
	_ = cw.Write([]string{"route", "date", "hour", "prediction"})
	for ri, route := range p.routes {
		for i, v := range pr.Values[ri] {
			_ = cw.Write([]string{strconv.Itoa(route), p.from.AddDate(0, 0, i/24).Format("2006-01-02"), strconv.Itoa(i % 24),
				strconv.FormatInt(int64(math.Round(v*p.k)), 10)})
		}
	}
	cw.Flush()
}

func (a *app) handleModel(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"model": a.name, "runner": a.info,
		"routes": routesAll, "from": horizonFrom.Format("2006-01-02"), "to": horizonTo.Format("2006-01-02"),
		"model_version": a.version, "gomaxprocs": runtime.GOMAXPROCS(0), "go": runtime.Version(),
	})
}

// stats - счётчики с момента старта; UI опрашивает их раз в секунду и сам считает RPS.
func (a *app) handleStats(w http.ResponseWriter, _ *http.Request) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	var ru syscall.Rusage
	_ = syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	cpu := float64(ru.Utime.Sec+ru.Stime.Sec) + float64(ru.Utime.Usec+ru.Stime.Usec)/1e6
	writeJSON(w, http.StatusOK, map[string]any{
		"uptime_s": time.Since(a.started).Seconds(), "requests": a.st.requests.Load(), "errors": a.st.errors.Load(),
		"rows": a.st.rows.Load(), "model_ms": float64(a.st.modelNs.Load()) / 1e6, "total_ms": float64(a.st.totalNs.Load()) / 1e6,
		"heap_mb": float64(m.HeapAlloc) / (1 << 20), "sys_mb": float64(m.Sys) / (1 << 20), "goroutines": runtime.NumGoroutine(),
		"cpu_s": cpu, "gomaxprocs": runtime.GOMAXPROCS(0),
	})
}

func main() {
	addr := flag.String("addr", ":8080", "listen")
	runnerURL := flag.String("runner", "http://localhost:8000", "адрес раннера CNN (python -m tram_forecast serve)")
	auth := flag.String("auth", "", "user:pass для basic auth на UI и API (пусто = выкл)")
	flag.Parse()

	rn := newRunner(*runnerURL)
	info, err := rn.model()
	if err != nil {
		log.Fatal(err)
	}
	name := fmt.Sprintf("cnn-%v", info["model_kind"])
	a := &app{runner: rn, info: info, name: name, version: fmt.Sprintf("%s-%d", name, time.Now().Unix()), started: time.Now()}
	if u, p, ok := strings.Cut(*auth, ":"); ok {
		a.user, a.pass = u, p
	}
	log.Printf("%s: %v параметров, %v; GOMAXPROCS=%d; http://localhost%s", name, info["parameters"], info["engine"], runtime.GOMAXPROCS(0), *addr)

	web, _ := fs.Sub(webFS, "web")
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte("ok")) })
	app := http.NewServeMux()
	app.HandleFunc("GET /api/v1/model", a.handleModel)
	app.HandleFunc("GET /api/v1/stats", a.handleStats)
	app.HandleFunc("GET /api/v1/forecast", a.handleForecast)
	app.HandleFunc("GET /api/v1/explain", a.handleExplain)
	app.HandleFunc("GET /api/v1/forecast/export", a.handleExport)
	app.Handle("GET /", http.FileServerFS(web))
	mux.Handle("/", a.auth(app))
	srv := &http.Server{Addr: *addr, Handler: mux, ReadHeaderTimeout: 5 * time.Second, WriteTimeout: 30 * time.Second}
	log.Fatal(srv.ListenAndServe())
}
