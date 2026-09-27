package api_test

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"stand/internal/api"
	"stand/internal/bundle"
	"stand/internal/catboost"
	"stand/internal/cnn"
	"stand/internal/conditions"
	"stand/internal/features"
	"stand/internal/forecast"
)

var open, closed *httptest.Server

var (
	loaded  []*forecast.Engine
	catalog *conditions.Catalog
)

func TestMain(m *testing.M) {
	var err error
	catalog, err = conditions.LoadCatalog("../../../artifacts/bundle/conditions.json")
	if err != nil {
		panic(err)
	}
	cal, err := features.LoadCalendars("../../../input/calendar")
	if err != nil {
		panic(err)
	}
	var engines []*forecast.Engine
	for _, name := range []string{"catboost", "cnn"} {
		b, err := bundle.Load("../../../artifacts/bundle/" + name)
		if err != nil {
			panic(err)
		}
		var model forecast.Model
		if name == "catboost" {
			model, err = catboost.Load(b.Dir)
		} else {
			model, err = cnn.Load(b.Dir)
		}
		if err != nil {
			panic(err)
		}
		e, err := forecast.New(name, b, cal, model)
		if err != nil {
			panic(err)
		}
		engines = append(engines, e)
	}
	geo := "../../../web/geo/routes.geojson"
	s, err := api.New(engines, catalog, api.Config{GeoPath: geo})
	if err != nil {
		panic(err)
	}
	a, err := api.New(engines, catalog, api.Config{GeoPath: geo, Auth: "jury:secret"})
	if err != nil {
		panic(err)
	}
	loaded = engines
	open, closed = httptest.NewServer(s.Handler()), httptest.NewServer(a.Handler())
	code := m.Run()
	open.Close()
	closed.Close()
	os.Exit(code)
}

func send(t *testing.T, req *http.Request) (int, http.Header, map[string]any) {
	t.Helper()
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("%s %s: status %d, body is not JSON: %v", req.Method, req.URL.Path, resp.StatusCode, err)
	}
	return resp.StatusCode, resp.Header, out
}

func post(t *testing.T, srv *httptest.Server, path string, body any) (int, http.Header, map[string]any) {
	t.Helper()
	b, ok := body.([]byte)
	if !ok {
		var err error
		if b, err = json.Marshal(body); err != nil {
			t.Fatal(err)
		}
	}
	req, err := http.NewRequest(http.MethodPost, srv.URL+path, bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	return send(t, req)
}

func get(t *testing.T, srv *httptest.Server, path string) (int, http.Header, map[string]any) {
	t.Helper()
	req, err := http.NewRequest(http.MethodGet, srv.URL+path, nil)
	if err != nil {
		t.Fatal(err)
	}
	return send(t, req)
}

func obj(v any) map[string]any { return v.(map[string]any) }

func list(v any) []any { return v.([]any) }

func nums(v any) []float64 {
	var out []float64
	for _, x := range list(v) {
		out = append(out, x.(float64))
	}
	return out
}

func rain(scope any) map[string]any {
	return map[string]any{"id": "c1", "type": "rain", "value": 1.2, "scope": scope}
}

func dayScope() map[string]any {
	return map[string]any{"routes": []int{7}, "dates": []string{"2025-11-11"}, "weekdays": nil, "hours": []int{16, 22}}
}

func dayBody(conds ...any) map[string]any {
	if conds == nil {
		conds = []any{}
	}
	return map[string]any{"routes": []int{7}, "from": "2025-11-11", "to": "2025-11-11", "granularity": "hour", "horizon": "day", "corridor": true, "conditions": conds}
}

func with(body map[string]any, kv ...any) map[string]any {
	out := map[string]any{}
	for k, v := range body {
		out[k] = v
	}
	for i := 0; i < len(kv); i += 2 {
		out[kv[i].(string)] = kv[i+1]
	}
	return out
}

func mustOK(t *testing.T, status int, body map[string]any) {
	t.Helper()
	if status != http.StatusOK {
		t.Fatalf("status %d: %v", status, body)
	}
}

func checkCorridor(t *testing.T, s map[string]any) {
	t.Helper()
	base, lo, hi := nums(s["base"]), nums(s["lo"]), nums(s["hi"])
	for i := range base {
		if lo[i] > base[i] || base[i] > hi[i] {
			t.Fatalf("point %d: lo %v base %v hi %v", i, lo[i], base[i], hi[i])
		}
	}
}

func TestHealthzWithoutAuth(t *testing.T) {
	status, _, body := get(t, closed, "/healthz")
	mustOK(t, status, body)
	models := obj(body["models"])
	if body["status"] != "ok" || models["catboost"] == nil || models["cnn"] == nil || body["bundle"] != models["catboost"] {
		t.Fatalf("healthz %v", body)
	}
}

func TestAuthRequired(t *testing.T) {
	status, h, body := post(t, closed, "/api/v1/forecast", dayBody())
	if status != http.StatusUnauthorized || obj(body["error"])["code"] != "unauthorized" || h.Get("WWW-Authenticate") == "" {
		t.Fatalf("status %d, %v, %v", status, h, body)
	}
	b, _ := json.Marshal(dayBody())
	req, _ := http.NewRequest(http.MethodPost, closed.URL+"/api/v1/forecast", bytes.NewReader(b))
	req.SetBasicAuth("jury", "secret")
	status, _, body = send(t, req)
	mustOK(t, status, body)
}

func TestForecastDayWithoutConditions(t *testing.T) {
	status, h, body := post(t, open, "/api/v1/forecast", dayBody())
	mustOK(t, status, body)
	if body["start"] != "2025-11-11T00:00" || body["step"] != "1h" {
		t.Fatalf("start %v step %v", body["start"], body["step"])
	}
	series := list(body["series"])
	if len(series) != 1 {
		t.Fatalf("%d series", len(series))
	}
	s := obj(series[0])
	for _, k := range []string{"base", "value", "usual", "lo", "hi"} {
		if n := len(list(s[k])); n != 24 {
			t.Fatalf("%s has %d points", k, n)
		}
	}
	base, value := nums(s["base"]), nums(s["value"])
	if !slices.Equal(base, value) {
		t.Fatalf("value %v differs from base %v", value, base)
	}
	checkCorridor(t, s)
	sum := 0.0
	for _, x := range base {
		sum += x
	}
	if math.Abs(s["base_total"].(float64)-sum) > 24 {
		t.Fatalf("base_total %v, sum %v", s["base_total"], sum)
	}
	if len(list(body["conditions"])) != 0 || len(list(body["warnings"])) != 0 {
		t.Fatalf("conditions %v warnings %v", body["conditions"], body["warnings"])
	}
	if !strings.Contains(h.Get("Server-Timing"), "model;dur=") {
		t.Fatalf("Server-Timing %q", h.Get("Server-Timing"))
	}
	if m := obj(body["meta"])["model"]; m != "catboost_cyclic_service" {
		t.Fatalf("meta.model %v", m)
	}
}

func TestRainAppliedOnDay(t *testing.T) {
	status, _, body := post(t, open, "/api/v1/forecast", dayBody(rain(dayScope())))
	mustOK(t, status, body)
	c := obj(list(body["conditions"])[0])
	if c["applied"] != true || c["points"] != 7.0 || obj(c["passport"])["effect_pct"] != -5.4 || c["contribution_pct"].(float64) >= 0 {
		t.Fatalf("condition %v", c)
	}
	prof := nums(c["profile"])
	if len(prof) != 24 {
		t.Fatalf("profile %v", prof)
	}
	s := obj(list(body["series"])[0])
	base, value := nums(s["base"]), nums(s["value"])
	for h := range 24 {
		in := h >= 16 && h <= 22
		if (prof[h] != 0) != in {
			t.Fatalf("profile %v", prof)
		}
		switch {
		case in && base[h] > 0 && value[h] >= base[h]:
			t.Fatalf("hour %d: value %v base %v", h, value[h], base[h])
		case !in && value[h] != base[h]:
			t.Fatalf("hour %d outside scope changed: value %v base %v", h, value[h], base[h])
		}
	}
}

func TestOperationalDeferredOnMonth(t *testing.T) {
	status, _, body := post(t, open, "/api/v1/forecast", map[string]any{
		"from": "2025-12-01", "to": "2025-12-31", "granularity": "day", "horizon": "month",
		"conditions": []any{map[string]any{"id": "c1", "type": "rain", "value": 1.2}},
	})
	mustOK(t, status, body)
	series := list(body["series"])
	if len(series) != 10 {
		t.Fatalf("%d series", len(series))
	}
	for _, x := range series {
		s := obj(x)
		if len(list(s["value"])) != 31 || !slices.Equal(nums(s["value"]), nums(s["base"])) {
			t.Fatalf("route %v: value %v base %v", s["route"], s["value"], s["base"])
		}
	}
	c := obj(list(body["conditions"])[0])
	if c["applied"] != false || c["factor"] != 1.0 {
		t.Fatalf("condition %v", c)
	}
	codes := map[string]any{}
	for _, w := range list(body["warnings"]) {
		codes[obj(w)["code"].(string)] = obj(w)["condition_id"]
	}
	if codes["condition_not_applicable_on_horizon"] != "c1" {
		t.Fatalf("warnings %v", body["warnings"])
	}
	if id, ok := codes["route_without_history"]; !ok || id != nil {
		t.Fatalf("warnings %v", body["warnings"])
	}
}

func TestLimitsAndErrors(t *testing.T) {
	many := make([]any, 17)
	for i := range many {
		many[i] = map[string]any{"id": fmt.Sprint("c", i), "type": "rain", "value": 1}
	}
	all := map[string]any{"from": "2025-01-01", "to": "2026-04-30", "granularity": "day", "horizon": "season"}
	cases := []struct {
		name   string
		body   any
		status int
		code   string
		field  string
	}{
		{"out of domain", with(dayBody(), "to", "2026-07-01"), 400, "out_of_domain", "to"},
		{"hourly too long", with(dayBody(), "from", "2025-11-01", "to", "2025-12-31"), 413, "period_too_long", "granularity"},
		{"too many conditions", with(dayBody(), "conditions", many), 413, "too_many_conditions", "conditions"},
		{"too many rows", all, 413, "too_many_rows", "to"},
		{"unknown route", with(dayBody(), "routes", []int{3}), 400, "bad_request", "routes"},
		{"not json", []byte("{"), 400, "bad_request", ""},
		{"granularity", with(dayBody(), "granularity", "week"), 400, "bad_request", "granularity"},
		{"rain value", dayBody(map[string]any{"id": "c1", "type": "rain", "value": 9}), 400, "bad_request", "conditions[0].value"},
		{"unknown model", with(dayBody(), "model", "mlp"), 400, "bad_request", "model"},
		{"cnn domain", with(dayBody(), "model", "cnn", "from", "2026-05-01", "to", "2026-05-01"), 400, "out_of_domain", "from"},
		{"trailing spaces", append(must(json.Marshal(dayBody())), bytes.Repeat([]byte(" "), 3<<20)...), 413, "bad_request", ""},
		{"trailing garbage", append(must(json.Marshal(dayBody())), []byte("garbage")...), 400, "bad_request", ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			status, h, body := post(t, open, "/api/v1/forecast", c.body)
			e, _ := body["error"].(map[string]any)
			if status != c.status || e["code"] != c.code || c.field != "" && e["field"] != c.field {
				t.Fatalf("status %d, error %v", status, e)
			}
			if e["message"] == "" || !strings.HasPrefix(h.Get("Content-Type"), "application/json") {
				t.Fatalf("error %v, content type %q", e, h.Get("Content-Type"))
			}
			if c.name == "cnn domain" && !strings.Contains(e["message"].(string), "cnn") {
				t.Fatalf("message %q", e["message"])
			}
		})
	}
}

func TestSeasonForNetworkFits(t *testing.T) {
	status, _, body := post(t, open, "/api/v1/forecast", map[string]any{"from": "2025-11-01", "to": "2026-04-30", "granularity": "day", "horizon": "season"})
	mustOK(t, status, body)
	series := list(body["series"])
	if len(series) != 10 || len(list(obj(series[0])["value"])) != 181 {
		t.Fatalf("%d series", len(series))
	}
}

func stepNames(body map[string]any) []string {
	var out []string
	for _, s := range list(body["steps"]) {
		out = append(out, obj(s)["step"].(string))
	}
	return out
}

func step(body map[string]any, name string) map[string]any {
	for _, s := range list(body["steps"]) {
		if obj(s)["step"] == name {
			return obj(s)
		}
	}
	return nil
}

func TestExplainMatchesForecastCell(t *testing.T) {
	status, _, fc := post(t, open, "/api/v1/forecast", dayBody(rain(dayScope())))
	mustOK(t, status, fc)
	status, _, ex := post(t, open, "/api/v1/explain", map[string]any{"route": 7, "date": "2025-11-11", "hour": 18, "horizon": "day", "conditions": []any{rain(dayScope())}})
	mustOK(t, status, ex)
	if want := nums(obj(list(fc["series"])[0])["value"])[18]; ex["value"] != want {
		t.Fatalf("explain %v, forecast %v", ex["value"], want)
	}
	names := stepNames(ex)
	if len(names) < 3 || !slices.Equal(names[:3], []string{"сырой выход модели", "обрезка нуля", "калибровка"}) {
		t.Fatalf("steps %v", names)
	}
	if s := step(ex, "условие c1"); s == nil || s["factor"] == nil {
		t.Fatalf("steps %v", ex["steps"])
	}
	feats := list(ex["features"])
	if len(feats) != 16 {
		t.Fatalf("%d features", len(feats))
	}
	for _, f := range feats {
		if obj(f)["name"] == "route" && (obj(f)["kind"] != "categorical" || obj(f)["hash"] == nil) {
			t.Fatalf("route feature %v", f)
		}
	}
}

func TestExplainRoute5(t *testing.T) {
	status, _, ex := post(t, open, "/api/v1/explain", map[string]any{"route": 5, "date": "2025-12-31", "hour": 20})
	mustOK(t, status, ex)
	if names := stepNames(ex); len(names) == 0 || names[0] != "маршрут 5" || step(ex, "новогодний блок") == nil || len(list(ex["features"])) != 0 {
		t.Fatalf("features %v, steps %v", ex["features"], names)
	}
	status, _, fc := post(t, open, "/api/v1/forecast", map[string]any{"routes": []int{5}, "from": "2025-12-31", "to": "2025-12-31", "granularity": "hour", "horizon": "day"})
	mustOK(t, status, fc)
	if want := nums(obj(list(fc["series"])[0])["value"])[20]; ex["value"] != want {
		t.Fatalf("explain %v, forecast %v", ex["value"], want)
	}
}

func TestCNNEqualToCatBoostInContract(t *testing.T) {
	status, _, fc := post(t, open, "/api/v1/forecast", with(dayBody(rain(dayScope())), "model", "cnn"))
	mustOK(t, status, fc)
	if m := obj(fc["meta"])["model"]; m != "cnn_14days_v9_events" {
		t.Fatalf("meta.model %v", m)
	}
	s := obj(list(fc["series"])[0])
	for _, k := range []string{"base", "value", "usual", "lo", "hi"} {
		if n := len(list(s[k])); n != 24 {
			t.Fatalf("%s has %d points", k, n)
		}
	}
	checkCorridor(t, s)
	if c := obj(list(fc["conditions"])[0]); c["applied"] != true || c["points"] != 7.0 {
		t.Fatalf("condition %v", c)
	}
	status, _, ex := post(t, open, "/api/v1/explain", map[string]any{"model": "cnn", "route": 7, "date": "2025-11-11", "hour": 18, "conditions": []any{rain(dayScope())}})
	mustOK(t, status, ex)
	if want := nums(s["value"])[18]; ex["value"] != want {
		t.Fatalf("explain %v, forecast %v", ex["value"], want)
	}
	if step(ex, "калибровка") != nil {
		t.Fatalf("steps %v", stepNames(ex))
	}
	feats := list(ex["features"])
	if len(feats) != 11 || obj(feats[10])["name"] != "lead" {
		t.Fatalf("features %v", feats)
	}
}

func TestReferenceEndpoints(t *testing.T) {
	status, _, body := get(t, open, "/api/v1/conditions")
	mustOK(t, status, body)
	if n := len(list(body["conditions"])); n != 6 {
		t.Fatalf("%d conditions", n)
	}
	status, _, body = get(t, open, "/api/v1/routes")
	mustOK(t, status, body)
	routes := list(body["routes"])
	if len(routes) != 10 {
		t.Fatalf("%d routes", len(routes))
	}
	for _, r := range routes {
		r := obj(r)
		if r["has_geometry"] != true || (r["has_history"] == false) != (r["route"] == 5.0) {
			t.Fatalf("route %v", r)
		}
	}
	status, _, body = get(t, open, "/api/v1/model")
	mustOK(t, status, body)
	models := list(body["models"])
	if body["default"] != "catboost" || len(models) != 2 {
		t.Fatalf("model %v", body)
	}
	cb := obj(models[0])
	if cb["name"] != "catboost" || len(list(cb["features"])) != 16 || cb["mode"] != "service" {
		t.Fatalf("catboost %v", cb)
	}
}

func TestExplainRoute5CNN(t *testing.T) {
	status, _, ex := post(t, open, "/api/v1/explain", map[string]any{"model": "cnn", "route": 5, "date": "2025-12-31", "hour": 20})
	mustOK(t, status, ex)
	names := stepNames(ex)
	if len(list(ex["features"])) != 0 || len(names) == 0 || names[0] != "маршрут 5" || step(ex, "новогодний блок") != nil {
		t.Fatalf("features %v, steps %v", ex["features"], names)
	}
	status, _, fc := post(t, open, "/api/v1/forecast", map[string]any{"model": "cnn", "routes": []int{5}, "from": "2025-12-31", "to": "2025-12-31", "granularity": "hour", "horizon": "day"})
	mustOK(t, status, fc)
	if want := nums(obj(list(fc["series"])[0])["value"])[20]; ex["value"] != want {
		t.Fatalf("explain %v, forecast %v", ex["value"], want)
	}
}

func export(t *testing.T, format string, body any) (*http.Response, []byte) {
	t.Helper()
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.Post(open.URL+"/api/v1/forecast/export?format="+format, "application/json", bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	out, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", resp.StatusCode, out)
	}
	return resp, out
}

func TestExportCSV(t *testing.T) {
	resp, body := export(t, "csv", dayBody(rain(dayScope())))
	if cd := resp.Header.Get("Content-Disposition"); !strings.Contains(cd, `filename="forecast_catboost_2025-11-11_r7.csv"`) {
		t.Fatalf("Content-Disposition %q", cd)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/csv; charset=utf-8" {
		t.Fatalf("Content-Type %q", ct)
	}
	lines := strings.Split(strings.TrimSpace(string(body)), "\n")
	if len(lines) != 25 || lines[0] != "route;date;hour;base;factor;prediction;lo;hi" {
		t.Fatalf("%d lines, header %q", len(lines), lines[0])
	}
	if !strings.HasPrefix(lines[1], "7;2025-11-11;0;") {
		t.Fatalf("first row %q", lines[1])
	}
	status, _, fc := post(t, open, "/api/v1/forecast", dayBody(rain(dayScope())))
	mustOK(t, status, fc)
	s := obj(list(fc["series"])[0])
	base, value := nums(s["base"]), nums(s["value"])
	for h := range 24 {
		f := strings.Split(lines[h+1], ";")
		if f[3] != fmt.Sprint(base[h]) || f[5] != fmt.Sprint(value[h]) {
			t.Fatalf("hour %d: row %q, forecast base %v value %v", h, lines[h+1], base[h], value[h])
		}
		if factor, _ := strconv.ParseFloat(f[4], 64); h >= 16 && h <= 22 && base[h] > 0 && factor >= 1 {
			t.Fatalf("hour %d inside the rain: factor %v", h, factor)
		}
	}
}

func TestExportXLSX(t *testing.T) {
	resp, body := export(t, "xlsx", dayBody(rain(dayScope())))
	if cd := resp.Header.Get("Content-Disposition"); !strings.Contains(cd, `filename="forecast_catboost_2025-11-11_r7.xlsx"`) {
		t.Fatalf("Content-Disposition %q", cd)
	}
	z, err := zip.NewReader(bytes.NewReader(body), int64(len(body)))
	if err != nil {
		t.Fatal(err)
	}
	names := map[string]bool{}
	for _, f := range z.File {
		names[f.Name] = true
	}
	if !names["xl/worksheets/sheet1.xml"] || !names["xl/worksheets/sheet2.xml"] {
		t.Fatalf("parts %v", names)
	}
}

func TestExportRejectsFormat(t *testing.T) {
	status, _, body := post(t, open, "/api/v1/forecast/export?format=pdf", dayBody())
	if e := obj(body["error"]); status != http.StatusBadRequest || e["code"] != "bad_request" || e["field"] != "format" {
		t.Fatalf("status %d, %v", status, body)
	}
}

func TestStatsCounts(t *testing.T) {
	status, _, body := post(t, open, "/api/v1/forecast", dayBody())
	mustOK(t, status, body)
	status, _, st := get(t, open, "/api/v1/stats")
	mustOK(t, status, st)
	if st["requests"].(float64) < 1 || st["rows"].(float64) < 24 || st["p95_ms"].(float64) < st["p50_ms"].(float64) {
		t.Fatalf("stats %v", st)
	}
	for _, k := range []string{"errors", "rps_1m", "cpu_pct", "rss_mb", "uptime_s"} {
		if _, ok := st[k].(float64); !ok {
			t.Fatalf("stats %v: no %s", st, k)
		}
	}
}

func must(b []byte, err error) []byte {
	if err != nil {
		panic(err)
	}
	return b
}

func TestTrailingDataInExplain(t *testing.T) {
	b := must(json.Marshal(map[string]any{"route": 7, "date": "2025-11-11", "hour": 18}))
	status, _, body := post(t, open, "/api/v1/explain", append(b, []byte(`{"route":1}`)...))
	if e, _ := body["error"].(map[string]any); status != http.StatusBadRequest || e["code"] != "bad_request" {
		t.Fatalf("status %d, %v", status, body)
	}
}

func TestAuthNeedsLoginAndPassword(t *testing.T) {
	for _, auth := range []string{":secret", "jury:"} {
		if _, err := api.New(loaded, catalog, api.Config{GeoPath: "../../../web/geo/routes.geojson", Auth: auth}); err == nil {
			t.Fatalf("auth %q accepted", auth)
		}
	}
}

func TestTinyEffectIsNotNegativeZero(t *testing.T) {
	status, _, body := post(t, open, "/api/v1/forecast", dayBody(map[string]any{"id": "c1", "type": "temperature", "value": -0.01, "scope": dayScope()}))
	mustOK(t, status, body)
	c := obj(list(body["conditions"])[0])
	for _, x := range append(nums(c["profile"]), c["contribution_pct"].(float64)) {
		if x == 0 && math.Signbit(x) {
			t.Fatalf("condition %v has -0", c)
		}
	}
}

func TestStaticServesSPA(t *testing.T) {
	dir := t.TempDir()
	for name, text := range map[string]string{"index.html": "<!doctype html>spa", "app.js": "app()"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(text), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	s, err := api.New(loaded, catalog, api.Config{GeoPath: "../../../web/geo/routes.geojson", Static: dir})
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(s.Handler())
	defer srv.Close()
	cases := []struct {
		method, path string
		status       int
		body         string
	}{
		{"GET", "/routes/7", 200, "<!doctype html>spa"},
		{"GET", "/", 200, "<!doctype html>spa"},
		{"GET", "/app.js", 200, "app()"},
		{"GET", "/api/v1/nope", 404, ""},
		{"GET", "/geo/nope", 404, ""},
		{"GET", "/api/v1/forecast", 405, ""},
		{"POST", "/api/v1/forecast", 400, ""},
	}
	for _, c := range cases {
		req, _ := http.NewRequest(c.method, srv.URL+c.path, nil)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != c.status || c.body != "" && string(b) != c.body {
			t.Fatalf("%s %s: status %d, body %q", c.method, c.path, resp.StatusCode, b)
		}
	}
}

func TestStaticNeedsIndex(t *testing.T) {
	geo := "../../../web/geo/routes.geojson"
	for _, dir := range []string{t.TempDir(), filepath.Join(t.TempDir(), "absent")} {
		if _, err := api.New(loaded, catalog, api.Config{GeoPath: geo, Static: dir}); err == nil {
			t.Fatalf("static %s without index.html accepted", dir)
		}
	}
}
