package api

import (
	"bytes"
	"cmp"
	"compress/gzip"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"log"
	"maps"
	"math"
	"mime"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"stand/internal/conditions"
	"stand/internal/features"
	"stand/internal/forecast"
)

const jsonType = "application/json; charset=utf-8"

type Config struct {
	GeoPath, Auth, Static string
}

type Server struct {
	engines    []*forecast.Engine
	catalog    *conditions.Catalog
	user, pass string
	geo        asset
	routes     []routeOut
	stats      *Stats
	static     map[string]asset
}

type apiError struct {
	status  int
	Code    string `json:"code"`
	Message string `json:"message"`
	Field   string `json:"field,omitempty"`
}

type routeOut struct {
	Route          int    `json:"route"`
	Name           string `json:"name"`
	HasGeometry    bool   `json:"has_geometry"`
	HasHistory     bool   `json:"has_history"`
	GeometrySource string `json:"geometry_source"`
}

func (e *apiError) Error() string { return e.Message }

func New(engines []*forecast.Engine, catalog *conditions.Catalog, cfg Config) (*Server, error) {
	if len(engines) == 0 {
		return nil, errors.New("нет ни одной модели")
	}
	s := &Server{engines: engines, catalog: catalog, stats: newStats()}
	if cfg.Auth != "" {
		var ok bool
		if s.user, s.pass, ok = strings.Cut(cfg.Auth, ":"); !ok || s.user == "" || s.pass == "" {
			return nil, errors.New("auth должен быть в виде логин:пароль")
		}
	}
	var err error
	if cfg.Static != "" {
		if s.static, err = loadStatic(cfg.Static); err != nil {
			return nil, err
		}
	}
	raw, err := os.ReadFile(cfg.GeoPath)
	if err != nil {
		return nil, err
	}
	s.geo = newAsset(raw, time.Now())
	var geo struct {
		Features []struct {
			Properties struct {
				Kind   string `json:"kind"`
				Route  int    `json:"route"`
				Name   string `json:"name"`
				Source string `json:"source"`
			} `json:"properties"`
		} `json:"features"`
	}
	if err := json.Unmarshal(raw, &geo); err != nil {
		return nil, fmt.Errorf("%s: %w", cfg.GeoPath, err)
	}
	for _, r := range features.Routes {
		out := routeOut{Route: r, Name: "Маршрут " + strconv.Itoa(r), HasHistory: r != 5, GeometrySource: "none"}
		for _, f := range geo.Features {
			if p := f.Properties; p.Kind == "line" && p.Route == r {
				out.Name, out.HasGeometry, out.GeometrySource = p.Name, true, p.Source
				break
			}
		}
		s.routes = append(s.routes, out)
	}
	return s, nil
}

func (s *Server) Handler() http.Handler {
	api := http.NewServeMux()
	api.HandleFunc("POST /api/v1/forecast", s.forecast)
	api.HandleFunc("POST /api/v1/forecast/export", s.export)
	api.HandleFunc("POST /api/v1/explain", s.explain)
	api.HandleFunc("GET /api/v1/stats", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, s.stats.snapshot())
	})
	api.HandleFunc("GET /api/v1/conditions", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{"conditions": s.catalog.Conditions})
	})
	api.HandleFunc("GET /api/v1/routes", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{"routes": s.routes})
	})
	api.HandleFunc("GET /api/v1/model", s.model)
	api.HandleFunc("GET /geo/routes.geojson", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/geo+json")
		s.geo.serve(w, r, "routes.geojson")
	})
	var h http.Handler = api
	if s.static != nil {
		h = spa(s.static, api)
	}
	root := http.NewServeMux()
	root.HandleFunc("GET /healthz", s.healthz)
	root.Handle("/", s.auth(h))
	return root
}

func (s *Server) auth(next http.Handler) http.Handler {
	if s.user == "" {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		u, p, ok := r.BasicAuth()
		if !ok || subtle.ConstantTimeCompare([]byte(u), []byte(s.user))&subtle.ConstantTimeCompare([]byte(p), []byte(s.pass)) != 1 {
			w.Header().Set("WWW-Authenticate", `Basic realm="forecast", charset="UTF-8"`)
			s.fail(w, &apiError{status: http.StatusUnauthorized, Code: "unauthorized", Message: "Нужен логин и пароль"})
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) healthz(w http.ResponseWriter, r *http.Request) {
	models := map[string]string{}
	for _, e := range s.engines {
		models[e.Name] = e.Bundle.Meta.Version
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "bundle": s.engines[0].Bundle.Meta.Version, "models": models})
}

func (s *Server) model(w http.ResponseWriter, r *http.Request) {
	var models []map[string]any
	for _, e := range s.engines {
		m := maps.Clone(e.Bundle.MetaRaw)
		m["name"], m["bundle"], m["describe"], m["files"] = e.Name, e.Bundle.Meta.Version, e.Model.Describe(), e.Bundle.Files
		models = append(models, m)
	}
	writeJSON(w, http.StatusOK, map[string]any{"default": s.engines[0].Name, "models": models})
}

func (s *Server) fail(w http.ResponseWriter, err error) {
	s.stats.fail()
	var e *apiError
	if !errors.As(err, &e) {
		log.Printf("внутренняя ошибка: %v", err)
		e = &apiError{status: http.StatusInternalServerError, Code: "internal", Message: "Внутренняя ошибка сервиса"}
	}
	writeJSON(w, e.status, map[string]any{"error": e})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	b, err := json.Marshal(v)
	if err != nil {
		log.Printf("ответ не сериализуется в JSON: %v", err)
		status, b = http.StatusInternalServerError, []byte(`{"error":{"code":"internal","message":"Внутренняя ошибка сервиса"}}`)
	}
	w.Header().Set("Content-Type", jsonType)
	w.WriteHeader(status)
	w.Write(b)
}

func ms(d time.Duration) float64 { return roundTo(float64(d)/float64(time.Millisecond), 3) }

func roundTo(v float64, digits int) float64 {
	p := math.Pow(10, float64(digits))
	if r := math.Round(v*p) / p; r != 0 {
		return r
	}
	return 0
}

func ints(v []float64) []int64 {
	out := make([]int64, len(v))
	for i, x := range v {
		out[i] = int64(math.Round(x))
	}
	return out
}

func rounded(in []conditions.Applied) []conditions.Applied {
	out := slices.Clone(in)
	for i := range out {
		c := &out[i]
		c.Factor, c.ContributionPct = roundTo(c.Factor, 4), roundTo(c.ContributionPct, 2)
		if c.Profile != nil {
			c.Profile = slices.Clone(c.Profile)
			for h := range c.Profile {
				c.Profile[h] = roundTo(c.Profile[h], 2)
			}
		}
	}
	return out
}

func spa(files map[string]asset, api http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead || strings.HasPrefix(r.URL.Path, "/api/") || strings.HasPrefix(r.URL.Path, "/geo/") {
			api.ServeHTTP(w, r)
			return
		}
		name := path.Clean("/" + r.URL.Path)
		a, ok := files[name]
		switch {
		case strings.HasPrefix(name, "/assets/") && !ok:
			http.NotFound(w, r)
			return
		case strings.HasPrefix(name, "/assets/"):
			w.Header().Set("Cache-Control", "max-age=31536000, immutable")
		default:
			if !ok {
				name, a = "/index.html", files["/index.html"]
			}
			w.Header().Set("Cache-Control", "no-cache")
		}
		a.serve(w, r, name)
	})
}

type asset struct {
	raw, gz []byte
	mod     time.Time
}

func newAsset(raw []byte, mod time.Time) asset {
	var b bytes.Buffer
	zw, _ := gzip.NewWriterLevel(&b, gzip.BestCompression)
	zw.Write(raw)
	zw.Close()
	a := asset{raw: raw, mod: mod}
	if b.Len() < len(raw)*9/10 {
		a.gz = b.Bytes()
	}
	return a
}

func loadStatic(dir string) (map[string]asset, error) {
	files := map[string]asset{}
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		raw, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(dir, p)
		if err != nil {
			return err
		}
		files["/"+filepath.ToSlash(rel)] = newAsset(raw, info.ModTime())
		return nil
	})
	if _, ok := files["/index.html"]; err != nil || !ok {
		return nil, fmt.Errorf("папка интерфейса %s: нет index.html", dir)
	}
	return files, nil
}

func (a asset) serve(w http.ResponseWriter, r *http.Request, name string) {
	body := a.raw
	if a.gz != nil {
		w.Header().Add("Vary", "Accept-Encoding")
		if acceptsGzip(r) {
			if w.Header().Get("Content-Type") == "" {
				w.Header().Set("Content-Type", cmp.Or(mime.TypeByExtension(path.Ext(name)), http.DetectContentType(a.raw)))
			}
			w.Header().Set("Content-Encoding", "gzip")
			body = a.gz
		}
	}
	http.ServeContent(w, r, name, a.mod, bytes.NewReader(body))
}

func acceptsGzip(r *http.Request) bool {
	for part := range strings.SplitSeq(r.Header.Get("Accept-Encoding"), ",") {
		name, params, _ := strings.Cut(part, ";")
		if strings.TrimSpace(name) == "gzip" {
			q, ok := strings.CutPrefix(strings.ReplaceAll(params, " ", ""), "q=")
			v, err := strconv.ParseFloat(q, 64)
			return !ok || err != nil || v > 0
		}
	}
	return false
}
