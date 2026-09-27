package main

import (
	"cmp"
	"flag"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"stand/internal/api"
	"stand/internal/bundle"
	"stand/internal/catboost"
	"stand/internal/cnn"
	"stand/internal/conditions"
	"stand/internal/features"
	"stand/internal/forecast"
)

func main() {
	addr := flag.String("addr", ":8080", "адрес HTTP-сервера")
	root := flag.String("bundle", "../artifacts/bundle", "папка бандлов: conditions.json и папки моделей")
	calDir := flag.String("calendar", "../input/calendar", "папка производственных календарей")
	geo := flag.String("geo", "../web/geo/routes.geojson", "GeoJSON маршрутов")
	auth := flag.String("auth", "", "basic auth в виде логин:пароль, пусто - без авторизации")
	static := flag.String("static", "", "папка собранного интерфейса, отдаётся на GET /; пусто - без интерфейса")
	flag.Parse()

	catalog, err := conditions.LoadCatalog(filepath.Join(*root, "conditions.json"))
	if err != nil {
		log.Fatal(err)
	}
	cal, err := features.LoadCalendars(*calDir)
	if err != nil {
		log.Fatal(err)
	}
	var engines []*forecast.Engine
	for _, name := range []string{"catboost", "cnn"} {
		dir := filepath.Join(*root, name)
		if _, err := os.Stat(dir); err != nil {
			log.Printf("модель %s: нет папки %s, пропускаю", name, dir)
			continue
		}
		e, err := load(name, dir, cal)
		if err != nil {
			log.Fatal(err)
		}
		c, err := e.Verify()
		if err != nil {
			log.Fatalf("модель %s: %v", name, err)
		}
		if err := e.CheckHorizon(); err != nil {
			log.Fatalf("модель %s: %v", name, err)
		}
		log.Printf("модель %s: бандл %s, %s, сверка с эталоном на %d строках пройдена (max |raw| %.3g)",
			name, e.Bundle.Meta.Version, e.Model.Describe(), c.Rows, c.MaxRaw)
		engines = append(engines, e)
	}
	if len(engines) == 0 {
		log.Fatalf("в %s нет ни одной модели", *root)
	}
	s, err := api.New(engines, catalog, api.Config{GeoPath: *geo, Auth: *auth, Static: *static})
	if err != nil {
		log.Fatal(err)
	}
	if *auth == "" {
		log.Print("авторизация выключена: задайте -auth или STAND_AUTH")
	}
	url := *addr
	if host, port, err := net.SplitHostPort(*addr); err == nil {
		url = net.JoinHostPort(cmp.Or(host, "localhost"), port)
	}
	log.Printf("GOMAXPROCS=%d; http://%s", runtime.GOMAXPROCS(0), url)
	srv := &http.Server{Addr: *addr, Handler: s.Handler(), ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 15 * time.Second,
		WriteTimeout: 30 * time.Second, IdleTimeout: 60 * time.Second}
	log.Fatal(srv.ListenAndServe())
}

func load(name, dir string, cal *features.Calendar) (*forecast.Engine, error) {
	b, err := bundle.Load(dir)
	if err != nil {
		return nil, err
	}
	var m forecast.Model
	if name == "catboost" {
		m, err = catboost.Load(dir)
	} else {
		m, err = cnn.Load(dir)
	}
	if err != nil {
		return nil, err
	}
	return forecast.New(name, b, cal, m)
}
