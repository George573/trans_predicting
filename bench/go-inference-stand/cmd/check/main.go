// Сверка Go-инференса с Python (ref_grid.csv из scripts/train_cyclic.py) и микробенчмарк.
package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"time"

	"stand/internal/cbfast"
	"stand/internal/features"
)

func main() {
	cal, err := features.LoadCalendar("../../input/calendar/2025.xml")
	if err != nil {
		log.Fatal(err)
	}
	m, err := cbfast.Load("model_cyclic.cbm")
	if err != nil {
		log.Fatal(err)
	}
	f, err := os.Open("ref_grid.csv")
	if err != nil {
		log.Fatal(err)
	}
	recs, _ := csv.NewReader(f).ReadAll()
	recs = recs[1:] // route,date_str,hour,cb
	rows := make([]features.Row, len(recs))
	ref := make([]float64, len(recs))
	for i, r := range recs {
		route, _ := strconv.Atoi(r[0])
		h, _ := strconv.Atoi(r[2])
		d, _ := time.Parse("2006-01-02", r[1])
		rows[i] = cal.Build(route, d, h)
		ref[i], _ = strconv.ParseFloat(r[3], 64)
	}
	t := time.Now()
	p, err := m.Predict(rows)
	if err != nil {
		log.Fatal(err)
	}
	el := time.Since(t)
	md, sd, s := 0.0, 0.0, 0.0
	for i := range p {
		d := math.Abs(math.Max(p[i], 0) - ref[i]) // Python: np.clip(pred, 0, None)
		md = math.Max(md, d)
		sd += d
		s += ref[i]
	}
	fmt.Printf("%s, %d деревьев, %d числовых признаков\n", m.Name(), m.Trees, m.NumFloat())
	fmt.Printf("вся сетка %d строк за %v: max|Go - Python| = %.3g, sum|diff|/sum = %.2e\n", len(p), el.Round(time.Microsecond), md, sd/s)
	for _, n := range []int{24, 240, 720, 7440} {
		t := time.Now()
		it := 0
		for time.Since(t) < 500*time.Millisecond {
			m.Predict(rows[:n])
			it++
		}
		us := float64(time.Since(t).Microseconds()) / float64(it)
		fmt.Printf("  батч %5d строк: %8.1f us (%.2f us/строку)\n", n, us, us/float64(n))
	}
}
