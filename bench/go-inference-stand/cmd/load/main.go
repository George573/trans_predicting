// Нагрузочный генератор без внешних зависимостей.
// -rate > 0: open-loop, запросы уходят по расписанию, задержка считается от запланированного
// момента (без coordinated omission). -rate 0: closed-loop с -c параллельными клиентами.
package main

import (
	"flag"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

func cpuTicks(pid int) (float64, int64) {
	b, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return 0, 0
	}
	f := strings.Fields(string(b)[strings.LastIndex(string(b), ")")+2:])
	u, _ := strconv.ParseFloat(f[11], 64)
	s, _ := strconv.ParseFloat(f[12], 64)
	st, _ := os.ReadFile(fmt.Sprintf("/proc/%d/status", pid))
	var rss int64
	for _, l := range strings.Split(string(st), "\n") {
		if strings.HasPrefix(l, "VmRSS:") {
			rss, _ = strconv.ParseInt(strings.Fields(l)[1], 10, 64)
		}
	}
	return (u + s) / 100, rss
}

func main() {
	urlsFile := flag.String("urls", "", "файл со списком URL (по одному в строке)")
	rate := flag.Int("rate", 0, "RPS для open-loop; 0 = closed-loop")
	conc := flag.Int("c", 32, "клиентов в closed-loop / макс. одновременных в open-loop")
	dur := flag.Duration("d", 20*time.Second, "длительность")
	pid := flag.Int("pid", 0, "pid сервера для замера CPU/RSS")
	auth := flag.String("auth", "", "user:pass")
	label := flag.String("label", "", "подпись строки результата")
	flag.Parse()

	raw, err := os.ReadFile(*urlsFile)
	if err != nil {
		panic(err)
	}
	var urls []string
	for _, l := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		if l != "" {
			urls = append(urls, l)
		}
	}
	user, pass, _ := strings.Cut(*auth, ":")
	tr := &http.Transport{MaxIdleConnsPerHost: *conc * 2, MaxConnsPerHost: 0, DisableCompression: true}
	cl := &http.Client{Transport: tr, Timeout: 10 * time.Second}

	var mu sync.Mutex
	lat := make([]time.Duration, 0, 1<<16)
	var errs, bytes atomic.Int64
	do := func(sched time.Time, u string) {
		req, _ := http.NewRequest("GET", u, nil)
		if user != "" {
			req.SetBasicAuth(user, pass)
		}
		resp, err := cl.Do(req)
		if err == nil {
			n, _ := io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			bytes.Add(n)
			if resp.StatusCode != 200 {
				err = fmt.Errorf("status %d", resp.StatusCode)
			}
		}
		d := time.Since(sched)
		if err != nil {
			errs.Add(1)
		}
		mu.Lock()
		lat = append(lat, d)
		mu.Unlock()
	}

	c0, _ := cpuTicks(*pid)
	start := time.Now()
	end := start.Add(*dur)
	var wg sync.WaitGroup
	var peakRSS int64
	stopRSS := make(chan struct{})
	go func() {
		for {
			select {
			case <-stopRSS:
				return
			case <-time.After(200 * time.Millisecond):
				if _, r := cpuTicks(*pid); r > peakRSS {
					peakRSS = r
				}
			}
		}
	}()
	if *rate > 0 {
		interval := time.Second / time.Duration(*rate)
		sem := make(chan struct{}, *conc*8)
		for i := 0; ; i++ {
			sched := start.Add(time.Duration(i) * interval)
			if sched.After(end) {
				break
			}
			if w := time.Until(sched); w > 0 {
				time.Sleep(w)
			}
			sem <- struct{}{}
			wg.Add(1)
			go func(s time.Time, u string) { defer wg.Done(); do(s, u); <-sem }(sched, urls[rand.Intn(len(urls))])
		}
	} else {
		for w := 0; w < *conc; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for time.Now().Before(end) {
					do(time.Now(), urls[rand.Intn(len(urls))])
				}
			}()
		}
	}
	wg.Wait()
	el := time.Since(start)
	close(stopRSS)
	c1, _ := cpuTicks(*pid)

	sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
	q := func(p float64) float64 {
		if len(lat) == 0 {
			return 0
		}
		return float64(lat[int(float64(len(lat)-1)*p)].Microseconds()) / 1000
	}
	cpu := (c1 - c0) / el.Seconds() * 100
	fmt.Printf("%-34s rps=%8.0f  p50=%7.2fms p95=%7.2fms p99=%7.2fms max=%7.1fms  err=%d  cpu=%4.0f%%  rss=%dMB  avg_resp=%dB\n",
		*label, float64(len(lat))/el.Seconds(), q(.5), q(.95), q(.99), q(1), errs.Load(), cpu, peakRSS/1024,
		bytes.Load()/int64(max(1, len(lat))))
}
