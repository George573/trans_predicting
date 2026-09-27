package api

import (
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

type sample struct {
	at  time.Time
	dur time.Duration
}

type second struct {
	unix int64
	n    int
}

type Stats struct {
	mu                     sync.Mutex
	start                  time.Time
	ring                   [4096]sample
	perSecond              [60]second
	samples                int
	requests, errors, rows int64
	lastWall               time.Time
	lastCPU                time.Duration
}

func newStats() *Stats {
	now := time.Now()
	return &Stats{start: now, lastWall: now, lastCPU: cpuTime()}
}

func (st *Stats) record(d time.Duration, rows int) {
	st.mu.Lock()
	defer st.mu.Unlock()
	now := time.Now()
	st.ring[st.samples%len(st.ring)] = sample{now, d}
	if sec := &st.perSecond[now.Unix()%60]; sec.unix != now.Unix() {
		*sec = second{now.Unix(), 1}
	} else {
		sec.n++
	}
	st.samples++
	st.requests++
	st.rows += int64(rows)
}

func (st *Stats) fail() {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.errors++
}

func (st *Stats) snapshot() map[string]any {
	now, cpu := time.Now(), cpuTime()
	st.mu.Lock()
	defer st.mu.Unlock()
	var recent []time.Duration
	for _, s := range st.ring[:min(st.samples, len(st.ring))] {
		if now.Sub(s.at) <= time.Minute {
			recent = append(recent, s.dur)
		}
	}
	slices.Sort(recent)
	lastMinute := 0
	for _, sec := range st.perSecond {
		if now.Unix()-sec.unix < 60 {
			lastMinute += sec.n
		}
	}
	quantile := func(p float64) float64 {
		if len(recent) == 0 {
			return 0
		}
		return ms(recent[int(p*float64(len(recent)-1))])
	}
	cpuPct := 0.0
	if wall := now.Sub(st.lastWall); wall > 0 {
		cpuPct = roundTo(100*float64(cpu-st.lastCPU)/float64(wall), 1)
	}
	st.lastWall, st.lastCPU = now, cpu
	return map[string]any{
		"requests": st.requests, "errors": st.errors, "rows": st.rows,
		"rps_1m": roundTo(float64(lastMinute)/60, 2), "p50_ms": quantile(0.5), "p95_ms": quantile(0.95),
		"cpu_pct": cpuPct, "rss_mb": roundTo(float64(rss())/(1<<20), 1), "uptime_s": roundTo(now.Sub(st.start).Seconds(), 1),
	}
}

func cpuTime() time.Duration {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return 0
	}
	return time.Duration(ru.Utime.Nano() + ru.Stime.Nano())
}

func rss() uint64 {
	if b, err := os.ReadFile("/proc/self/statm"); err == nil {
		if f := strings.Fields(string(b)); len(f) > 1 {
			if pages, err := strconv.ParseUint(f[1], 10, 64); err == nil {
				return pages * uint64(os.Getpagesize())
			}
		}
	}
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.Sys
}
