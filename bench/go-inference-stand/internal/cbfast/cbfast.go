// Package cbfast - тонкая cgo-обёртка над libcatboostmodel с хэшированными категориями.
//
// catboost-cgo передаёт категории строками: на каждую строку и признак он делает
// C.CString, а библиотека потом хэширует эти строки заново. На нашей модели это съедало
// ~80% времени: 2.4 us/строку против 0.5 us/строку у самого CatBoost в Python. Здесь
// значения категорий один раз хэшируются функцией самой библиотеки
// (GetStringCatFeatureHash), а батч уходит в CalcModelPredictionWithHashedCatFeatures
// одним вызовом, без аллокаций на строку.
package cbfast

/*
#cgo LDFLAGS: -L${SRCDIR}/../../lib -lcatboostmodel -Wl,-rpath,${SRCDIR}/../../lib
#include <stdlib.h>
#include <stdbool.h>
#include "c_api.h"

static bool calc_hashed(ModelCalcerHandle* h, size_t n,
                        const float* fl, size_t nf, const int* cat, size_t nc, double* res) {
	const float** fp = (const float**)malloc(n * sizeof(float*));
	const int** cp = (const int**)malloc(n * sizeof(int*));
	for (size_t i = 0; i < n; i++) { fp[i] = fl + i * nf; cp[i] = cat + i * nc; }
	bool ok = CalcModelPredictionWithHashedCatFeatures(h, n, fp, nf, cp, nc, res, n);
	free(fp); free(cp);
	return ok;
}
*/
import "C"

import (
	"errors"
	"fmt"
	"strconv"
	"unsafe"

	"stand/internal/features"
)

const nCat = 6

type Model struct {
	h       unsafe.Pointer
	nFloat  int // 1 = base (hour), 7 = cyclic (hour + 6 sin/cos)
	Trees   int
	route   [128]int32
	weekday [8]int32
	bit     [2]int32
	season  map[string]int32
}

func hash(s string) int32 {
	cs := C.CString(s)
	defer C.free(unsafe.Pointer(cs))
	return int32(C.GetStringCatFeatureHash(cs, C.size_t(len(s))))
}

func lastErr() error { return errors.New(C.GoString(C.GetErrorString())) }

func Load(path string) (*Model, error) {
	h := C.ModelCalcerCreate()
	cp := C.CString(path)
	defer C.free(unsafe.Pointer(cp))
	if !C.LoadFullModelFromFile(h, cp) {
		return nil, lastErr()
	}
	nf := int(C.GetFloatFeaturesCount(h))
	if (nf != 1 && nf != 7) || C.GetCatFeaturesCount(h) != nCat {
		return nil, fmt.Errorf("ожидалась base (1 числовой) или cyclic (7 числовых) модель с 6 категориями, а тут %d числовых", nf)
	}
	m := &Model{h: h, nFloat: nf, Trees: int(C.GetTreeCount(h)), season: map[string]int32{}}
	for i := range m.route {
		m.route[i] = hash(strconv.Itoa(i))
	}
	for i := range m.weekday {
		m.weekday[i] = hash(strconv.Itoa(i))
	}
	m.bit[0], m.bit[1] = hash("0"), hash("1")
	for _, s := range []string{"winter", "spring", "summer", "autumn"} {
		m.season[s] = hash(s)
	}
	return m, nil
}

func (m *Model) Name() string {
	if m.nFloat == 7 {
		return "catboost-cyclic"
	}
	return "catboost-base"
}

func (m *Model) NumFloat() int { return m.nFloat }

func b(v bool) int {
	if v {
		return 1
	}
	return 0
}

func (m *Model) catHashes(r features.Row) [nCat]int32 {
	return [nCat]int32{m.route[r.Route], m.bit[b(r.IsHoliday)], m.bit[b(r.IsWeekend)], m.bit[b(r.IsShortWorkingDay)],
		m.season[r.Season], m.weekday[r.Weekday]}
}

func (m *Model) floats(r features.Row) []float32 {
	f := r.Floats()
	return f[:m.nFloat]
}

// Predict - сырой выход модели (без обрезки нуля), в порядке rows.
func (m *Model) Predict(rows []features.Row) ([]float64, error) {
	n := len(rows)
	if n == 0 {
		return nil, nil
	}
	fl := make([]float32, n*m.nFloat)
	cat := make([]int32, n*nCat)
	for i, r := range rows {
		copy(fl[i*m.nFloat:], m.floats(r))
		h := m.catHashes(r)
		copy(cat[i*nCat:], h[:])
	}
	res := make([]float64, n)
	if !C.calc_hashed(m.h, C.size_t(n), (*C.float)(&fl[0]), C.size_t(m.nFloat),
		(*C.int)(unsafe.Pointer(&cat[0])), nCat, (*C.double)(&res[0])) {
		return nil, lastErr()
	}
	return res, nil
}

// Trace - что именно уходит в модель для одной строки: для разбора точки в UI.
type Trace struct {
	FloatNames []string  `json:"float_names"`
	Floats     []float32 `json:"floats"`
	CatNames   []string  `json:"cat_names"`
	CatStrings []string  `json:"cat_strings"`
	CatHashes  []int32   `json:"cat_hashes"`
	Raw        float64   `json:"raw"`
}

func (m *Model) Explain(r features.Row) (Trace, error) {
	p, err := m.Predict([]features.Row{r})
	if err != nil {
		return Trace{}, err
	}
	h := m.catHashes(r)
	names := []string{"hour", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "day_of_month_sin", "day_of_month_cos"}
	return Trace{
		FloatNames: names[:m.nFloat], Floats: m.floats(r),
		CatNames:   []string{"route", "is_holiday", "is_weekend", "is_short_working_day", "season", "weekday"},
		CatStrings: r.CatStrings(), CatHashes: h[:], Raw: p[0],
	}, nil
}
