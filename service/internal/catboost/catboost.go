package catboost

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
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"unsafe"

	"stand/internal/features"
)

type Model struct {
	h      unsafe.Pointer
	nFloat int
	nCat   int
	Trees  int
	Spec   *features.Spec
	hashes map[string]int32
}

func Load(dir string) (*Model, error) {
	b, err := os.ReadFile(filepath.Join(dir, "features.json"))
	if err != nil {
		return nil, err
	}
	var f struct {
		Features    []string `json:"features"`
		CatFeatures []string `json:"cat_features"`
	}
	if err := json.Unmarshal(b, &f); err != nil {
		return nil, fmt.Errorf("features.json: %w", err)
	}
	spec, err := features.NewSpec(f.Features, f.CatFeatures)
	if err != nil {
		return nil, err
	}
	h := C.ModelCalcerCreate()
	cp := C.CString(filepath.Join(dir, "model.cbm"))
	defer C.free(unsafe.Pointer(cp))
	if !C.LoadFullModelFromFile(h, cp) {
		return nil, lastErr()
	}
	m := &Model{
		h: h, nFloat: int(C.GetFloatFeaturesCount(h)), nCat: int(C.GetCatFeaturesCount(h)),
		Trees: int(C.GetTreeCount(h)), Spec: spec, hashes: map[string]int32{},
	}
	if spec.NumFloat() != m.nFloat || spec.NumCat() != m.nCat {
		return nil, fmt.Errorf("features.json (%d числовых, %d категориальных) не совпадает с моделью (%d, %d)",
			spec.NumFloat(), spec.NumCat(), m.nFloat, m.nCat)
	}
	for i := range 100 {
		m.hashes[strconv.Itoa(i)] = hash(strconv.Itoa(i))
	}
	for _, s := range []string{"winter", "spring", "summer", "autumn"} {
		m.hashes[s] = hash(s)
	}
	return m, nil
}

func (m *Model) Hash(s string) int32 {
	if v, ok := m.hashes[s]; ok {
		return v
	}
	return hash(s)
}

func (m *Model) Predict(cells []features.Cell, usual bool) ([]float64, error) {
	rows := make([]features.Row, 0, len(cells)*24)
	for _, c := range cells {
		d := c.Day
		if usual {
			d = d.Ordinary()
		}
		for h := range 24 {
			rows = append(rows, features.Row{Route: c.Route, Hour: h, Day: d})
		}
	}
	return m.predictRows(rows)
}

func (m *Model) Ordinary(c features.Cell) bool { return c.Day == c.Day.Ordinary() }

func (m *Model) Explain(c features.Cell, hour int) ([]features.Feature, float64, error) {
	r := features.Row{Route: c.Route, Hour: hour, Day: c.Day}
	raw, err := m.predictRows([]features.Row{r})
	if err != nil {
		return nil, 0, err
	}
	out := m.Spec.Values(r)
	for i, f := range out {
		if f.Kind == "categorical" {
			h := m.Hash(f.Value.(string))
			out[i].Hash = &h
		}
	}
	return out, raw[0], nil
}

func (m *Model) Describe() string { return fmt.Sprintf("CatBoost, %d деревьев", m.Trees) }

func (m *Model) predictRows(rows []features.Row) ([]float64, error) {
	n, nf, nc := len(rows), m.nFloat, m.nCat
	if n == 0 {
		return nil, nil
	}
	fl, cat, strs := make([]float32, n*nf), make([]int32, n*nc), make([]string, nc)
	for i, r := range rows {
		m.Spec.Encode(r, fl[i*nf:(i+1)*nf], strs)
		for j, s := range strs {
			cat[i*nc+j] = m.Hash(s)
		}
	}
	res := make([]float64, n)
	if !C.calc_hashed(m.h, C.size_t(n), (*C.float)(&fl[0]), C.size_t(nf),
		(*C.int)(unsafe.Pointer(&cat[0])), C.size_t(nc), (*C.double)(&res[0])) {
		return nil, lastErr()
	}
	return res, nil
}

func hash(s string) int32 {
	cs := C.CString(s)
	defer C.free(unsafe.Pointer(cs))
	return int32(C.GetStringCatFeatureHash(cs, C.size_t(len(s))))
}

func lastErr() error { return errors.New(C.GoString(C.GetErrorString())) }
