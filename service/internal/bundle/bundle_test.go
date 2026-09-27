package bundle

import "testing"

func TestCheckRejectsBadPairsAndFactors(t *testing.T) {
	b, err := Load("../../../artifacts/bundle/catboost")
	if err != nil {
		t.Fatal(err)
	}
	recheck := func() error {
		b.Corridor.band = [24]string{}
		return b.check()
	}
	if err := recheck(); err != nil {
		t.Fatal(err)
	}
	daily := b.Corridor.Daily["*"]
	b.Corridor.Daily["*"] = [2]float64{1.1, 1.2}
	if recheck() == nil {
		t.Fatal("daily pair without 1 accepted")
	}
	b.Corridor.Daily["*"] = daily
	hourly := b.Corridor.Hourly["*"]["night"]
	b.Corridor.Hourly["*"]["night"] = [2]float64{0.5, 0.9}
	if recheck() == nil {
		t.Fatal("hourly pair without 1 accepted")
	}
	b.Corridor.Hourly["*"]["night"] = hourly
	for d, row := range b.Postprocess.NewYearTable {
		row[3] = 0
		b.Postprocess.NewYearTable[d] = row
		break
	}
	if recheck() == nil {
		t.Fatal("zero New Year factor accepted")
	}
}
