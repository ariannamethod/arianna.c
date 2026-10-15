package main

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func putF(b []byte, off int, v float32) {
	binary.LittleEndian.PutUint32(b[off:off+4], math.Float32bits(v))
}
func putU(b []byte, off int, v uint32) { binary.LittleEndian.PutUint32(b[off:off+4], v) }
func putI(b []byte, off int, v int32)  { binary.LittleEndian.PutUint32(b[off:off+4], uint32(v)) }

// buildField lays out a 56-byte AMFieldShared image exactly as the C writer would,
// so the Go reader is tested against the real on-disk format (offsets ariannamethod.h:515-530).
func buildField(magic, version, seq uint32, fs fieldSnapshot) []byte {
	b := make([]byte, fieldSize)
	putU(b, offMagic, magic)
	putU(b, offVersion, version)
	putU(b, offSeq, seq)
	putF(b, offDebt, fs.debt)
	putF(b, offTemporal, fs.temporalDebt)
	putI(b, offVelMode, fs.velocityMode)
	putF(b, offVelMag, fs.velocityMagnitude)
	putI(b, offSeason, fs.season)
	putF(b, offSeasonPhase, fs.seasonPhase)
	putF(b, offSeasonInten, fs.seasonIntensity)
	putF(b, offSpring, fs.spring)
	putF(b, offSummer, fs.summer)
	putF(b, offAutumn, fs.autumn)
	putF(b, offWinter, fs.winter)
	return b
}

func writeTempField(t *testing.T, b []byte) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "arianna.field")
	if err := os.WriteFile(p, b, 0644); err != nil {
		t.Fatal(err)
	}
	return p
}

// TestFieldReadRoundTrip mmaps a real on-disk field image and reads it back through
// the seqlock — the same path field_probe.c exercises cross-process.
func TestFieldReadRoundTrip(t *testing.T) {
	want := fieldSnapshot{velocityMode: velRUN, velocityMagnitude: 0.7, season: 1, seasonPhase: 0.4,
		seasonIntensity: 0.8, debt: 3.5, temporalDebt: 1.2, spring: 0.1, summer: 0.9}
	fr := newFieldReader(writeTempField(t, buildField(amFieldMagic, amFieldVersion, 2, want)))
	defer fr.close()
	got := fr.read()
	if !got.valid {
		t.Fatal("expected a valid read of a well-formed field")
	}
	if got.velocityMode != velRUN || got.season != 1 {
		t.Errorf("gait/season parsed wrong: mode=%d season=%d", got.velocityMode, got.season)
	}
	if math.Abs(float64(got.debt-3.5)) > 1e-6 || math.Abs(float64(got.summer-0.9)) > 1e-6 ||
		math.Abs(float64(got.temporalDebt-1.2)) > 1e-6 || math.Abs(float64(got.seasonIntensity-0.8)) > 1e-6 {
		t.Errorf("payload parsed wrong: %+v", got)
	}
}

// TestFieldRejects covers every degrade case: a reader must never trust a field
// that is wrong-magic / wrong-version / mid-publish (odd seq) / short / absent.
func TestFieldRejects(t *testing.T) {
	good := fieldSnapshot{velocityMode: velWALK, seasonIntensity: 0.5}
	cases := []struct {
		name string
		path string
	}{
		{"wrong-magic", writeTempField(t, buildField(0xDEADBEEF, amFieldVersion, 2, good))},
		{"wrong-version", writeTempField(t, buildField(amFieldMagic, 99, 2, good))},
		{"odd-seq", writeTempField(t, buildField(amFieldMagic, amFieldVersion, 3, good))},
	}
	for _, c := range cases {
		fr := newFieldReader(c.path)
		if fr.read().valid {
			t.Errorf("%s: should degrade to no-signal", c.name)
		}
		fr.close()
	}
	// short file (creator hasn't ftruncate'd to full size) → never mmap, degrade.
	short := filepath.Join(t.TempDir(), "arianna.field")
	if err := os.WriteFile(short, make([]byte, 10), 0644); err != nil {
		t.Fatal(err)
	}
	if newFieldReader(short).read().valid {
		t.Error("short file: should degrade to no-signal")
	}
	// absent file → degrade, and the reader must NOT create it.
	absent := filepath.Join(t.TempDir(), "nope.field")
	if newFieldReader(absent).read().valid {
		t.Error("absent file: should degrade to no-signal")
	}
	if _, err := os.Stat(absent); !os.IsNotExist(err) {
		t.Error("reader must never create the field file")
	}
}

// TestFieldModulate checks the field → {cooldown, threshold, bloom} mapping: no
// signal is identity (today's behaviour), a hot field quickens + blooms, a strained
// wintering field rests + collapses, and the knobs stay inside their clamps.
func TestFieldModulate(t *testing.T) {
	cm, tm, bl := (fieldSnapshot{}).modulate()
	if cm != 1.0 || tm != 1.0 || bl != 4 {
		t.Errorf("no-signal must be identity (1,1,4), got (%v,%v,%d)", cm, tm, bl)
	}

	hot := fieldSnapshot{valid: true, velocityMode: velRUN, summer: 1.0, seasonIntensity: 1.0}
	hcm, htm, hbl := hot.modulate()
	if hcm >= 1.0 || htm >= 1.0 || hbl < 5 {
		t.Errorf("hot field (RUN/summer) must quicken+lower-threshold+bloom, got cooldown=%v threshold=%v bloom=%d", hcm, htm, hbl)
	}

	// A strained wintering field rests by lengthening cooldown, raising the
	// threshold inside the clamp, and collapsing bloom.
	cold := fieldSnapshot{valid: true, velocityMode: velNOMOVE, winter: 1.0, seasonIntensity: 1.0, debt: 80}
	ccm, ctm, cbl := cold.modulate()
	if ccm <= 1.0 || cbl > 3 {
		t.Errorf("strained field must rest+collapse, got cooldown=%v bloom=%d", ccm, cbl)
	}
	if ctm <= 1.0 {
		t.Errorf("strained field must raise the threshold inside the clamp, got threshold=%v", ctm)
	}
	if ccm > 10.0 || ctm > 2.75 || cbl < 1 || htm < 0.75 {
		t.Errorf("knobs escaped their clamps: cooldown=%v bloom=%d hot-threshold=%v", ccm, cbl, htm)
	}
}

// TestFieldGuards: a corrupt mmap (NaN/inf/out-of-range) must not poison the knobs.
func TestFieldGuards(t *testing.T) {
	g := fieldSnapshot{valid: true, debt: float32(math.Inf(1)), summer: float32(math.NaN()), winter: 5.0}.guarded()
	if g.debt != 0 {
		t.Errorf("inf debt not guarded: %v", g.debt)
	}
	if g.summer != 0 {
		t.Errorf("NaN summer not guarded: %v", g.summer)
	}
	if g.winter != 1.0 {
		t.Errorf("out-of-range winter not clamped to 1: %v", g.winter)
	}
}

// TestFieldSurfaces: the inner dream reaches the face only when the field is
// expressive (summer / RUN); quiet / wintering / strained / no-signal keeps it inward.
func TestFieldSurfaces(t *testing.T) {
	if (fieldSnapshot{}).surfaces() {
		t.Error("no signal must not surface the dream to the face")
	}
	if !(fieldSnapshot{valid: true, summer: 0.8}).surfaces() {
		t.Error("summer-dominant field should surface")
	}
	if !(fieldSnapshot{valid: true, velocityMode: velRUN}).surfaces() {
		t.Error("RUN gait should surface")
	}
	if (fieldSnapshot{valid: true, velocityMode: velNOMOVE, winter: 0.9, debt: 30}).surfaces() {
		t.Error("strained wintering field must keep the dream inward")
	}
}

// TestDreamCue verifies that autonomous generation is seeded only by Arianna's
// carried output or numerical body/field state, never a handwritten sentence.
func TestDreamCue(t *testing.T) {
	carried := dreamCue(Snapshot{}, fieldSnapshot{valid: true, velocityMode: velRUN, summer: 0.9}, "the tide remembers", "")
	if carried != "the tide remembers" {
		t.Fatalf("carried dream changed: %q", carried)
	}
	fromField := dreamCue(Snapshot{}, fieldSnapshot{valid: true, velocityMode: velRUN, summer: 0.9}, "", "")
	if fromField != "0.000 0.000 0.000 0.000 0.000 0.900 2 0" {
		t.Fatalf("field state cue = %q", fromField)
	}
	fromBody := dreamCue(Snapshot{Arousal: 0.2, Valence: -0.1, Coherence: 0.8, WanderPull: 0.4}, fieldSnapshot{}, "", "")
	if fromBody != "0.200 -0.100 0.800 0.400" {
		t.Fatalf("body state cue = %q", fromBody)
	}
}

// TestFieldGuardsDiscreteRange: an out-of-range velocity_mode / season (a corrupt
// payload that still passed magic/seq) must distrust the whole read (like the C
// reader refusing to commit it), not slip through as a WALK / "?" gait.
func TestFieldGuardsDiscreteRange(t *testing.T) {
	if (fieldSnapshot{valid: true, velocityMode: 99}).guarded().valid {
		t.Error("out-of-range velocity_mode must invalidate the snapshot")
	}
	if (fieldSnapshot{valid: true, velocityMode: -2}).guarded().valid {
		t.Error("velocity_mode < -1 must invalidate the snapshot")
	}
	if (fieldSnapshot{valid: true, season: 7}).guarded().valid {
		t.Error("out-of-range season must invalidate the snapshot")
	}
}

// TestFieldSeasonIntensityIndependent: the heat (effective_temp analog) must mirror
// the C recipe, which does NOT scale by season_intensity (the energies already bake
// it). Two snapshots differing only in season_intensity must modulate identically.
func TestFieldSeasonIntensityIndependent(t *testing.T) {
	lo := fieldSnapshot{valid: true, velocityMode: velRUN, summer: 1.0, seasonIntensity: 0.2}
	hi := fieldSnapshot{valid: true, velocityMode: velRUN, summer: 1.0, seasonIntensity: 1.0}
	lcm, ltm, lbl := lo.modulate()
	hcm, htm, hbl := hi.modulate()
	if lcm != hcm || ltm != htm || lbl != hbl {
		t.Errorf("season_intensity must not change the heat mapping: lo=(%v,%v,%d) hi=(%v,%v,%d)", lcm, ltm, lbl, hcm, htm, hbl)
	}
}
