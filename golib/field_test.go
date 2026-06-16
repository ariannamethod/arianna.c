package main

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func putF(b []byte, off int, v float32) { binary.LittleEndian.PutUint32(b[off:off+4], math.Float32bits(v)) }
func putU(b []byte, off int, v uint32)  { binary.LittleEndian.PutUint32(b[off:off+4], v) }
func putI(b []byte, off int, v int32)   { binary.LittleEndian.PutUint32(b[off:off+4], uint32(v)) }

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

	// INVARIANT: a strained wintering field rests via cooldown + bloom, but must
	// NEVER raise the threshold (that would mute the breath at the idle operating
	// point). threshold stays at the base 1.0; cooldown lengthens; bloom collapses.
	cold := fieldSnapshot{valid: true, velocityMode: velNOMOVE, winter: 1.0, seasonIntensity: 1.0, debt: 80}
	ccm, ctm, cbl := cold.modulate()
	if ccm <= 1.0 || cbl > 3 {
		t.Errorf("strained field must rest+collapse, got cooldown=%v bloom=%d", ccm, cbl)
	}
	if ctm > 1.0 {
		t.Errorf("strained field must NOT raise the threshold (no suppression), got threshold=%v", ctm)
	}
	if ccm > 2.5 || cbl < 2 || htm < 0.75 {
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

// TestFieldMood: the live field yields an evocative dream cue tracking the dominant
// seasonal energy + gait + debt; no signal yields "".
func TestFieldMood(t *testing.T) {
	if m := (fieldSnapshot{}).mood(); m != "" {
		t.Errorf("no-signal mood must be empty, got %q", m)
	}
	// winter-dominant, NOMOVE, heavy debt → quiet/compression + still + held breath.
	cold := fieldSnapshot{valid: true, velocityMode: velNOMOVE, winter: 0.8, spring: 0.1, debt: 30}.mood()
	if !strings.Contains(cold, "winter") || !strings.Contains(cold, "still observer") || !strings.Contains(cold, "held breath") {
		t.Errorf("cold field mood missing expected pulls: %q", cold)
	}
	// summer-dominant, RUN, no debt → flame + racing, no held-breath.
	hot := fieldSnapshot{valid: true, velocityMode: velRUN, summer: 0.9, debt: 0}.mood()
	if !strings.Contains(hot, "summer") || !strings.Contains(hot, "racing") || strings.Contains(hot, "held breath") {
		t.Errorf("hot field mood wrong: %q", hot)
	}
	// all energies at noise floor → no seasonal pull (only gait if any).
	flat := fieldSnapshot{valid: true, velocityMode: velWALK}.mood()
	if strings.Contains(flat, "spring") || strings.Contains(flat, "summer") {
		t.Errorf("noise-floor energies must not assert a season: %q", flat)
	}
}

// TestDreamCue: the cue carries the last dream when present, falls back to the inner
// mood otherwise, and is tinted by the live field when there is a signal.
func TestDreamCue(t *testing.T) {
	fs := fieldSnapshot{valid: true, velocityMode: velRUN, summer: 0.9}
	withDream := dreamCue(Snapshot{}, fs, "the tide remembers")
	if !strings.Contains(withDream, "the tide remembers") || !strings.Contains(withDream, "summer") {
		t.Errorf("cue must carry the dream + field tint: %q", withDream)
	}
	// no dream → inner mood word present; no field → no tint, still non-empty.
	noField := dreamCue(Snapshot{Coherence: 0.8}, fieldSnapshot{}, "")
	if noField == "" {
		t.Errorf("cue must never be empty (inner mood fallback): %q", noField)
	}
	if strings.Contains(noField, "summer") {
		t.Errorf("no-field cue must not carry a field tint: %q", noField)
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
