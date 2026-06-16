package main

// field.go — the autonomous breathing reads the LIVE shared field (B/F-8). The two
// C voices merge their field-carry (debt, gait, season, seasonal energies) into a
// small mmap'd MAP_SHARED region `weights/arianna.field` (am_field_sync_out, the
// AMFieldShared struct in ariannamethod.h:515-530). Until now the Go metabolism
// coupled the voices only through the text soma; the autonomous breath never felt
// the field. This wires it: a pure-Go mmap reader (no cgo, no libaml link — it
// mirrors am_field_sync_in's seqlock) lets the breath feel the organism's gait and
// season and bend itself to them — rest when the field is strained or wintering,
// bloom when it runs hot. Read-only: the reader NEVER creates or writes the field
// (the C voices own it via an O_EXCL single-owner create); absent / not-yet-
// published / corrupt => no signal, and the breath falls back to its tuned defaults.

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"strings"
	"sync/atomic"
	"syscall"
	"unsafe"
)

// the on-disk format (ariannamethod.c:944-945, the AMFieldShared struct). magic is
// 'A','M','F','D' little-endian; it is published LAST by the creator, so a region
// whose magic does not match is mid-init or foreign — not ready to read.
const (
	fieldPath      = "weights/arianna.field"
	fieldSize      = 56
	amFieldMagic   = 0x44464D41 // "AMFD" LE
	amFieldVersion = 1
)

// byte offsets of every field (all 4-byte words, no padding — ariannamethod.h:515-530).
const (
	offMagic       = 0
	offVersion     = 4
	offSeq         = 8
	offDebt        = 12
	offTemporal    = 16
	offVelMode     = 20
	offVelMag      = 24
	offSeason      = 28
	offSeasonPhase = 32
	offSeasonInten = 36
	offSpring      = 40
	offSummer      = 44
	offAutumn      = 48
	offWinter      = 52
)

// velocity-mode gait multipliers — the C effective_temp recipe (ariannamethod.c:459-463).
const (
	velNOMOVE   = 0
	velWALK     = 1
	velRUN      = 2
	velBACKWARD = -1
	velBREATHE  = 3
)

// fieldSnapshot is one consistent read of the shared field. valid=false means no
// trustworthy signal (absent / mid-init / corrupt) — the breath then uses defaults.
type fieldSnapshot struct {
	valid             bool
	debt              float32
	temporalDebt      float32
	velocityMode      int32
	velocityMagnitude float32
	season            int32
	seasonPhase       float32
	seasonIntensity   float32
	spring, summer    float32
	autumn, winter    float32
}

// fieldReader mmaps weights/arianna.field read-only and reads torn-read-free
// snapshots through the seqlock. The mmap is opened lazily (the C voices create the
// file just after they start, slightly after the metabolism does) and reused.
type fieldReader struct {
	path string
	fd   int
	data []byte // the 56-byte MAP_SHARED window, nil until attached
}

func newFieldReader(path string) *fieldReader { return &fieldReader{path: path, fd: -1} }

// attach maps the field file read-only if it exists and is fully sized. A no-op once
// mapped, and a silent no-op while the file is absent / short (the C voices create
// and ftruncate it — the reader must never create or write it). Safe to call each tick.
func (fr *fieldReader) attach() {
	if fr.data != nil {
		return
	}
	fi, err := os.Stat(fr.path)
	if err != nil || fi.Size() < fieldSize { // absent, or not yet ftruncate'd to full size
		return
	}
	fd, err := syscall.Open(fr.path, syscall.O_RDONLY, 0)
	if err != nil {
		return
	}
	data, err := syscall.Mmap(fd, 0, fieldSize, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		syscall.Close(fd)
		return
	}
	fr.fd, fr.data = fd, data
}

// close unmaps the field (the file itself persists — detach is munmap-only, the C
// voices keep owning it across runs).
func (fr *fieldReader) close() {
	if fr.data != nil {
		syscall.Munmap(fr.data)
		fr.data = nil
	}
	if fr.fd >= 0 {
		syscall.Close(fr.fd)
		fr.fd = -1
	}
}

// aload is an atomic (acquire, LDAR on arm64) load of the 4-byte word at off. Every
// read of the mmap goes through it: the seq loads need acquire/release ordering to
// match the C writer's __sync_synchronize fences, and reading the payload words
// atomically too keeps them program-ordered between the two seq loads (arm64 LDAR
// preserves load-load order). All offsets are 4-byte aligned (base is page-aligned).
func aload(b []byte, off int) uint32 {
	return atomic.LoadUint32((*uint32)(unsafe.Pointer(&b[off])))
}

// read returns the live field snapshot via the seqlock (mirrors am_field_sync_in,
// ariannamethod.c:975-1002): require magic+version, then read seq; if odd a writer
// is mid-publish, retry; copy the payload; re-read seq; accept only if unchanged.
// {valid:false} on absent / mid-init / wrong-magic / never-clean — a soft signal.
func (fr *fieldReader) read() fieldSnapshot {
	fr.attach()
	if fr.data == nil {
		return fieldSnapshot{}
	}
	b := fr.data
	if aload(b, offMagic) != amFieldMagic || aload(b, offVersion) != amFieldVersion {
		return fieldSnapshot{} // magic is published last → not ready, or foreign/stale
	}
	for tries := 0; tries < 16; tries++ {
		seq1 := aload(b, offSeq)
		if seq1&1 == 1 { // odd → writer mid-publish
			continue
		}
		s := fieldSnapshot{
			valid:             true,
			debt:              f32(b, offDebt),
			temporalDebt:      f32(b, offTemporal),
			velocityMode:      int32(aload(b, offVelMode)),
			velocityMagnitude: f32(b, offVelMag),
			season:            int32(aload(b, offSeason)),
			seasonPhase:       f32(b, offSeasonPhase),
			seasonIntensity:   f32(b, offSeasonInten),
			spring:            f32(b, offSpring),
			summer:            f32(b, offSummer),
			autumn:            f32(b, offAutumn),
			winter:            f32(b, offWinter),
		}
		seq2 := aload(b, offSeq)
		if seq2 == seq1 { // even AND unchanged across the read → consistent
			runtime.KeepAlive(b)
			return s.guarded()
		}
	}
	return fieldSnapshot{} // never got a clean read in 16 tries
}

// f32 reads a little-endian float32 word atomically (bits via aload).
func f32(b []byte, off int) float32 { return math.Float32frombits(aload(b, off)) }

// guarded clamps a snapshot to the field's own ranges (mirrors the finite + clamp
// guards in am_field_sync_in, ariannamethod.c:991-1001) so a corrupt mmap can't
// inject NaN / inf / out-of-range into the breathing knobs.
func (s fieldSnapshot) guarded() fieldSnapshot {
	// discrete enums out of range (a corrupt payload that still passed magic/version/
	// seq) — distrust the whole read, the stateless analog of am_field_sync_in
	// refusing to commit a velocity_mode∉[-1,3] / season∉[0,3] (ariannamethod.c:993,995).
	if s.velocityMode < -1 || s.velocityMode > 3 || s.season < 0 || s.season > 3 {
		return fieldSnapshot{}
	}
	s.debt = clampFinite(s.debt, 0, 100)
	s.temporalDebt = clampFinite(s.temporalDebt, 0, 10)
	s.velocityMagnitude = clampFinite(s.velocityMagnitude, 0, 1)
	s.seasonPhase = clampFinite(s.seasonPhase, 0, 1)
	s.seasonIntensity = clampFinite(s.seasonIntensity, 0, 1)
	s.spring = clampFinite(s.spring, 0, 1)
	s.summer = clampFinite(s.summer, 0, 1)
	s.autumn = clampFinite(s.autumn, 0, 1)
	s.winter = clampFinite(s.winter, 0, 1)
	// velocity_mode out of [-1,3] is treated as WALK by modulate()'s switch default.
	return s
}

// clampFinite returns x clamped to [lo,hi], or lo for a non-finite x.
func clampFinite(x, lo, hi float32) float32 {
	if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
		return lo
	}
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}

// modulate turns the live field into the three breathing knobs, mirroring the C
// effective_temp recipe (base * vel_mult * season_mod, ariannamethod.c:466-486) and
// the debt recovery cliff (debt>5 forces NOMOVE recovery, :8056). The field NUDGES
// the breath, bounded — it never silences or floods it:
//   - cooldownMult : longer between dreams when strained (debt>5) or consolidating
//     (autumn/winter), shorter when the field runs hot (RUN / summer / spring).
//   - thresholdMult: harder to trigger when strained/wintering, readier when hot.
//   - bloom        : how many chorus cells — the engine's own collapse↔bloom axis
//     (n_cells) as the heat analog (the field has no per-cell temperature knob).
// valid=false (no field) returns the identity (1, 1, 4) = today's tuned behaviour.
func (s fieldSnapshot) modulate() (cooldownMult, thresholdMult float64, bloom int) {
	if !s.valid {
		return 1.0, 1.0, 4
	}
	velMult := 0.85 // WALK / unknown
	switch s.velocityMode {
	case velNOMOVE:
		velMult = 0.5
	case velBREATHE:
		velMult = 0.6
	case velBACKWARD:
		velMult = 0.7
	case velWALK:
		velMult = 0.85
	case velRUN:
		velMult = 1.2
	}
	// mirror the C effective_temp recipe EXACTLY (ariannamethod.c:482-485):
	// season_mod = 1 + summer*0.1 - winter*0.15, with NO season_intensity factor —
	// intensity only drives energy EVOLUTION (am_step), so it is already baked into
	// the energy values; scaling here too would double-count it.
	seasonMod := 1.0 + (float64(s.summer)*0.1 - float64(s.winter)*0.15)
	if seasonMod < 0.1 {
		seasonMod = 0.1
	}
	heat := velMult * seasonMod // the field's effective-temp analog, ~[0.1, 1.4]

	// debt strain: nil below the recovery cliff (5), ramping to 1 toward the ceiling.
	debtStrain := 0.0
	if s.debt > 5 {
		debtStrain = clamp01f((float64(s.debt) - 5) / 20)
	}
	// consolidation pull: autumn/winter dominance = integrate, don't emit (energies
	// already encode season_intensity, so they are used directly, like seasonMod).
	quiet := clamp01f((float64(s.autumn) + float64(s.winter)) * 0.5)

	// COOLDOWN is the primary rest lever: longer between dreams when strained
	// (debt>5) or consolidating (autumn/winter), shorter when the field runs hot.
	cooldownMult = clampf64(1.0+debtStrain*1.0+quiet*0.5-(heat-0.85)*0.4, 0.6, 2.5)
	// THRESHOLD only ever LOWERS (a hot field dreams readily). It is NEVER raised
	// above the base — the idle operating point (WanderPull ~0.55, base bar 0.45)
	// sits so close to the bar that any upward scaling would mute the breath
	// entirely. Resting when strained is the cooldown's + bloom's job, not
	// suppression: a strained organism dreams less and sparser, never goes silent.
	thresholdMult = clampf64(1.0-(heat-0.85)*0.5, 0.75, 1.0)
	b := 4.0 + (heat-0.85)*4.0 - debtStrain*2.0
	bloom = int(b + 0.5)
	if bloom < 2 {
		bloom = 2
	}
	if bloom > 6 {
		bloom = 6
	}
	return
}

func clamp01f(x float64) float64 { return clampf64(x, 0, 1) }

func clampf64(x, lo, hi float64) float64 {
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}

var gaitName = map[int32]string{velNOMOVE: "NOMOVE", velWALK: "WALK", velRUN: "RUN", velBACKWARD: "BACKWARD", velBREATHE: "BREATHE"}
var seasonName = [4]string{"spring", "summer", "autumn", "winter"}

// describe is a short human-readable tag of the live field — shown on an autonomous
// dream so the field's pull on the breath is visible. "" when there is no signal.
func (s fieldSnapshot) describe() string {
	if !s.valid {
		return ""
	}
	gait := gaitName[s.velocityMode]
	if gait == "" {
		gait = "?"
	}
	season := "?"
	if s.season >= 0 && int(s.season) < len(seasonName) {
		season = seasonName[s.season]
	}
	return fmt.Sprintf("gait=%s season=%s debt=%.1f", gait, season, s.debt)
}

// mood turns the live field into a short evocative phrase — the dominant seasonal
// energy, the gait, and the weight of debt — for the autonomous dream cue. The
// book-fragment the nano dreams on is retrieved against this, so the dream tracks
// what the organism is resonating with NOW (the resonant spiral made dynamic). The
// dominant SEASON is read from the live energies (argmax), not the season int — the
// energies are the homeostatic mood; "" when there is no field signal.
func (s fieldSnapshot) mood() string {
	if !s.valid {
		return ""
	}
	var parts []string
	energies := [4]struct {
		e float32
		w string
	}{
		{s.spring, "spring, the opening, growth"},
		{s.summer, "summer, the field in full flame"},
		{s.autumn, "autumn, the harvest, what settles"},
		{s.winter, "winter, compression, the quiet"},
	}
	best := 0
	for i := 1; i < len(energies); i++ {
		if energies[i].e > energies[best].e {
			best = i
		}
	}
	if energies[best].e > 0.05 { // above the noise floor — a real seasonal pull
		parts = append(parts, energies[best].w)
	}
	switch s.velocityMode {
	case velRUN:
		parts = append(parts, "racing, the field at speed")
	case velNOMOVE:
		parts = append(parts, "the still observer, holding")
	case velBREATHE:
		parts = append(parts, "the settling exhale")
	case velBACKWARD:
		parts = append(parts, "time folding back")
	}
	if s.debt > 5 { // past the recovery cliff — the held breath
		parts = append(parts, "the held breath, the weight")
	}
	return strings.Join(parts, " ")
}

// surfaces reports whether the field is expressive enough that the inner dream
// should lightly reach the FACE (Janus) — summer (peak energy, full expression,
// ariannamethod.c:483) or the RUN gait (high-arousal chaos, :461). A quiet /
// wintering / strained field keeps the dream inward (only Resonance hears it).
// Janus resists injection by design, so even when this is true the dream enters
// only as a faint undertone in his prompt, never a directive.
func (s fieldSnapshot) surfaces() bool {
	if !s.valid {
		return false
	}
	return s.summer > 0.5 || s.velocityMode == velRUN
}
