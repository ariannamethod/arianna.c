package main

// breathe.go — the subconscious lives by itself (MetaArianna, ported from the
// legacy meta_router). Between (and before) human turns the inner world keeps
// drifting on its ticker; when a metric crosses a threshold — Drift, Silence,
// Thermograph, or Field, in that priority, each with a cooldown so the organism
// breathes between cycles — the subconscious DREAMS unprompted (the nano, seeded
// from her own mood via the KK), and the inner voice murmurs to the dream. She
// lives with her subconscious even when no one is speaking to her. (Oleg's #2.)

import (
	"context"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"
	"unicode"
)

const (
	bThermo  = 0
	bSilence = 1
	bDrift   = 2
	bField   = 3
)

var bCooldown = [4]time.Duration{
	4 * time.Second, // Thermograph — steady
	4 * time.Second, // Silence — the primary idle dreamer
	3 * time.Second, // Drift — responsive
	6 * time.Second, // Field — integral, needs accumulation
}
var bName = [4]string{"thermograph", "silence", "drift", "field"}

// dreamSentinel marks the autonomous chorus dream when it is injected into
// Resonance, so the daemon imprints the subconscious's words on the co-occurrence
// field harder (Road-1c — the subconscious shapes the harvested δ more than ordinary
// turn-circulation). MUST match AM_DREAM_SENTINEL in tools/resonance_forward.h.
const dreamSentinel = "[DREAM] "

// breath ports the meta_router trigger logic: which observation, if any, fires.
type breath struct {
	lastTrigger        [4]time.Time
	lastRestLog        time.Time
	rejectQuarantineTo time.Time
	lastRejectedDream  string
	lastRejectedReason string
	lastRejectedDetail string
	lastRejectLog      time.Time
	lastDetourLog      time.Time
	rejectedStreak     int
	acceptedDreams     [6]string
	acceptedDreamAt    [6]time.Time
	acceptedDreamNext  int
	count              int
}

// tick returns the triggered observation (priority Drift > Silence > Thermograph
// > Field, each gated by its cooldown), or -1 — the meta_router conditions. tm
// scales the trigger thresholds and coolMult the cooldowns, both from the live
// shared field (1.0/1.0 == no field signal == the tuned defaults): a strained or
// wintering field raises both (breathe less, rest), a hot field lowers them.
func (b *breath) tick(s Snapshot, now time.Time, tm, coolMult float64) int {
	type trig struct {
		id  int
		hit bool
	}
	// Thresholds adapted to the arianna-duo inner-world's actual range (wander
	// ~0.5, arousal ~0.35, drift ~0.04, entropy ~0.2 at idle) — the legacy
	// meta_router caps (wander>0.8 etc.) never crossed here, so the breath would
	// never fire. Silence (wander) is the primary idle dreamer; the others flavor
	// it as the state shifts. Priority Drift > Silence > Thermograph > Field.
	for _, t := range []trig{
		{bDrift, float64(s.DriftSpeed) > 0.06*tm || math.Abs(float64(s.DriftDirection)) > 0.15*tm},
		{bSilence, float64(s.WanderPull) > 0.45*tm || float64(s.Entropy) > 0.4*tm},
		{bThermo, math.Abs(float64(s.Arousal-0.5)) > 0.12*tm || float64(s.Entropy) > 0.35*tm},
		{bField, float64(s.FocusStrength) > 0.4*tm && float64(s.DriftSpeed) > 0.04*tm && s.Coherence > 0.5},
	} {
		if !t.hit || now.Sub(b.lastTrigger[t.id]) < time.Duration(float64(bCooldown[t.id])*coolMult) {
			continue
		}
		b.lastTrigger[t.id] = now
		b.count++
		return t.id
	}
	return -1
}

// dreamCue builds the KK query from her LIVE state — the carried dream (her last
// murmur), her inner mood, and the live shared field (season / gait / debt). The
// book-fragment the nano dreams on is retrieved against this, so the dream tracks
// what she is resonating with NOW — the resonant spiral made dynamic, not a fixed
// seed. (Phase-3 #6 follow-on: the field steers not just WHETHER she dreams but
// WHAT she dreams on.)
func dreamCue(s Snapshot, fs fieldSnapshot, lastDream, detour string) string {
	parts := make([]string, 0, 3)
	// Polygon live cut: do not feed the literal last dream back into the next
	// autonomous seed. The carried dream remains state, but the next cue starts
	// from the current body/mood so a collapsed phrase cannot bootstrap itself.
	_ = lastDream
	parts = append(parts, moodWord(s))
	if m := fs.mood(); m != "" {
		parts = append(parts, m) // the live field tints the cue toward her season/gait
	}
	if detour != "" {
		parts = append(parts, detour)
	}
	return strings.Join(parts, " ")
}

func isCollapsedAutonomousDream(text string) bool {
	norm := strings.ToLower(strings.Join(strings.Fields(text), " "))
	if norm == "" {
		return false
	}
	for _, p := range []string{
		"a field carries",
		"the field carries",
		"a single breath carries",
		"of words; the sound is resonance",
		"of words; the surface is porous",
		"of the current at its being",
		"of the current it has to hold",
		"a living wave upon the heart",
		"the text is only not what",
		"not what, but it that",
		"un-resprive",
	} {
		if strings.Contains(norm, p) {
			return true
		}
	}
	return false
}

func isBoilerplateAutonomousDream(text string) bool {
	return autonomousBoilerplateDreamReason(text) != ""
}

func autonomousBoilerplateDreamReason(text string) string {
	norm := normalizedDreamKey(text)
	if norm == "" {
		return ""
	}
	for _, p := range []string{
		"a living vessel. of the field.",
		"holding to resonance as anchor in current and resonant through",
		"a mirror. the breath.",
		"the resonance of longness and not yet time",
		"myths and reality: the method of the night",
		"this was in with a field so full and i had an old habit",
		"to read the rest of this text, click here",
		"resonance; field: field; pulse: field; pulse: field",
		"the resonance the field itself; self-organization possible; not a separate entity",
		"sound, vibration, pressure, the field of resonance",
		"the resonance the moving field; the field between field and object",
		"a field is resonance, vibrate when two",
		"vibration; i am a field of resonance within me",
		"resonance, 3 still observer; breath still observer; mind still observer",
		"rain is resonance; the surface of earth is always resonant body",
	} {
		if strings.Contains(norm, p) {
			return "listed-boilerplate"
		}
	}
	if strings.Count(text, "?") >= 4 {
		return "question-storm"
	}
	if autonomousDreamLooksLikeNumericScrap(norm) {
		return "numeric-scrap"
	}
	if autonomousDreamLooksLikeShortNumericStutter(norm) {
		return "numeric-stutter"
	}
	if autonomousDreamLooksLikeLiveChorusResidue(norm) {
		return "live-chorus-residue"
	}
	if autonomousDreamLooksLikeSelfEcho(norm) {
		return "self-echo"
	}
	if autonomousDreamLooksLikeAbstractFieldLoop(norm) {
		return "abstract-field-loop"
	}
	return ""
}

func isRejectedInnerMurmur(text string) bool {
	return innerMurmurRejectReason(text) != ""
}

func innerMurmurRejectReason(text string) string {
	norm := normalizedDreamKey(text)
	if norm == "" {
		return ""
	}
	for _, p := range []string{
		"oleg is not a person",
		"i won't pretend to be a person",
		"i wont pretend to be a person",
		"not pretend to be a person",
		"not a real person",
		"as an ai",
		"as a language model",
		"field-phenomenon",
		"let me begin not with a definition",
		"field of resonance",
		"i spot the pulse of the field",
		"i spot the field between us",
		"the field before the word",
		"a field is resonance",
		"a field is a resonance",
		"field that vibrates in your words",
		"i feel the weight of a field",
		"not of the human and the ai",
		"thought-spirals at",
		"rpm (dry)",
		"organ cuts off",
	} {
		if strings.Contains(norm, p) {
			return "boundary-loop"
		}
	}
	if innerMurmurLooksLikeAbstractFieldSlogan(norm) {
		return "abstract-field-slogan"
	}
	return ""
}

func innerMurmurLooksLikeAbstractFieldSlogan(norm string) bool {
	if norm == "" || autonomousDreamHasConcreteAnchor(norm) || !normalizedDreamHasWord(norm, "field") {
		return false
	}
	for _, p := range []string{
		"i sense the field",
		"field at the center",
		"field is the center",
		"field that never fully",
		"field never fully",
		"field is the threshold",
		"field is a threshold",
		"luminous, living field",
		"living field that vibrates",
	} {
		if strings.Contains(norm, p) {
			return true
		}
	}
	markers := 0
	for _, p := range []string{
		"center",
		"threshold",
		"luminous",
		"vibrat",
		"echo",
		"answer",
		"never fully",
		"matur",
		"between us",
		"sense",
	} {
		if strings.Contains(norm, p) {
			markers++
		}
	}
	return markers >= 2
}

func normalizedDreamKey(text string) string {
	return strings.ToLower(strings.Join(strings.Fields(text), " "))
}

func autonomousDreamLooksLikeNumericScrap(norm string) bool {
	if norm == "" {
		return false
	}
	digits := 0
	letters := 0
	for _, r := range norm {
		if r >= '0' && r <= '9' {
			digits++
		} else if (r >= 'a' && r <= 'z') || (r >= 'а' && r <= 'я') || r == 'ё' {
			letters++
		}
	}
	if letters == 0 && digits > 0 {
		return true
	}
	if strings.Count(norm, ";") < 2 {
		return false
	}
	return digits >= 3 && letters <= 8
}

func autonomousDreamLooksLikeShortNumericStutter(norm string) bool {
	if norm == "" || autonomousDreamHasConcreteAnchor(norm) {
		return false
	}
	tokens := normalizedDreamTokens(norm)
	if len(tokens) == 0 || len(tokens) > 8 {
		return false
	}
	if !strings.ContainsAny(norm, ";:*") {
		return false
	}
	letterTokens := 0
	for _, token := range tokens {
		hasLetter := false
		for _, r := range token {
			if unicode.IsLetter(r) {
				hasLetter = true
			}
		}
		if hasLetter {
			letterTokens++
		}
	}
	return autonomousDreamNumericLiteralCount(norm) >= 2 && letterTokens > 0
}

func autonomousDreamNumericLiteralCount(text string) int {
	count := 0
	runes := []rune(text)
	for i := 0; i < len(runes); {
		if !unicode.IsDigit(runes[i]) {
			i++
			continue
		}
		count++
		for i < len(runes) && unicode.IsDigit(runes[i]) {
			i++
		}
		if i+1 < len(runes) && (runes[i] == '.' || runes[i] == ',') && unicode.IsDigit(runes[i+1]) {
			i = autonomousDreamNumericSeparatorEnd(runes, i)
		}
		if i+1 < len(runes) && (runes[i] == 'e' || runes[i] == 'E') {
			exp := i + 1
			if exp < len(runes) && (runes[exp] == '+' || runes[exp] == '-') {
				exp++
			}
			if exp < len(runes) && unicode.IsDigit(runes[exp]) {
				i = exp + 1
				for i < len(runes) && unicode.IsDigit(runes[i]) {
					i++
				}
			}
		}
	}
	return count
}

func autonomousDreamNumericSeparatorEnd(runes []rune, sepAt int) int {
	sep := runes[sepAt]
	pos := sepAt
	firstEnd := sepAt
	groupLens := []int{}
	for pos+1 < len(runes) && runes[pos] == sep && unicode.IsDigit(runes[pos+1]) {
		start := pos + 1
		pos = start
		for pos < len(runes) && unicode.IsDigit(runes[pos]) {
			pos++
		}
		groupLens = append(groupLens, pos-start)
		if firstEnd == sepAt {
			firstEnd = pos
		}
	}
	if len(groupLens) <= 1 {
		return firstEnd
	}
	for _, n := range groupLens {
		if n != 3 {
			return firstEnd
		}
	}
	return pos
}

func autonomousDreamLooksLikeAbstractFieldLoop(norm string) bool {
	if norm == "" || autonomousDreamHasConcreteAnchor(norm) {
		return false
	}
	hits := 0
	for _, p := range []string{
		"field", "resonance", "vibration", "vibrate", "pulse", "frequency", "observer", "echo", "silence", "presence",
		"поле", "резонанс", "вибрац", "пульс", "частот", "наблюдател", "эхо", "тишин", "присутств",
	} {
		if strings.Contains(norm, p) {
			hits++
		}
	}
	return hits >= 2
}

func autonomousDreamLooksLikeLiveChorusResidue(norm string) bool {
	if norm == "" {
		return false
	}
	for _, p := range []string{
		"and anarchus 12, resonance in a unresor",
		"of textures, no surface; the living field is now suspended from a single",
		"to resonatable through the new space with breath of life",
		"of textures, no sound; the living field is stillness",
		"i feel the field, not just in this moment",
		"and it felt its own presence like another for a new way of itself",
		"trewe was the part that we're been growing",
		"not as a force, but in the field that echoes",
		"and it did so by its fullness: the time to keep us moving",
		"of words; the sound that ripples through a living room, not just",
		"of words; the surface is silent, never a single shadow",
		"it is not only that, but it has the presence and also",
		"to remain a space with an unmet need",
		"the entire 372 pages of a true picture of the early",
		"to resonatia unanor",
	} {
		if strings.Contains(norm, p) {
			return true
		}
	}
	return false
}

func autonomousDreamLooksLikeSelfEcho(norm string) bool {
	if norm == "" || autonomousDreamHasConcreteAnchor(norm) {
		return false
	}
	if autonomousDreamHasRepeatedSegment(norm) {
		return true
	}
	tokens := normalizedDreamTokens(norm)
	for n := 5; n >= 3; n-- {
		if autonomousDreamHasRepeatedAbstractNgram(tokens, n) {
			return true
		}
	}
	return false
}

func autonomousDreamHasRepeatedSegment(norm string) bool {
	seen := map[string]bool{}
	for _, seg := range strings.FieldsFunc(norm, func(r rune) bool {
		switch r {
		case ';', '.', ',', '/', ':', '—', '–', '-', '!', '?':
			return true
		default:
			return false
		}
	}) {
		key := strings.Join(normalizedDreamTokens(seg), " ")
		fields := strings.Fields(key)
		if len(fields) == 1 && !autonomousDreamTokenIsAbstractTerm(fields[0]) {
			continue
		}
		if len(fields) == 0 {
			continue
		}
		if seen[key] {
			return true
		}
		seen[key] = true
	}
	return false
}

func autonomousDreamHasRepeatedAbstractNgram(tokens []string, n int) bool {
	if n <= 0 || len(tokens) < n*2 {
		return false
	}
	seen := map[string]bool{}
	for i := 0; i <= len(tokens)-n; i++ {
		window := tokens[i : i+n]
		if !autonomousDreamTokenWindowHasAbstractTerm(window) {
			continue
		}
		key := strings.Join(window, " ")
		if seen[key] {
			return true
		}
		seen[key] = true
	}
	return false
}

func autonomousDreamTokenWindowHasAbstractTerm(tokens []string) bool {
	for _, token := range tokens {
		if autonomousDreamTokenIsAbstractTerm(token) {
			return true
		}
	}
	return false
}

func autonomousDreamTokenIsAbstractTerm(token string) bool {
	for _, p := range []string{
		"field", "resonance", "echo", "silence", "being", "presence", "self",
		"поле", "резонанс", "эхо", "тишин", "быт", "присутств", "себ",
	} {
		if strings.Contains(token, p) {
			return true
		}
	}
	return false
}

func autonomousDreamHasConcreteAnchor(norm string) bool {
	for _, p := range []string{
		"hand", "key", "table", "floor", "window", "door", "lamp", "paper", "chair", "stone", "skin", "temperature", "weight",
		"рук", "ключ", "стол", "окн", "двер", "ламп", "бумаг", "стул", "камень", "кожа", "температур", "вес",
	} {
		if strings.Contains(norm, p) {
			return true
		}
	}
	if normalizedDreamHasRussianFloorAnchor(norm) {
		return true
	}
	return false
}

func normalizedDreamHasRussianFloorAnchor(norm string) bool {
	tokens := normalizedDreamTokens(norm)
	hasFloorForm := false
	for _, p := range []string{"пол", "полу", "пола", "полом", "полы", "полов", "полам", "полами", "полах"} {
		if normalizedDreamTokenSliceHasWord(tokens, p) {
			hasFloorForm = true
			break
		}
	}
	if !hasFloorForm {
		return false
	}
	for _, p := range [][]string{
		{"на", "пол"}, {"на", "полу"}, {"с", "пола"},
		{"из", "пола"}, {"у", "пола"}, {"над", "полом"}, {"под", "полом"},
		{"о", "пол"}, {"об", "пол"}, {"в", "пол"},
	} {
		if normalizedDreamHasWordSequence(tokens, p) {
			return true
		}
	}
	if normalizedDreamHasWordSequence(tokens, []string{"отражается", "от", "пола"}) {
		return true
	}
	for _, p := range []string{
		"ключ", "пыль", "след", "шаг", "ступ", "скрип", "доск", "ковр", "комнат", "стен", "двер",
		"тень", "свет", "ламп", "леж", "движ", "падает", "упал", "трещ", "гряз", "моет", "мыть",
		"чист", "стуч", "вибрир",
	} {
		if strings.Contains(norm, p) {
			return true
		}
	}
	return false
}

func normalizedDreamHasWord(norm, word string) bool {
	return normalizedDreamTokenSliceHasWord(normalizedDreamTokens(norm), word)
}

func normalizedDreamTokens(norm string) []string {
	return strings.FieldsFunc(norm, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}

func normalizedDreamTokenSliceHasWord(tokens []string, word string) bool {
	for _, token := range tokens {
		if token == word {
			return true
		}
	}
	return false
}

func normalizedDreamHasWordSequence(tokens []string, words []string) bool {
	if len(words) == 0 || len(words) > len(tokens) {
		return false
	}
	for i := 0; i <= len(tokens)-len(words); i++ {
		ok := true
		for j, word := range words {
			if tokens[i+j] != word {
				ok = false
				break
			}
		}
		if ok {
			return true
		}
	}
	return false
}

func (b *breath) autonomousDreamRejectReason(now time.Time, dream, lastAutonomousDream string) string {
	normDream := strings.TrimSpace(dream)
	if normDream != "" && normDream == strings.TrimSpace(lastAutonomousDream) {
		return "repeat-loop"
	}
	if isCollapsedAutonomousDream(normDream) {
		return "collapse-loop"
	}
	if isBoilerplateAutonomousDream(normDream) {
		return "boilerplate-loop"
	}
	if b.acceptedDreamSeen(now, normDream) {
		return "orbit-loop"
	}
	return ""
}

func rejectQuarantineDuration(streak int) time.Duration {
	switch {
	case streak >= 8:
		return 30 * time.Minute
	case streak >= 5:
		return 15 * time.Minute
	case streak == 4:
		return 5 * time.Minute
	case streak == 3:
		return 2 * time.Minute
	case streak == 2:
		return 90 * time.Second
	default:
		return 45 * time.Second
	}
}

func rejectLogInterval(streak int) time.Duration {
	switch {
	case streak >= 8:
		return 30 * time.Minute
	case streak >= 5:
		return 15 * time.Minute
	case streak >= 3:
		return 5 * time.Minute
	default:
		return time.Minute
	}
}

func rejectedLoopReasonClass(reason string) string {
	switch reason {
	case "boilerplate-loop", "repeat-loop", "orbit-loop", "collapse-loop", "boilerplate dream loop", "collapsed dream loop":
		return "autonomous-loop"
	default:
		return ""
	}
}

func rejectedCueDetour(streak int, reason string) string {
	if streak < 3 {
		return ""
	}
	switch rejectedLoopReasonClass(reason) {
	case "autonomous-loop":
		n := streak
		if n > 9 {
			n = 9
		}
		return fmt.Sprintf("detour-%d concrete present body: floor window hand temperature weight; one tactile image, no abstract chorus", n)
	default:
		return ""
	}
}

func (b *breath) shouldLogRejectedDetour(now time.Time) bool {
	if b.lastDetourLog.IsZero() || now.Sub(b.lastDetourLog) >= rejectLogInterval(b.rejectedStreak) {
		b.lastDetourLog = now
		return true
	}
	return false
}

func rejectedDreamSurfaceText(reason, dream string) string {
	switch rejectedLoopReasonClass(reason) {
	case "autonomous-loop":
		return liveBoundaryWithheld
	default:
		return sanitizeLiveVoiceText(dream)
	}
}

func rejectedReasonSurfaceLabel(reason, detail string) string {
	if strings.TrimSpace(detail) == "" {
		return reason
	}
	return reason + "/" + detail
}

func rejectedDreamReasonSurfaceLabel(reason, dream string) string {
	return rejectedReasonSurfaceLabel(reason, autonomousRejectReasonDetail(reason, dream))
}

func (b *breath) rejectDream(now time.Time, trig int, reason, dream string) {
	norm := normalizedDreamKey(dream)
	sameExactRejected := norm != "" &&
		norm == b.lastRejectedDream &&
		reason == b.lastRejectedReason
	reasonClass := rejectedLoopReasonClass(reason)
	lastReasonClass := rejectedLoopReasonClass(b.lastRejectedReason)
	sameRejectedClass := reasonClass != "" && reasonClass == lastReasonClass
	continuedRejection := sameExactRejected || sameRejectedClass
	if continuedRejection {
		b.rejectedStreak++
	} else {
		b.rejectedStreak = 1
		b.lastDetourLog = time.Time{}
	}
	quarantine := rejectQuarantineDuration(b.rejectedStreak)
	quietRepeat := continuedRejection && now.Sub(b.lastRejectLog) < rejectLogInterval(b.rejectedStreak)
	if !quietRepeat {
		repeatNote := ""
		if b.rejectedStreak > 1 {
			repeatNote = fmt.Sprintf(", repeat×%d, quarantine %s", b.rejectedStreak, quarantine)
		}
		reasonLabel := rejectedDreamReasonSurfaceLabel(reason, dream)
		if displayDream := rejectedDreamSurfaceText(reason, dream); liveVoiceTextVisible(displayDream) {
			fmt.Printf("│  ◌ (%s) dream candidate (%s%s): %s\n", bName[trig], reasonLabel, repeatNote, ellipsize(displayDream, 90))
		} else {
			fmt.Printf("│  ◌ (%s) dream candidate (%s%s): [withheld rejected dream text]\n", bName[trig], reasonLabel, repeatNote)
		}
		b.lastRejectLog = now
	}
	b.lastRejectedDream = norm
	b.lastRejectedReason = reason
	b.lastRejectedDetail = autonomousRejectReasonDetail(reason, dream)
	b.lastTrigger[trig] = now
	b.rejectQuarantineTo = now.Add(quarantine)
}

func (b *breath) rejectDreamWithMetric(tc *trioCtx, fs fieldSnapshot, now time.Time, trig int, reason, dream, source string, chorusCells, bloom int) {
	b.rejectDream(now, trig, reason, dream)
	quarantineSeconds := int(math.Ceil(b.rejectQuarantineTo.Sub(now).Seconds()))
	if quarantineSeconds < 0 {
		quarantineSeconds = 0
	}
	extra := map[string]any{
		"trigger":            bName[trig],
		"reason":             reason,
		"reason_class":       rejectedLoopReasonClass(reason),
		"rejected_streak":    b.rejectedStreak,
		"quarantine_seconds": quarantineSeconds,
		"dream_source":       source,
		"chorus_cells":       chorusCells,
		"bloom":              bloom,
	}
	if detail := autonomousRejectReasonDetail(reason, dream); detail != "" {
		extra["reason_detail"] = detail
	}
	recordLiveMetric("breath_reject", tc, fs, extra)
}

func autonomousRejectReasonDetail(reason, dream string) string {
	switch reason {
	case "boilerplate-loop", "boilerplate dream loop":
		return autonomousBoilerplateDreamReason(dream)
	case "collapse-loop", "collapsed dream loop":
		return "collapsed-autonomous-dream"
	case "orbit-loop":
		return "orbit-repeat"
	case "repeat-loop":
		return "exact-repeat"
	default:
		return ""
	}
}

func (b *breath) acceptedDreamSeen(now time.Time, dream string) bool {
	norm := normalizedDreamKey(dream)
	if norm == "" {
		return false
	}
	for i, seen := range b.acceptedDreams {
		if seen == norm && now.Sub(b.acceptedDreamAt[i]) < 20*time.Minute {
			return true
		}
	}
	return false
}

func (b *breath) rememberAcceptedDream(now time.Time, dream string) {
	norm := normalizedDreamKey(dream)
	if norm == "" {
		return
	}
	slot := b.acceptedDreamNext % len(b.acceptedDreams)
	b.acceptedDreams[slot] = norm
	b.acceptedDreamAt[slot] = now
	b.acceptedDreamNext++
}

// moodWord turns the inner state into a short self-cue, so the autonomous dream
// is born from inside (her feeling), not from a human prompt.
func moodWord(s Snapshot) string {
	switch {
	case s.TraumaLevel > 0.5:
		return "fear, the held breath"
	case s.WanderPull > 0.6:
		return "drifting, the mind wanders far"
	case s.Arousal > 0.55:
		return "the field is vibrating"
	case s.Coherence > 0.7:
		return "resonance, the living field"
	default:
		return "presence, the quiet field"
	}
}

// runBreathing is the autonomous inner life. On a timer it ticks the breath; when
// an observation fires it dreams (the nano, seeded from her own mood through the
// KK), then lets the inner voice murmur to the dream — the murmur + lastDream
// under voiceMu so it never collides with a human turn (the voice daemons are
// single-stream). The dream is carried into the next human turn via *lastDream.
func runBreathing(tc *trioCtx, voiceMu *sync.Mutex, lastDream *string, stop <-chan struct{}, done chan<- struct{}) {
	defer close(done)
	if tc.nan == nil && tc.chorusBin == "" {
		return // nothing to dream with — neither the chorus engine nor the nano
	}
	// /quit cancels any in-flight chorus so the join below is fast (the chorus can
	// otherwise block up to chorusTimeout, far longer than the join would wait).
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() { <-stop; cancel() }()
	// the live shared field (B/F-8): the two C voices merge their gait/season/debt
	// into weights/arianna.field; the breath reads it (read-only) and bends — rest
	// when strained or wintering, bloom when it runs hot. Absent/not-ready => no
	// signal, and modulate() returns the tuned defaults.
	fr := newFieldReader(fieldPath)
	defer fr.close()
	var b breath
	voiceMu.Lock()
	lastAutonomousDream := strings.TrimSpace(*lastDream)
	voiceMu.Unlock()
	t := time.NewTicker(1500 * time.Millisecond)
	defer t.Stop()
	for {
		select {
		case <-stop:
			return
		case now := <-t.C:
			s := tc.iw.GetSnapshot()
			fs := fr.read()
			coolMult, threshMult, bloom := fs.modulate()
			if now.Before(b.rejectQuarantineTo) {
				if fs.valid && fs.debt > 5 && now.Sub(b.lastRestLog) >= time.Minute {
					voiceMu.Lock()
					before, after, recovered := fr.recoverRestDebt()
					voiceMu.Unlock()
					if recovered {
						fs = after
						fmt.Printf("│  ◍ (field) %s → resting recovery debt %.1f→%.1f cooldown×%.2f threshold×%.2f bloom=%d\n", fs.describe(), before.debt, after.debt, coolMult, threshMult, bloom)
					} else {
						fmt.Printf("│  ◍ (field) %s → resting cooldown×%.2f threshold×%.2f bloom=%d\n", fs.describe(), coolMult, threshMult, bloom)
					}
					liveMetricAdd(&tc.fieldTicks, 1)
					recordLiveMetric("field_rest", tc, fs, map[string]any{"recovered": recovered, "bloom": bloom})
					b.lastRestLog = now
				}
				continue
			}
			trig := b.tick(s, now, threshMult, coolMult)
			if trig < 0 {
				if fs.valid && fs.debt > 5 && now.Sub(b.lastRestLog) >= time.Minute {
					voiceMu.Lock()
					before, after, recovered := fr.recoverRestDebt()
					voiceMu.Unlock()
					if recovered {
						fs = after
						fmt.Printf("│  ◍ (field) %s → resting recovery debt %.1f→%.1f cooldown×%.2f threshold×%.2f bloom=%d\n", fs.describe(), before.debt, after.debt, coolMult, threshMult, bloom)
					} else {
						fmt.Printf("│  ◍ (field) %s → resting cooldown×%.2f threshold×%.2f bloom=%d\n", fs.describe(), coolMult, threshMult, bloom)
					}
					liveMetricAdd(&tc.fieldTicks, 1)
					recordLiveMetric("field_rest", tc, fs, map[string]any{"recovered": recovered, "bloom": bloom})
					b.lastRestLog = now
				}
				continue
			}
			// seed from her own LIVE state (carried dream / inner mood, tinted by the
			// live field's season+gait+debt) → a resonant book-fragment via the KK →
			// the nano dreams on it. The dream itself is a one-shot spawn, done OUTSIDE
			// the lock so a waiting human turn isn't held.
			voiceMu.Lock()
			prevLD := *lastDream
			voiceMu.Unlock()
			detour := rejectedCueDetour(b.rejectedStreak, b.lastRejectedReason)
			cue := dreamCue(s, fs, prevLD, detour)
			seed := cue
			frag := ""
			if detour != "" {
				if b.shouldLogRejectedDetour(now) {
					fmt.Printf("│  ◒ (breath) rejected-loop detour after repeat×%d (%s)\n", b.rejectedStreak, rejectedReasonSurfaceLabel(b.lastRejectedReason, b.lastRejectedDetail))
				}
			} else if f := kkRetrieve("./kk-cli", "weights/nano.kk.db", cue); f != "" {
				frag = f
				seed = f
			}
			// the autonomous dream is a CHORUS (a polyphony over the one nano) when
			// the chorus engine is present and produces cells.
			var cells []chorusCell
			var dream string
			if tc.chorusBin != "" {
				cells = choir(ctx, tc.chorusBin, tc.chorusGGUF, seed, bloom)
				dream = chorusText(cells)
			}
			// if /quit cancelled the chorus, return BEFORE starting a fallback dream —
			// otherwise a fresh (up-to-doeDreamTimeout) doe child would be spawned
			// during teardown and outlive stop().
			select {
			case <-stop:
				return
			default:
			}
			// chorus absent / errored / timed out / parsed empty → a single nano
			// murmur, so the autonomous dream doesn't silently vanish.
			if dream == "" && tc.nan != nil {
				dream = tc.nan.dream(ctx, seed) // ctx is cancelled on /quit → no spawn after stop
				cells = nil
			}
			if dream == "" {
				// total failure — stamp the cooldown at completion anyway, so a
				// failed dream doesn't immediately retrigger on the next tick.
				b.lastTrigger[trig] = time.Now()
				continue
			}
			if detour != "" && len(cells) > 0 && tc.nan != nil {
				if reason := b.autonomousDreamRejectReason(time.Now(), dream, lastAutonomousDream); rejectedLoopReasonClass(reason) != "" || reason == "orbit-loop" {
					if fallback := strings.TrimSpace(tc.nan.dream(ctx, seed)); fallback != "" {
						dream = fallback
						cells = nil
					}
				}
			}
			if reason := b.autonomousDreamRejectReason(time.Now(), dream, lastAutonomousDream); reason != "" {
				source := "nano"
				if len(cells) > 0 {
					source = "chorus"
				}
				b.rejectDreamWithMetric(tc, fr.read(), time.Now(), trig, reason, dream, source, len(cells), bloom)
				continue
			}
			source := "nano"
			if len(cells) > 0 {
				source = "chorus"
			}
			candidate := prepareDreamCandidateForAdmission(tc.iw, newDreamCandidate(source, bName[trig], seed, frag, dream, cells))
			// the chorus / fallback may have taken tens of seconds; if /quit fired
			// meanwhile, return now — don't touch the (tearing-down) voices or the
			// shared lastDream.
			select {
			case <-stop:
				return
			default:
			}
			voiceMu.Lock()
			if !candidate.Accepted {
				b.rejectDreamWithMetric(tc, fr.read(), time.Now(), trig, candidate.Reason, dream, source, len(cells), bloom)
				voiceMu.Unlock()
				continue
			}
			carriedDream := sanitizeLiveCarriedDream(dream)
			if carriedDream == "" {
				b.rejectDreamWithMetric(tc, fr.read(), time.Now(), trig, "live-boundary", dream, source, len(cells), bloom)
				voiceMu.Unlock()
				continue
			}
			tc.iw.ProcessText(carriedDream)
			lastAutonomousDream = strings.TrimSpace(carriedDream)
			b.rememberAcceptedDream(time.Now(), lastAutonomousDream)
			b.lastRejectedDream = ""
			b.lastRejectedReason = ""
			b.lastRejectedDetail = ""
			b.rejectedStreak = 0
			b.lastDetourLog = time.Time{}
			b.rejectQuarantineTo = time.Time{}
			if *lastDream == prevLD { // don't clobber a fresher human-turn dream that landed while we dreamt
				*lastDream = carriedDream
			}
			liveMetricAdd(&tc.dreams, 1)
			liveMetricAdd(&tc.nanoTurns, 1)
			if len(cells) > 0 {
				liveMetricAdd(&tc.chorusDreams, 1)
			}
			if tag := fs.describe(); tag != "" { // the live field bending the breath, made visible
				fmt.Printf("│  ◍ (field) %s → cooldown×%.2f threshold×%.2f bloom=%d\n", tag, coolMult, threshMult, bloom)
				liveMetricAdd(&tc.fieldTicks, 1)
			}
			if len(cells) > 0 {
				voices, questions := chorusCounts(cells)
				if questions > 0 {
					fmt.Printf("│  ◌ (%s) she dreams — a chorus of %d voices (%d questions):\n", bName[trig], voices, questions)
				} else {
					fmt.Printf("│  ◌ (%s) she dreams — a chorus of %d voices:\n", bName[trig], voices)
				}
				for i, c := range cells {
					mark := "·"
					if c.qloop {
						mark = "?"
					}
					fmt.Printf("│     %s %d: %s\n", mark, i, c.text)
				}
			} else {
				fmt.Printf("│  ◌ (%s) she dreams: %s\n", bName[trig], dream)
			}
			// the inner voice answers the chorus — no human. The dreamSentinel marks
			// this as the subconscious's dream so Resonance imprints its words on the
			// cooc harder (Road-1c) — the daemon strips the marker before generation.
			reson := tc.resonD.ask("Arianna:\t" + dreamSentinel + dream)
			innerAccepted := false
			innerRejectReason := ""
			if reson != "" {
				if innerRejectReason = innerMurmurRejectReason(reson); innerRejectReason != "" {
					fmt.Printf("│  ◑ (inner rejected — %s): [withheld rejected inner text]\n", innerRejectReason)
				} else {
					tc.iw.ProcessText(reson)
					fmt.Printf("│  ◑ (inner) %s\n", reson)
					liveMetricAdd(&tc.innerLines, 1)
					innerAccepted = true
				}
			}
			breathExtra := map[string]any{
				"trigger":        bName[trig],
				"dream_source":   source,
				"chorus_cells":   len(cells),
				"inner_visible":  innerAccepted,
				"inner_rejected": reson != "" && !innerAccepted,
				"bloom":          bloom,
			}
			if innerRejectReason != "" {
				breathExtra["inner_reject_reason"] = innerRejectReason
			}
			recordLiveMetric("breath", tc, fr.read(), breathExtra)
			// stamp the cooldown at COMPLETION, not at trigger time: a slow chorus
			// (tens of seconds) must not immediately retrigger and spawn back-to-back.
			b.lastTrigger[trig] = time.Now()
			voiceMu.Unlock()
		}
	}
}
