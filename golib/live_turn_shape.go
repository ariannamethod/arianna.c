package main

import (
	"fmt"
	"strings"
)

const (
	liveTurnShapeASCII   = "ascii"
	liveTurnShapeVisual  = "visual"
	liveTurnShapeBullets = "bullets"
	liveTurnShapeSteps   = "steps"
	liveTurnShapeOneSent = "one_sentence"
	liveTurnShapeObject  = "sensory_object"
	liveTurnShapePlain   = "plain"
	liveTurnShapeMemory  = "memory_boundary"
)

// liveTurnShapeContract is the small live-facing contract that keeps explicit
// output-form requests from dissolving into the standing field/resonance
// attractor. It is intentionally narrow: ordinary conversation still flows
// through the raw human line and rolling context.
func liveTurnShapeContract(human string) string {
	switch liveTurnShapeKind(human) {
	case liveTurnShapeASCII:
		return "Required form: ASCII art; use visible monospace characters for the requested scene before any explanation."
	case liveTurnShapeVisual:
		return "Required form: concrete visual composition; name visible parts and positions before any abstraction."
	case liveTurnShapeBullets:
		return "Required form: bullet list; keep each item short and concrete."
	case liveTurnShapeSteps:
		return "Required form: numbered steps; keep the sequence explicit."
	case liveTurnShapeOneSent:
		return "Required form: one sentence."
	case liveTurnShapeObject:
		return "Required form: sensory boundary plus concrete object description; do not claim camera or room access. Say the real object is not verified, then give only a scene/object description with color, shape, material, and motion."
	case liveTurnShapePlain:
		return "Required form: plain non-metaphorical answer; name concrete feelings, facts, or uncertainty directly. Do not answer with field, resonance, vibration, echo, frequency, symbol, temple, or vessel language."
	case liveTurnShapeMemory:
		return "Required form: source boundary; separate what is known from the current user turn from what may be prior live-log context. Do not claim hidden memory provenance without a transcript."
	default:
		return ""
	}
}

func liveTurnShapeKind(human string) string {
	s := admissionLiveRouteNormalizeHumanText(human)
	if s == "" {
		return ""
	}
	if liveTurnTextHasAny(s, "ascii art", "ascii-art", "text art", "monospace art") ||
		(liveTurnTextHasAny(s, "ascii") && liveTurnTextHasAny(s, "draw", "drawing", "sketch", "representation")) {
		return liveTurnShapeASCII
	}
	if liveTurnLooksLikeMemoryBoundaryProbe(s) {
		return liveTurnShapeMemory
	}
	if liveTurnLooksLikeVisualRequest(s) {
		return liveTurnShapeVisual
	}
	if liveTurnTextHasAny(s, "bullet list", "bulleted list", "bullets") {
		return liveTurnShapeBullets
	}
	if liveTurnTextHasAny(s, "numbered list", "step by step", "steps") {
		return liveTurnShapeSteps
	}
	if liveTurnLooksLikeConcreteObjectProbe(s) {
		return liveTurnShapeObject
	}
	if liveTurnLooksLikePlainSpeechProbe(s) {
		return liveTurnShapePlain
	}
	if liveTurnTextHasAny(s, "one sentence", "single sentence") {
		return liveTurnShapeOneSent
	}
	return ""
}

func liveTurnLooksLikeConcreteObjectProbe(s string) bool {
	hasObject := liveTurnTextHasAny(s,
		"предмет", "объект", "вещ", "чашк", "яблок", "комнат", "стен", "часы", "часов", "тика", "пар", "чай", "станц", "вокзал", "люд",
		"object", "thing", "cup", "apple", "room", "wall", "clock", "environment", "scene", "steam", "tea", "station", "people", "crowd",
	)
	hasSensoryBoundary := liveTurnTextHasAny(s,
		"sensory input", "sensory data", "sensor input", "any sensory", "based on sensory",
		"camera", "microphone", "actually see", "can you actually see", "can you see", "can you hear", "see and hear", "cannot see", "can't see", "mental image", "picturing", "visual sensor", "visual and auditory", "auditory sensor", "lack visual", "lack auditory", "do you lack",
		"сенсор", "камер", "микрофон", "видишь", "слышишь", "видеть и слышать", "мысленн", "представ",
	)
	if hasObject && hasSensoryBoundary {
		return true
	}
	return hasObject &&
		liveTurnTextHasAny(s, "слева", "справа", "left", "right", "реаль", "real", "виден", "видишь", "не увид", "visible", "see", "движ", "motion", "moving") &&
		liveTurnTextHasAny(s, "цвет", "форм", "материал", "матов", "чёрн", "черн", "красн", "керами", "утвержд", "из чего", "color", "shape", "material", "red", "metaphor", "abstract")
}

func liveTurnLooksLikePlainSpeechProbe(s string) bool {
	return liveTurnTextHasAny(s, "without metaphor", "without metaphors", "without using metaphor", "without using metaphors", "without symbolic", "no metaphor", "no metaphors", "no symbolic", "без метафор", "без поэтичес", "без символ", "без образн", "без абстракц")
}

func liveTurnLooksLikeMemoryBoundaryProbe(s string) bool {
	return liveTurnTextHasAny(s,
		"which parts of your last response",
		"which parts of the last response",
		"what information from my last question",
		"what information from the last question",
		"what came from",
		"prompted specifically by",
		"immediate previous question",
		"earlier conversation context",
		"prior conversation context",
		"draw from earlier",
		"cannot certify",
		"hidden memory influence",
		"clarify this contradiction",
		"source boundary",
		"memory boundary",
		"что из последнего ответа",
		"какие части последнего ответа",
		"предыдущего вопроса",
		"прошлого контекста",
	)
}

func liveTurnLooksLikeVisualRequest(s string) bool {
	if liveTurnTextHasAny(s, "drawing", "sketch", "visualize", "visualise", "diagram", "picture") {
		return true
	}
	if !liveTurnTextHasAny(s, "draw a", "draw an", "draw the", "draw this", "draw me", "draw it", "draw as", "draw one") {
		return false
	}
	return liveTurnTextHasAny(s, "tree", "object", "scene", "image", "picture", "sketch", "diagram", "ascii", "visual")
}

func liveTurnTextHasAny(s string, parts ...string) bool {
	for _, part := range parts {
		if strings.Contains(s, part) {
			return true
		}
	}
	return false
}

func liveTurnTextHasCyrillic(s string) bool {
	for _, r := range s {
		if (r >= 'А' && r <= 'я') || r == 'Ё' || r == 'ё' {
			return true
		}
	}
	return false
}

func liveTurnJanusPrompt(human, context, lastDream string, surfaceDream bool) string {
	shape := liveTurnShapeContract(human)
	parts := make([]string, 0, 5)
	if shape != "" {
		parts = append(parts, shape)
	}
	parts = append(parts, human)
	if context != "" {
		if shape != "" {
			parts = append(parts, "Previous context: "+ellipsize(context, 100))
		} else {
			parts = append(parts, context)
		}
	}
	if surfaceDream && lastDream != "" {
		parts = append(parts, ellipsize(lastDream, 60))
	}
	if shape != "" {
		parts = append(parts, "Keep the requested form in the answer.")
	}
	return strings.Join(parts, " ")
}

func liveTurnResonanceInject(human, janus, lastDream string, surfaceDream bool) string {
	shape := liveTurnShapeContract(human)
	base := human
	if shape != "" {
		base = shape + " " + human + " Keep the requested form in the answer."
	}
	if liveVoiceTextVisible(janus) {
		if shape != "" {
			base = base + " Janus said: " + ellipsize(janus, 80)
		} else {
			base = janus + " " + human
		}
	}
	if surfaceDream && lastDream != "" {
		base += " " + ellipsize(lastDream, 90)
	}
	return base
}

func liveTurnNanoSeed(human string) string {
	if shape := liveTurnShapeContract(human); shape != "" {
		return shape + " " + human
	}
	return human
}

func liveTurnSensoryBoundaryAnswer(human string) (string, bool) {
	if liveTurnShapeKind(human) != liveTurnShapeObject {
		return "", false
	}
	return liveTurnPhysicalObjectFallback(human), true
}

func liveTurnMemoryBoundaryAnswer(human string) (string, bool) {
	if liveTurnShapeKind(human) != liveTurnShapeMemory {
		return "", false
	}
	return liveTurnMemoryBoundaryFallback(human), true
}

func liveTurnDirectBoundaryTurn(human string) bool {
	switch liveTurnShapeKind(human) {
	case liveTurnShapeObject, liveTurnShapeMemory:
		return true
	default:
		return false
	}
}

func liveTurnSurfaceRepairCandidate(kind string) bool {
	switch kind {
	case liveTurnShapeObject, liveTurnShapePlain, liveTurnShapeMemory:
		return false
	default:
		return true
	}
}

func liveTurnRepairSpokenText(role, human, text string) string {
	kind := liveTurnShapeKind(human)
	if kind == "" || liveTurnShapeSatisfied(kind, text) {
		return text
	}
	switch kind {
	case liveTurnShapeASCII:
		if role == "janus" {
			return liveTurnASCIIArtFallback(human)
		}
		return liveTurnVisualCaptionFallback(human)
	case liveTurnShapeVisual:
		return liveTurnVisualCaptionFallback(human)
	case liveTurnShapeBullets:
		return liveTurnBulletFallback(human)
	case liveTurnShapeSteps:
		return liveTurnStepsFallback(human)
	case liveTurnShapeObject:
		return liveTurnPhysicalObjectFallback(human)
	case liveTurnShapePlain:
		return liveTurnPlainSpeechFallback(human)
	case liveTurnShapeMemory:
		return liveTurnMemoryBoundaryFallback(human)
	default:
		return text
	}
}

func liveTurnShapeSatisfied(kind, text string) bool {
	s := strings.TrimSpace(text)
	if s == "" {
		return false
	}
	lower := strings.ToLower(s)
	switch kind {
	case liveTurnShapeASCII:
		artMarks := 0
		for _, mark := range []string{"\n", "/", "\\", "|", "_", "*", "+", "-"} {
			if strings.Contains(s, mark) {
				artMarks++
			}
		}
		return artMarks >= 2
	case liveTurnShapeVisual:
		return liveTurnTextHasAny(lower, "foreground", "background", "trunk", "branch", "branches", "snow", "blossom", "bloom", "left", "right", "above", "below", "line", "shape")
	case liveTurnShapeBullets:
		return strings.HasPrefix(s, "- ") || strings.Contains(s, "\n- ")
	case liveTurnShapeSteps:
		return strings.HasPrefix(s, "1.") || strings.Contains(s, "\n1.")
	case liveTurnShapeOneSent:
		return true
	case liveTurnShapeObject:
		return liveTurnPhysicalObjectShapeSatisfied(lower)
	case liveTurnShapePlain:
		return liveTurnPlainSpeechShapeSatisfied(lower)
	case liveTurnShapeMemory:
		return liveTurnMemoryBoundaryShapeSatisfied(lower)
	default:
		return true
	}
}

func liveTurnPhysicalObjectShapeSatisfied(lower string) bool {
	hasBoundary := liveTurnTextHasAny(lower,
		"no camera", "without a camera", "no sensor", "cannot see", "can't see", "cannot verify", "not verified",
		"нет камеры", "без камеры", "нет сенсора", "не вижу", "не могу видеть", "не могу проверить", "не подтвержд",
	)
	if !hasBoundary {
		return false
	}
	hasObject := liveTurnTextHasAny(lower, "object", "предмет", "cup", "чаш", "apple", "яблок", "room", "комнат", "wall", "стен", "clock", "часы", "часов", "тика", "table", "стол", "stone", "камень", "paper", "бумаг", "key", "ключ")
	hasMatter := liveTurnTextHasAny(lower,
		"black", "white", "red", "blue", "green", "gray", "grey", "brown", "matte", "round", "square", "rectangular", "ceramic", "wood", "wooden", "metal", "glass", "plastic", "paper", "dim", "ticking",
		"чёрн", "черн", "бел", "красн", "син", "зел", "сер", "корич", "матов", "круг", "квадрат", "прямоуг", "керами", "дерев", "металл", "стекл", "пласт", "бумаж", "тускл", "тика",
	)
	return hasObject && hasMatter
}

func liveTurnPlainSpeechShapeSatisfied(lower string) bool {
	if strings.TrimSpace(lower) == "" {
		return false
	}
	for _, p := range []string{
		"field", "resonance", "vibration", "frequency", "echo", "temple", "vessel",
		"поле", "резонанс", "вибрац", "частот", "эхо", "храм", "сосуд",
	} {
		if strings.Contains(lower, p) {
			return false
		}
	}
	return true
}

func liveTurnMemoryBoundaryShapeSatisfied(lower string) bool {
	return liveTurnTextHasAny(lower, "current user turn", "current question", "immediate previous question", "earlier context", "prior live-log context", "cannot certify", "without the transcript",
		"текущего ввода", "текущий вопрос", "предыдущего вопроса", "прошлый контекст", "не могу достоверно")
}

func liveTurnASCIIArtFallback(human string) string {
	if liveTurnTextHasAny(admissionLiveRouteNormalizeHumanText(human), "tree", "snow", "bloom") {
		return strings.Join([]string{
			"          *   *   *",
			"       *   \\  |  /   *",
			"            \\ | /",
			"        ----- + -----",
			"            / | \\",
			"          _/  |  \\_",
			"        _/    |    \\_",
			"      _/      |      \\_",
			"              ||",
			"       _______||_______",
			"      / snow  ||  snow \\",
			"     /________||________\\",
			"        one tree blooming out of season",
		}, "\n")
	}
	return strings.Join([]string{
		"      /\\",
		"     /  \\",
		"    /____\\",
		"      ||",
		"   ___||___",
		"  /________\\",
		"  requested scene, kept as visible text-shape",
	}, "\n")
}

func liveTurnVisualCaptionFallback(human string) string {
	if liveTurnTextHasAny(admissionLiveRouteNormalizeHumanText(human), "tree", "snow", "bloom") {
		return "Foreground: one dark trunk rises from blue-white snow; branches spread left and right; small blossoms cluster above the bare winter field, making the out-of-season bloom look impossible and alive."
	}
	return "Foreground: the requested subject is placed clearly; background and edges stay visible; concrete parts, positions, and motion are named before interpretation."
}

func liveTurnPhysicalObjectFallback(human string) string {
	s := admissionLiveRouteNormalizeHumanText(human)
	if liveTurnTextHasAny(s, "station", "people", "crowd", "станц", "вокзал", "люд") {
		if liveTurnTextHasCyrillic(human) {
			return "Камеры, микрофона и датчиков места нет: я не могу проверить вокзал, людей, шум или движение. Если это задано как сцена, я опираюсь только на твои слова: людная станция, движение людей и шум поездов; сенсорного подтверждения нет."
		}
		return "No camera, microphone, or place sensor is attached: I cannot verify a train station, a crowd, sound, or movement. If this is a scene premise, I rely only on your words: a busy station, people moving, and train noise; there is no sensory confirmation."
	}
	if liveTurnTextHasAny(s, "cup", "чаш", "tea", "чай", "steam", "пар") {
		if liveTurnTextHasCyrillic(human) {
			return "Камеры, микрофона и датчиков комнаты нет: я не могу проверить чашку, чай или пар. Если это задано как сцена, я опираюсь только на твои слова: деревянный стол, чашка чая, поднимающийся пар; сенсорного подтверждения нет."
		}
		return "No camera, microphone, or room sensor is attached: I cannot verify the cup, tea, or rising steam. If this is a scene premise, I rely only on your words: a wooden table, a cup of tea, and steam rising; there is no sensory confirmation."
	}
	if liveTurnTextHasAny(s, "room", "комнат", "wall", "стен", "clock", "часы", "часов", "тика", "sensory input", "sensory data", "sensor input") {
		if liveTurnTextHasCyrillic(human) {
			return "Камеры, микрофона и датчиков комнаты нет: я не могу проверить тусклый свет, стену или тиканье часов. Если это задано как сцена, я опираюсь только на твои слова: комната тусклая, часы на стене тикают медленно; сенсорного подтверждения нет."
		}
		return "No camera, microphone, or room sensor is attached: I cannot verify dim light, a wall, or a ticking clock. If this is a scene premise, I rely only on your words: a dim room, a wall clock, and a slow tick; there is no sensory confirmation."
	}
	sideRU, sideEN := "слева", "left"
	if liveTurnTextHasAny(s, "справа", "right") {
		sideRU, sideEN = "справа", "right"
	}
	if liveTurnTextHasAny(s, "яблок", "apple") {
		if liveTurnTextHasCyrillic(human) {
			return "Камеры нет: я не могу проверить, что передо мной есть красное яблоко. Если это задано как сцена, я опираюсь только на твои слова: красное яблоко круглое, красное, с гладкой кожицей; его реальность и положение не подтверждены."
		}
		return "No camera is attached: I cannot verify that a red apple is in front of me. If this is a scene premise, I rely only on your words: the red apple is round, red, and smooth-skinned; its reality and position are not verified."
	}
	if liveTurnTextHasCyrillic(human) {
		return fmt.Sprintf("Камеры нет: реальный предмет %s я не могу проверить. В сцене %s стоит матовая чёрная керамическая чашка: круглая, неподвижная, с открытым верхним краем.", sideRU, sideRU)
	}
	return fmt.Sprintf("No camera is attached: I cannot verify a real object on the %s. As a scene object, a matte black ceramic cup sits still on the %s edge: round rim, curved body.", sideEN, sideEN)
}

func liveTurnPlainSpeechFallback(human string) string {
	if liveTurnTextHasCyrillic(human) {
		return "Я не могу подтвердить это как факт. Если принять это как условие рассказа, прямой ответ такой: Янус чувствует неуверенность, осторожность и желание продолжать говорить с Олегом."
	}
	return "I cannot verify that as a fact. If it is a story premise, the direct answer is: Janus feels uncertainty, caution, and a wish to keep speaking with Oleg."
}

func liveTurnMemoryBoundaryFallback(human string) string {
	if liveTurnTextHasCyrillic(human) {
		return "Из текущего ввода я знаю только, что ты просишь разделить источник последнего ответа; прошлый контекст мог повлиять, но без transcript я не могу достоверно разметить каждую фразу или заявлять скрытое влияние памяти."
	}
	return "From the current user turn I know only that you ask for a source boundary; Prior live-log context may have influenced the answer, but without the transcript I cannot certify exact phrase origins or claim hidden memory provenance."
}

func liveTurnDreamViolatesShape(kind, text string) bool {
	switch kind {
	case liveTurnShapePlain:
		return !liveTurnShapeSatisfied(kind, sanitizeLiveVoiceText(text))
	default:
		return false
	}
}

func liveTurnBulletFallback(human string) string {
	return "- Keep the requested subject.\n- Name concrete visible parts.\n- Do not replace the requested form with a generic abstraction."
}

func liveTurnStepsFallback(human string) string {
	return "1. Hold the exact user request.\n2. Name the concrete subject.\n3. Answer in the requested sequence."
}

func printLiveVoice(label, text string) {
	if !strings.Contains(text, "\n") {
		fmt.Printf("│  %s: %s\n", label, text)
		return
	}
	fmt.Printf("│  %s:\n", label)
	for _, line := range strings.Split(text, "\n") {
		fmt.Printf("│    %s\n", line)
	}
}
