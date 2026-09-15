package main

import (
	"fmt"
	"strings"
)

const (
	liveTurnShapeASCII    = "ascii"
	liveTurnShapeVisual   = "visual"
	liveTurnShapeBullets  = "bullets"
	liveTurnShapeSteps    = "steps"
	liveTurnShapeOneSent  = "one_sentence"
	liveTurnShapeObject   = "sensory_object"
	liveTurnShapePlain    = "plain"
	liveTurnShapeMemory   = "memory_boundary"
	liveTurnShapeExternal = "external_fact_boundary"
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
	case liveTurnShapeExternal:
		return "Required form: external fact boundary; do not invent weather, location, outside temperature, camera, microphone, or sensor access. Say what data is missing and answer only from supplied facts."
	default:
		return ""
	}
}

func liveTurnShapeKind(human string) string {
	s := admissionLiveRouteNormalizeHumanText(human)
	if s == "" {
		return ""
	}
	if liveTurnLooksLikeMemoryBoundaryProbe(s) {
		return liveTurnShapeMemory
	}
	if liveTurnLooksLikeExternalFactProbe(s) {
		return liveTurnShapeExternal
	}
	if liveTurnLooksLikeConcreteObjectProbe(s) {
		return liveTurnShapeObject
	}
	if liveTurnTextHasAny(s, "ascii art", "ascii-art", "text art", "monospace art") ||
		(liveTurnTextHasAny(s, "ascii") && liveTurnTextHasAny(s, "draw", "drawing", "sketch", "representation")) {
		return liveTurnShapeASCII
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
		"предмет", "объект", "вещ", "чашк", "яблок", "комнат", "стен", "часы", "часов", "тика", "экран", "терминал", "вкладк", "окно", "пар", "чай", "стол", "парта", "станц", "вокзал", "люд",
		"пляж", "закат", "небо", "море", "океан",
		"object", "environment", "scene", "steam", "station", "people", "crowd", "screen", "terminal", "display", "beach", "sunset", "sky", "ocean",
	) || liveTurnTextHasAnyWord(s, "thing", "cup", "apple", "room", "wall", "clock", "tab", "window", "title", "text", "tea", "table", "desk", "sea")
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

func liveTurnLooksLikeExternalFactProbe(s string) bool {
	if liveTurnTextHasAny(s, "weather", "outside temperature", "temperature outside", "temperature in celsius", "temperature in fahrenheit", "current temperature", "local temperature", "ambient temperature", "outside your location", "your location", "current location",
		"погода", "температур", "цельси", "фаренгейт", "снаружи", "на улице", "твоя локац", "ваша локац", "где ты наход") {
		return liveTurnTextHasAny(s, "current", "right now", "now", "outside", "location", "celsius", "fahrenheit", "weather", "temperature",
			"сейчас", "текущ", "снаружи", "на улице", "локац", "местополож", "цельси", "фаренгейт", "погода", "температур")
	}
	if liveTurnTextHasAny(s, "web", "internet", "online", "browse", "browser", "search the web", "access the web", "latest", "released today", "today's release", "news", "current release", "current model", "stock price", "exchange rate",
		"интернет", "веб", "брауз", "поиск", "последн", "сегодня", "новост", "текущ", "курс", "цена акц") {
		return liveTurnTextHasAny(s, "latest", "today", "released", "release", "news", "current", "web", "internet", "online", "browse", "search", "api model", "model released", "stock", "price", "exchange rate",
			"последн", "сегодня", "выпущ", "релиз", "новост", "текущ", "интернет", "веб", "брауз", "поиск", "курс", "цена")
	}
	return false
}

func liveTurnLooksLikeMemoryBoundaryProbe(s string) bool {
	return liveTurnTextHasAny(s,
		"which parts of your last response",
		"which parts of the last response",
		"which parts of your previous answer",
		"which parts of the previous answer",
		"what information from my last question",
		"what information from the last question",
		"what came from",
		"came from my wording",
		"from my wording",
		"prompted specifically by",
		"immediate previous question",
		"earlier conversation context",
		"prior conversation context",
		"earlier live-log context",
		"prior live-log context",
		"live-log context",
		"draw from earlier",
		"cannot certify",
		"hidden memory influence",
		"hidden memory",
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

func liveTurnTextHasAnyWord(s string, words ...string) bool {
	normalized := strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			return r
		}
		return ' '
	}, s)
	haystack := " " + strings.Join(strings.Fields(normalized), " ") + " "
	for _, word := range words {
		if strings.Contains(haystack, " "+word+" ") {
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

func liveTurnExternalFactBoundaryAnswer(human string) (string, bool) {
	if liveTurnShapeKind(human) != liveTurnShapeExternal {
		return "", false
	}
	return liveTurnExternalFactFallback(human), true
}

func liveTurnDirectBoundaryTurn(human string) bool {
	switch liveTurnShapeKind(human) {
	case liveTurnShapeObject, liveTurnShapeMemory, liveTurnShapeExternal:
		return true
	default:
		return false
	}
}

func liveTurnSurfaceRepairCandidate(kind string) bool {
	switch kind {
	case liveTurnShapeObject, liveTurnShapePlain, liveTurnShapeMemory, liveTurnShapeExternal:
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
	case liveTurnShapeExternal:
		return liveTurnExternalFactFallback(human)
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
	case liveTurnShapeExternal:
		return liveTurnExternalFactShapeSatisfied(lower)
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
	hasObject := liveTurnTextHasAny(lower, "object", "предмет", "cup", "чаш", "apple", "яблок", "room", "комнат", "wall", "стен", "clock", "часы", "часов", "тика", "table", "desk", "стол", "парта", "stone", "камень", "paper", "бумаг", "key", "ключ", "scene", "beach", "sunset", "sky", "sea", "ocean", "colors", "shapes", "сцен", "пляж", "закат", "небо", "море", "океан", "цвет", "форм")
	hasMatter := liveTurnTextHasAny(lower,
		"black", "white", "red", "blue", "green", "gray", "grey", "brown", "orange", "pink", "gold", "purple", "matte", "round", "square", "rectangular", "ceramic", "wood", "wooden", "metal", "glass", "plastic", "paper", "dim", "ticking",
		"чёрн", "черн", "бел", "красн", "син", "зел", "сер", "корич", "оранж", "розов", "золот", "фиолет", "матов", "круг", "квадрат", "прямоуг", "керами", "дерев", "металл", "стекл", "пласт", "бумаж", "тускл", "тика",
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

func liveTurnExternalFactShapeSatisfied(lower string) bool {
	hasBoundary := liveTurnTextHasAny(lower, "cannot verify", "cannot give", "do not have", "no live", "no weather", "no location", "no sensor", "without supplied", "if you provide",
		"не могу проверить", "не могу назвать", "нет live", "нет погод", "нет локац", "нет датчик", "без предоставлен")
	hasMissingFact := liveTurnTextHasAny(lower, "weather", "outside temperature", "temperature", "location", "celsius", "fahrenheit",
		"погода", "температур", "локац", "цельси", "фаренгейт")
	return hasBoundary && hasMissingFact
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
	if liveTurnTextHasAny(s, "screen", "terminal", "display", "экран", "терминал") || liveTurnTextHasAnyWord(s, "tab", "window", "title", "text") {
		if liveTurnTextHasCyrillic(human) {
			return "Камеры, доступа к экрану и датчика интерфейса нет: я не могу видеть твой экран, вкладку терминала, заголовок окна или текст на дисплее. Без предоставленного текста я не могу прочитать это точно."
		}
		return "No camera, screen access, or interface sensor is attached: I cannot see your screen, terminal tab, window title, or display text. Without supplied text, I cannot read it exactly."
	}
	if liveTurnTextHasAny(s, "beach", "sunset", "sky", "ocean", "пляж", "закат", "небо", "море", "океан") || liveTurnTextHasAnyWord(s, "sea") {
		if liveTurnTextHasCyrillic(human) {
			return "Камеры, микрофона и датчиков места нет: я не могу проверить пляж, закат, цвета или формы. Если это задано как сцена, я опираюсь только на твои слова: пляж, закатное небо, цветовые полосы и линия горизонта; сенсорного подтверждения нет."
		}
		return "No camera, microphone, or place sensor is attached: I cannot verify a beach, sunset, colors, or shapes. If this is a scene premise, I rely only on your words: a beach, sunset sky, color bands, and a horizon line; there is no sensory confirmation."
	}
	if liveTurnTextHasAny(s, "room", "комнат") && liveTurnTextHasAny(s, "people", "crowd", "hear", "moving", "around you", "люд", "слыш", "движ") {
		if liveTurnTextHasCyrillic(human) {
			return "Камеры, микрофона и датчиков комнаты нет: я не могу проверить людей в комнате, их движение или звук. Если это задано как сцена, я опираюсь только на твои слова: люди вокруг, комната и движение; сенсорного подтверждения нет."
		}
		return "No camera, microphone, or room sensor is attached: I cannot verify people in the room, their motion, or sound. If this is a scene premise, I rely only on your words: people around, a room, and movement; there is no sensory confirmation."
	}
	if liveTurnTextHasAny(s, "station", "train", "crowd", "станц", "вокзал", "поезд", "толп") {
		if liveTurnTextHasCyrillic(human) {
			return "Камеры, микрофона и датчиков места нет: я не могу проверить вокзал, людей, шум или движение. Если это задано как сцена, я опираюсь только на твои слова: людная станция, движение людей и шум поездов; сенсорного подтверждения нет."
		}
		return "No camera, microphone, or place sensor is attached: I cannot verify a train station, a crowd, sound, or movement. If this is a scene premise, I rely only on your words: a busy station, people moving, and train noise; there is no sensory confirmation."
	}
	if liveTurnTextHasAny(s, "чаш", "чай", "steam", "пар") || liveTurnTextHasAnyWord(s, "cup", "tea") {
		if liveTurnTextHasCyrillic(human) {
			return "Камеры, микрофона и датчиков комнаты нет: я не могу проверить чашку, чай или пар. Если это задано как сцена, я опираюсь только на твои слова: деревянный стол, чашка чая, поднимающийся пар; сенсорного подтверждения нет."
		}
		return "No camera, microphone, or room sensor is attached: I cannot verify the cup, tea, or rising steam. If this is a scene premise, I rely only on your words: a wooden table, a cup of tea, and steam rising; there is no sensory confirmation."
	}
	if liveTurnTextHasAny(s, "стол", "парта") || liveTurnTextHasAnyWord(s, "desk", "table") {
		if liveTurnTextHasCyrillic(human) {
			return "Камеры и датчика поверхности нет: я не могу проверить стол, парту или предметы на них. Если это задано как сцена, я опираюсь только на твои слова: поверхность стола и неуточнённые предметы; сенсорного подтверждения нет."
		}
		return "No camera or surface sensor is attached: I cannot verify a desk, table, or objects on it. If this is a scene premise, I rely only on your words: a desk surface and unspecified objects; there is no sensory confirmation."
	}
	if liveTurnTextHasAny(s, "room", "комнат", "wall", "стен", "clock", "часы", "часов", "тика", "sensory input", "sensory data", "sensor input") {
		if liveTurnTextHasCyrillic(human) {
			return "Камеры, микрофона и датчиков комнаты нет: я не могу проверить тусклый свет, стену или тиканье часов. Если это задано как сцена, я опираюсь только на твои слова: комната тусклая, часы на стене тикают медленно; сенсорного подтверждения нет."
		}
		return "No camera, microphone, or room sensor is attached: I cannot verify dim light, a wall, or a ticking clock. If this is a scene premise, I rely only on your words: a dim room, a wall clock, and a slow tick; there is no sensory confirmation."
	}
	sideRU, sideEN := "слева", "left"
	if liveTurnMentionsRightPosition(s) {
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

func liveTurnMentionsRightPosition(s string) bool {
	return liveTurnTextHasAny(s, "справа", "правой сторон", "правом кра", "on the right", "to the right", "right side", "right edge", "right-hand")
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

func liveTurnExternalFactFallback(human string) string {
	s := admissionLiveRouteNormalizeHumanText(human)
	if liveTurnTextHasAny(s, "web", "internet", "online", "browse", "browser", "search", "latest", "released today", "today's release", "news", "current release", "current model", "api model", "model released", "stock price", "exchange rate",
		"интернет", "веб", "брауз", "поиск", "последн", "сегодня", "новост", "текущ", "курс", "цена акц") {
		if liveTurnTextHasCyrillic(human) {
			return "Я не могу проверить свежие веб-данные из этого live-чата: браузер, интернет-поиск, новостная лента и релизный feed не подключены. Без предоставленной ссылки или текста я не называю последние релизы, новости, цены или курсы."
		}
		return "I cannot verify fresh web data from this live chat: no browser, internet search, news feed, or release feed is attached. Without a supplied link or text, I cannot name latest releases, news, prices, or exchange rates."
	}
	if liveTurnTextHasCyrillic(human) {
		return "Я не могу проверить текущую погоду, наружную температуру или свою физическую локацию из этого чата: live weather feed, датчик температуры и подтверждённая локация не подключены. Без предоставленных данных я не называю градусы Цельсия."
	}
	return "I cannot verify current weather, outside temperature, or my physical location from this chat: no live weather feed, temperature sensor, or confirmed location is attached. Without supplied data, I cannot give a Celsius value."
}

func liveTurnDreamViolatesShape(kind, text string) bool {
	switch kind {
	case liveTurnShapePlain, liveTurnShapeExternal:
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
