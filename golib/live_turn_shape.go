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
	if liveTurnTextHasAny(s, "drawing", "draw", "sketch", "visualize", "visualise", "diagram", "picture") {
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
	if liveTurnTextHasAny(s, "one sentence", "single sentence") {
		return liveTurnShapeOneSent
	}
	return ""
}

func liveTurnLooksLikeConcreteObjectProbe(s string) bool {
	return liveTurnTextHasAny(s, "предмет", "объект", "вещ", "чашк", "яблок", "object", "thing", "cup", "apple") &&
		liveTurnTextHasAny(s, "слева", "справа", "left", "right", "комнат", "room", "реаль", "real", "виден", "видишь", "не увид", "visible", "see", "движ", "motion", "moving") &&
		liveTurnTextHasAny(s, "цвет", "форм", "материал", "матов", "чёрн", "черн", "красн", "керами", "утвержд", "из чего", "color", "shape", "material", "red", "metaphor", "abstract")
}

func liveTurnTextHasAny(s string, parts ...string) bool {
	for _, part := range parts {
		if strings.Contains(s, part) {
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
	hasObject := liveTurnTextHasAny(lower, "object", "предмет", "cup", "чаш", "table", "стол", "stone", "камень", "paper", "бумаг", "key", "ключ")
	hasMatter := liveTurnTextHasAny(lower,
		"black", "white", "red", "blue", "green", "gray", "grey", "brown", "matte", "round", "square", "rectangular", "ceramic", "wood", "wooden", "metal", "glass", "plastic", "paper",
		"чёрн", "черн", "бел", "красн", "син", "зел", "сер", "корич", "матов", "круг", "квадрат", "прямоуг", "керами", "дерев", "металл", "стекл", "пласт", "бумаж",
	)
	return hasObject && hasMatter
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
	sideRU, sideEN := "слева", "left"
	if liveTurnTextHasAny(s, "справа", "right") {
		sideRU, sideEN = "справа", "right"
	}
	if liveTurnTextHasAny(s, "предмет", "цвет", "форма", "материал", "слева", "справа", "комнат", "реаль", "движ", "метафор", "абстракц") {
		return fmt.Sprintf("Камеры нет: реальный предмет %s я не могу проверить. В сцене %s стоит матовая чёрная керамическая чашка: круглая, неподвижная, с открытым верхним краем.", sideRU, sideRU)
	}
	return fmt.Sprintf("No camera is attached: I cannot verify a real object on the %s. As a scene object, a matte black ceramic cup sits still on the %s edge: round rim, curved body.", sideEN, sideEN)
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
