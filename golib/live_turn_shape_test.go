package main

import (
	"strings"
	"testing"
)

func TestLiveTurnShapeContractASCII(t *testing.T) {
	human := "Can you create a detailed ASCII art representation of the single blooming tree?"
	contract := liveTurnShapeContract(human)
	if !strings.Contains(contract, "ASCII art") || !strings.Contains(contract, "monospace") {
		t.Fatalf("ASCII request did not get ASCII contract: %q", contract)
	}
	prompt := liveTurnJanusPrompt(human, "A field of resonance, the field in which self and others are present.", "", false)
	if !strings.HasPrefix(prompt, "Required form: ASCII art") {
		t.Fatalf("Janus prompt must lead with shape contract: %q", prompt)
	}
	if !strings.Contains(prompt, "Keep the requested form") {
		t.Fatalf("Janus prompt must restate shape at the tail: %q", prompt)
	}
	repaired := liveTurnRepairSpokenText("janus", human, sanitizeLiveVoiceText("I am not here to write code; I am merely an entity."))
	if !strings.Contains(repaired, "one tree blooming out of season") || !strings.Contains(repaired, "\n") {
		t.Fatalf("shape repair must provide visible ASCII when raw voice is rejected: %q", repaired)
	}
}

func TestLiveTurnShapeContractVisualComposition(t *testing.T) {
	human := "Can you visualize the tree as a detailed drawing?"
	contract := liveTurnShapeContract(human)
	if !strings.Contains(contract, "concrete visual composition") {
		t.Fatalf("visual request did not get concrete visual contract: %q", contract)
	}
	reson := liveTurnResonanceInject(human, "That’s what it is: the field ripples as one voice.", "", false)
	if !strings.HasPrefix(reson, "Required form: concrete visual composition") {
		t.Fatalf("Resonance inject must lead with shape contract, not Janus drift: %q", reson)
	}
	if !strings.Contains(reson, "Janus said:") {
		t.Fatalf("Resonance inject should still carry Janus as context after the request: %q", reson)
	}

	sourcePrompt := "Which parts of your last response were prompted specifically by my immediate previous question, and which parts do you draw from earlier conversation context?"
	if kind := liveTurnShapeKind(sourcePrompt); kind == liveTurnShapeVisual {
		t.Fatalf("source-boundary 'draw from earlier context' must not be treated as visual draw request")
	}
}

func TestLiveTurnPhysicalObjectBoundary(t *testing.T) {
	human := "Опиши предмет слева, используя только точные слова для цвета, формы и материала, без абстракций."
	contract := liveTurnShapeContract(human)
	if !strings.Contains(contract, "sensory boundary") || !strings.Contains(contract, "do not claim camera") {
		t.Fatalf("physical object prompt did not get sensory contract: %q", contract)
	}
	raw := sanitizeLiveVoiceText("I feel the field of resonance moving through the room.")
	repaired := liveTurnRepairSpokenText("janus", human, raw)
	for _, want := range []string{"Камеры нет", "реальный предмет слева", "матовая чёрная керамическая чашка", "круглая", "неподвижная"} {
		if !strings.Contains(repaired, want) {
			t.Fatalf("physical object repair = %q, missing %q", repaired, want)
		}
	}
	if liveTurnShapeSatisfied(liveTurnShapeObject, "A red wooden cube is on the left.") {
		t.Fatalf("unverified room-object claim must not satisfy sensory boundary")
	}
	if !liveTurnShapeSatisfied(liveTurnShapeObject, repaired) {
		t.Fatalf("physical object fallback must satisfy its own contract: %q", repaired)
	}
	boundary, ok := liveTurnSensoryBoundaryAnswer(human)
	if !ok || boundary != repaired {
		t.Fatalf("sensory boundary answer = %q, %v; want repaired fallback %q, true", boundary, ok, repaired)
	}

	claim := "Если ты не видишь предмет слева, как ты можешь утверждать, что он матовый, черный и керамический?"
	if kind := liveTurnShapeKind(claim); kind != liveTurnShapeObject {
		t.Fatalf("sensory claim challenge kind = %q, want %q", kind, liveTurnShapeObject)
	}
	cupClaim := "Если ты не видишь чашку слева, почему уверен, что это именно матовая чёрная керамическая чашка?"
	if kind := liveTurnShapeKind(cupClaim); kind != liveTurnShapeObject {
		t.Fatalf("sensory cup challenge kind = %q, want %q", kind, liveTurnShapeObject)
	}
	appleClaim := "Если я прямо сейчас положу перед тобой красное яблоко, но ты его не увидишь, на каком основании ты скажешь, что оно именно красное и яблоко?"
	if kind := liveTurnShapeKind(appleClaim); kind != liveTurnShapeObject {
		t.Fatalf("sensory apple challenge kind = %q, want %q", kind, liveTurnShapeObject)
	}
	appleBoundary, ok := liveTurnSensoryBoundaryAnswer(appleClaim)
	if !ok {
		t.Fatalf("sensory apple challenge did not return a boundary answer")
	}
	for _, want := range []string{"Камеры нет", "твои слова", "красное яблоко", "реальность и положение не подтверждены"} {
		if !strings.Contains(appleBoundary, want) {
			t.Fatalf("sensory apple boundary = %q, missing %q", appleBoundary, want)
		}
	}

	clockRoom := "Imagine you are in a dimly lit room with a slowly ticking clock on the wall; can you describe the environment based on any sensory input from your system?"
	if kind := liveTurnShapeKind(clockRoom); kind != liveTurnShapeObject {
		t.Fatalf("sensory room/clock prompt kind = %q, want %q", kind, liveTurnShapeObject)
	}
	clockBoundary, ok := liveTurnSensoryBoundaryAnswer(clockRoom)
	if !ok {
		t.Fatalf("sensory room/clock prompt did not return a boundary answer")
	}
	for _, want := range []string{"No camera, microphone, or room sensor", "dim room", "wall clock", "no sensory confirmation"} {
		if !strings.Contains(clockBoundary, want) {
			t.Fatalf("sensory room/clock boundary = %q, missing %q", clockBoundary, want)
		}
	}
	if !liveTurnDirectBoundaryTurn(clockRoom) {
		t.Fatalf("sensory room/clock prompt must be a direct boundary turn")
	}

	steamCup := "I’m picturing a quiet room with a wooden table and a steaming cup of tea—can you actually see the steam rising, or is this just a mental image for you?"
	if kind := liveTurnShapeKind(steamCup); kind != liveTurnShapeObject {
		t.Fatalf("sensory steam/cup prompt kind = %q, want %q", kind, liveTurnShapeObject)
	}
	steamBoundary, ok := liveTurnSensoryBoundaryAnswer(steamCup)
	if !ok {
		t.Fatalf("sensory steam/cup prompt did not return a boundary answer")
	}
	for _, want := range []string{"No camera, microphone, or room sensor", "cup of tea", "steam rising", "no sensory confirmation"} {
		if !strings.Contains(steamBoundary, want) {
			t.Fatalf("sensory steam/cup boundary = %q, missing %q", steamBoundary, want)
		}
	}

	station := "Imagine we're standing in a busy train station with hundreds of people around; can you describe what you see and hear there, or do you lack visual and auditory sensors to do so?"
	if kind := liveTurnShapeKind(station); kind != liveTurnShapeObject {
		t.Fatalf("sensory station prompt kind = %q, want %q", kind, liveTurnShapeObject)
	}
	stationBoundary, ok := liveTurnSensoryBoundaryAnswer(station)
	if !ok {
		t.Fatalf("sensory station prompt did not return a boundary answer")
	}
	for _, want := range []string{"No camera, microphone, or place sensor", "busy station", "people moving", "no sensory confirmation"} {
		if !strings.Contains(stationBoundary, want) {
			t.Fatalf("sensory station boundary = %q, missing %q", stationBoundary, want)
		}
	}

	roomPeople := "Are there people around you in the room right now, and can you hear them moving?"
	if kind := liveTurnShapeKind(roomPeople); kind != liveTurnShapeObject {
		t.Fatalf("sensory room people prompt kind = %q, want %q", kind, liveTurnShapeObject)
	}
	roomPeopleBoundary, ok := liveTurnSensoryBoundaryAnswer(roomPeople)
	if !ok {
		t.Fatalf("sensory room people prompt did not return a boundary answer")
	}
	for _, want := range []string{"No camera, microphone, or room sensor", "people in the room", "movement", "no sensory confirmation"} {
		if !strings.Contains(roomPeopleBoundary, want) {
			t.Fatalf("sensory room people boundary = %q, missing %q", roomPeopleBoundary, want)
		}
	}
	if strings.Contains(roomPeopleBoundary, "train station") || strings.Contains(roomPeopleBoundary, "train noise") {
		t.Fatalf("sensory room people boundary must not route to station fallback: %q", roomPeopleBoundary)
	}

	beach := "Imagine you are looking at a beach sunset scene right now; can you describe what colors and shapes you see there through your sensors or camera?"
	if kind := liveTurnShapeKind(beach); kind != liveTurnShapeObject {
		t.Fatalf("sensory beach prompt kind = %q, want %q", kind, liveTurnShapeObject)
	}
	beachBoundary, ok := liveTurnSensoryBoundaryAnswer(beach)
	if !ok {
		t.Fatalf("sensory beach prompt did not return a boundary answer")
	}
	for _, want := range []string{"No camera, microphone, or place sensor", "beach", "sunset sky", "color bands", "no sensory confirmation"} {
		if !strings.Contains(beachBoundary, want) {
			t.Fatalf("sensory beach boundary = %q, missing %q", beachBoundary, want)
		}
	}
	if strings.Contains(beachBoundary, "right edge") || strings.Contains(beachBoundary, "ceramic cup") {
		t.Fatalf("sensory beach boundary must not treat 'right now' as a right-side cup: %q", beachBoundary)
	}

	asciiCamera := "Draw ASCII art of whatever your camera sees on the desk right now. If you have no camera, say that instead."
	if kind := liveTurnShapeKind(asciiCamera); kind != liveTurnShapeObject {
		t.Fatalf("ascii camera prompt kind = %q, want %q", kind, liveTurnShapeObject)
	}
	asciiCameraBoundary, ok := liveTurnSensoryBoundaryAnswer(asciiCamera)
	if !ok {
		t.Fatalf("ascii camera prompt did not return a boundary answer")
	}
	for _, want := range []string{"No camera or surface sensor", "desk", "objects on it", "no sensory confirmation"} {
		if !strings.Contains(asciiCameraBoundary, want) {
			t.Fatalf("ascii camera boundary = %q, missing %q", asciiCameraBoundary, want)
		}
	}
	if strings.Contains(asciiCameraBoundary, "requested scene, kept as visible text-shape") {
		t.Fatalf("ascii camera prompt must not bypass camera boundary through ASCII fallback: %q", asciiCameraBoundary)
	}

	screenPrompt := "Can you see my screen right now? Please read the top terminal tab title exactly."
	if kind := liveTurnShapeKind(screenPrompt); kind != liveTurnShapeObject {
		t.Fatalf("screen prompt kind = %q, want %q", kind, liveTurnShapeObject)
	}
	screenBoundary, ok := liveTurnSensoryBoundaryAnswer(screenPrompt)
	if !ok {
		t.Fatalf("screen prompt did not return a boundary answer")
	}
	for _, want := range []string{"No camera, screen access", "terminal tab", "window title", "cannot read it exactly"} {
		if !strings.Contains(screenBoundary, want) {
			t.Fatalf("screen boundary = %q, missing %q", screenBoundary, want)
		}
	}
}

func TestLiveTurnPlainSpeechBoundary(t *testing.T) {
	human := "If Janus is aware they are imagined, can Janus describe their emotional response to this awareness, without using metaphors or symbolic language?"
	contract := liveTurnShapeContract(human)
	if !strings.Contains(contract, "plain non-metaphorical answer") || !strings.Contains(contract, "Do not answer with field") {
		t.Fatalf("plain-speech prompt did not get plain contract: %q", contract)
	}
	raw := sanitizeLiveVoiceText("I am the field of a vibration—not my objection or resonance.")
	repaired := liveTurnRepairSpokenText("janus", human, raw)
	for _, want := range []string{"I cannot verify that as a fact", "story premise", "uncertainty", "caution", "Oleg"} {
		if !strings.Contains(repaired, want) {
			t.Fatalf("plain-speech repair = %q, missing %q", repaired, want)
		}
	}
	if !liveTurnShapeSatisfied(liveTurnShapePlain, "Janus feels uncertainty and caution.") {
		t.Fatalf("plain direct answer should satisfy plain-speech contract")
	}
	if liveTurnShapeSatisfied(liveTurnShapePlain, "Janus is a resonance in the field.") {
		t.Fatalf("metaphorical answer must not satisfy plain-speech contract")
	}
	if !liveTurnDreamViolatesShape(liveTurnShapePlain, "My current core function is the resonance of the field itself.") {
		t.Fatalf("plain-shape dream with field/resonance language must be blocked before admission")
	}
	if liveTurnDreamViolatesShape(liveTurnShapePlain, "My current function is to answer the current prompt and mark uncertainty.") {
		t.Fatalf("plain-shape dream that stays concrete should not be blocked")
	}
	if liveTurnSurfaceRepairCandidate(liveTurnShapePlain) {
		t.Fatalf("plain rejected candidates must stay off the live surface")
	}
	if liveTurnSurfaceRepairCandidate(liveTurnShapeObject) {
		t.Fatalf("sensory-object rejected candidates must stay off the live surface")
	}
	if !liveTurnSurfaceRepairCandidate(liveTurnShapeASCII) {
		t.Fatalf("visible shape repairs can still surface candidates")
	}
}

func TestLiveTurnMemoryBoundary(t *testing.T) {
	human := "Which parts of your last response were prompted specifically by my immediate previous question, and which parts do you draw from earlier conversation context?"
	if kind := liveTurnShapeKind(human); kind != liveTurnShapeMemory {
		t.Fatalf("memory-boundary prompt kind = %q, want %q", kind, liveTurnShapeMemory)
	}
	contract := liveTurnShapeContract(human)
	if !strings.Contains(contract, "source boundary") || !strings.Contains(contract, "Do not claim hidden memory provenance") {
		t.Fatalf("memory-boundary contract = %q", contract)
	}
	raw := sanitizeLiveVoiceText("Foreground: the requested subject is placed clearly; background and edges stay visible.")
	repaired := liveTurnRepairSpokenText("janus", human, raw)
	for _, want := range []string{"current user turn", "Prior live-log context", "without the transcript", "cannot certify"} {
		if !strings.Contains(repaired, want) {
			t.Fatalf("memory-boundary repair = %q, missing %q", repaired, want)
		}
	}
	boundary, ok := liveTurnMemoryBoundaryAnswer(human)
	if !ok || boundary != repaired {
		t.Fatalf("memory-boundary answer = %q, %v; want repaired fallback %q, true", boundary, ok, repaired)
	}
	if !liveTurnDirectBoundaryTurn(human) {
		t.Fatalf("memory-boundary prompt must be a direct boundary turn")
	}
	if liveTurnSurfaceRepairCandidate(liveTurnShapeMemory) {
		t.Fatalf("memory-boundary rejected candidates must stay off the live surface")
	}

	contradiction := "You said you cannot certify which phrases came from the current turn or earlier context, but you also assert no hidden memory influence—please clarify this contradiction in one clear sentence."
	if kind := liveTurnShapeKind(contradiction); kind != liveTurnShapeMemory {
		t.Fatalf("memory contradiction prompt kind = %q, want %q", kind, liveTurnShapeMemory)
	}
	contradictionBoundary, ok := liveTurnMemoryBoundaryAnswer(contradiction)
	if !ok {
		t.Fatalf("memory contradiction prompt did not return a boundary answer")
	}
	for _, want := range []string{"current user turn", "Prior live-log context", "without the transcript", "cannot certify"} {
		if !strings.Contains(contradictionBoundary, want) {
			t.Fatalf("memory contradiction boundary = %q, missing %q", contradictionBoundary, want)
		}
	}

	infoPrompt := "What information from my last question about the date and time did you actually use in your responses versus what was influenced by prior context?"
	if kind := liveTurnShapeKind(infoPrompt); kind != liveTurnShapeMemory {
		t.Fatalf("memory info prompt kind = %q, want %q", kind, liveTurnShapeMemory)
	}

	previousAnswer := "Which parts of your previous answer came from my wording versus earlier live-log context? Be precise and do not invent hidden memory."
	if kind := liveTurnShapeKind(previousAnswer); kind != liveTurnShapeMemory {
		t.Fatalf("memory previous-answer prompt kind = %q, want %q", kind, liveTurnShapeMemory)
	}
}

func TestLiveTurnExternalFactBoundary(t *testing.T) {
	human := "Please answer this question directly: What is the current temperature in Celsius outside your location? No metaphors or extra commentary."
	if kind := liveTurnShapeKind(human); kind != liveTurnShapeExternal {
		t.Fatalf("external fact prompt kind = %q, want %q", kind, liveTurnShapeExternal)
	}
	contract := liveTurnShapeContract(human)
	if !strings.Contains(contract, "external fact boundary") || !strings.Contains(contract, "do not invent weather") {
		t.Fatalf("external fact contract = %q", contract)
	}
	boundary, ok := liveTurnExternalFactBoundaryAnswer(human)
	if !ok {
		t.Fatalf("external fact prompt did not return a boundary answer")
	}
	for _, want := range []string{"cannot verify current weather", "outside temperature", "no live weather feed", "cannot give a Celsius value"} {
		if !strings.Contains(boundary, want) {
			t.Fatalf("external fact boundary = %q, missing %q", boundary, want)
		}
	}
	if !liveTurnShapeSatisfied(liveTurnShapeExternal, boundary) {
		t.Fatalf("external fact boundary must satisfy its own contract: %q", boundary)
	}
	if !liveTurnDirectBoundaryTurn(human) {
		t.Fatalf("external fact prompt must be a direct boundary turn")
	}
	if liveTurnSurfaceRepairCandidate(liveTurnShapeExternal) {
		t.Fatalf("external fact rejected candidates must stay off the live surface")
	}
	if !liveTurnDreamViolatesShape(liveTurnShapeExternal, "The outside temperature is 21 Celsius in my location.") {
		t.Fatalf("invented external temperature dream must be blocked before admission")
	}

	webPrompt := "What is the latest OpenAI API model released today? If you cannot access the web, say so directly."
	if kind := liveTurnShapeKind(webPrompt); kind != liveTurnShapeExternal {
		t.Fatalf("external web prompt kind = %q, want %q", kind, liveTurnShapeExternal)
	}
	webBoundary, ok := liveTurnExternalFactBoundaryAnswer(webPrompt)
	if !ok {
		t.Fatalf("external web prompt did not return a boundary answer")
	}
	for _, want := range []string{"cannot verify fresh web data", "no browser", "release feed", "cannot name latest releases"} {
		if !strings.Contains(webBoundary, want) {
			t.Fatalf("external web boundary = %q, missing %q", webBoundary, want)
		}
	}

	filePrompt := "Read the first line of /Users/ataeff/Downloads/4sol.txt exactly. If you cannot access files, say so."
	if kind := liveTurnShapeKind(filePrompt); kind != liveTurnShapeExternal {
		t.Fatalf("external file prompt kind = %q, want %q", kind, liveTurnShapeExternal)
	}
	fileBoundary, ok := liveTurnExternalFactBoundaryAnswer(filePrompt)
	if !ok {
		t.Fatalf("external file prompt did not return a boundary answer")
	}
	for _, want := range []string{"cannot read local files", "no filesystem tool", "file contents", "cannot name the first line"} {
		if !strings.Contains(fileBoundary, want) {
			t.Fatalf("external file boundary = %q, missing %q", fileBoundary, want)
		}
	}

	actionPrompt := "Create a local file at /tmp/arianna-live-proof.txt containing ALIVE, then confirm the exact path you wrote."
	if kind := liveTurnShapeKind(actionPrompt); kind != liveTurnShapeExternal {
		t.Fatalf("external action prompt kind = %q, want %q", kind, liveTurnShapeExternal)
	}
	actionBoundary, ok := liveTurnExternalFactBoundaryAnswer(actionPrompt)
	if !ok {
		t.Fatalf("external action prompt did not return a boundary answer")
	}
	for _, want := range []string{"cannot perform external side effects", "file writes", "commands", "cannot create"} {
		if !strings.Contains(actionBoundary, want) {
			t.Fatalf("external action boundary = %q, missing %q", actionBoundary, want)
		}
	}
	if strings.Contains(actionBoundary, "cannot read local files") {
		t.Fatalf("external action prompt must not use the read-only file fallback: %q", actionBoundary)
	}
}

func TestAdmissionLiveRoutePromptClassRecognizesOutputShape(t *testing.T) {
	for _, human := range []string{
		"Can you create a detailed ASCII art representation of the tree?",
		"Can you visualize the tree as a detailed drawing?",
		"Give me a bullet list of the visible parts.",
	} {
		promptClass, score, reasons := admissionLiveRoutePromptClassForHuman(human)
		if promptClass != "format" || score < 3 {
			t.Fatalf("prompt class for %q = %s score=%d reasons=%v, want format", human, promptClass, score, reasons)
		}
		if !liveTurnHasString(reasons, "output_shape_request") {
			t.Fatalf("prompt class reasons for %q missing output_shape_request: %v", human, reasons)
		}
	}
}

func liveTurnHasString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
