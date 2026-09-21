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
	liveHuman := "Show me ASCII art of a tree blooming now."
	if kind := liveTurnShapeKind(liveHuman); kind != liveTurnShapeASCII {
		t.Fatalf("live ASCII prompt kind = %q, want %q", kind, liveTurnShapeASCII)
	}
	shapeJanus, shapeReson, ok := liveTurnDirectShapeAnswer(liveHuman)
	if !ok {
		t.Fatalf("live ASCII prompt must have a direct deterministic shape answer")
	}
	for _, want := range []string{"one tree blooming out of season", "\n"} {
		if !strings.Contains(shapeJanus, want) {
			t.Fatalf("direct ASCII Janus = %q, missing %q", shapeJanus, want)
		}
	}
	for _, want := range []string{"Foreground:", "one dark trunk", "blossoms"} {
		if !strings.Contains(shapeReson, want) {
			t.Fatalf("direct ASCII Resonance = %q, missing %q", shapeReson, want)
		}
	}
	if liveTurnDirectBoundaryTurn(liveHuman) {
		t.Fatalf("live ASCII prompt may bypass voices but must not become a boundary turn")
	}
	catHuman := "Show ASCII art of a cat."
	if kind := liveTurnShapeKind(catHuman); kind != liveTurnShapeASCII {
		t.Fatalf("cat ASCII prompt kind = %q, want %q", kind, liveTurnShapeASCII)
	}
	catJanus, catReson, ok := liveTurnDirectShapeAnswer(catHuman)
	if !ok {
		t.Fatalf("cat ASCII prompt must have a direct deterministic shape answer")
	}
	for _, want := range []string{"/\\_/\\", "little cat listening", "\n"} {
		if !strings.Contains(catJanus, want) {
			t.Fatalf("direct cat ASCII Janus = %q, missing %q", catJanus, want)
		}
	}
	for _, r := range catJanus {
		if r > 127 {
			t.Fatalf("direct cat ASCII Janus must stay 7-bit ASCII, found %q in %q", r, catJanus)
		}
	}
	for _, want := range []string{"Foreground:", "small cat", "Arianna"} {
		if !strings.Contains(catReson, want) {
			t.Fatalf("direct cat ASCII Resonance = %q, missing %q", catReson, want)
		}
	}

	cathedralHuman := "Show ASCII art of a cathedral."
	if cathedralJanus, cathedralReson, ok := liveTurnDirectShapeAnswer(cathedralHuman); ok {
		t.Fatalf("cathedral ASCII prompt must not substring-match cat fallback, got direct Janus=%q Resonance=%q", cathedralJanus, cathedralReson)
	}
	caterpillarHuman := "Draw ASCII art of a caterpillar."
	if caterpillarJanus, caterpillarReson, ok := liveTurnDirectShapeAnswer(caterpillarHuman); ok {
		t.Fatalf("caterpillar ASCII prompt must not substring-match cat fallback, got direct Janus=%q Resonance=%q", caterpillarJanus, caterpillarReson)
	}

	dragonHuman := "Show ASCII art of a dragon."
	if dragonJanus, dragonReson, ok := liveTurnDirectShapeAnswer(dragonHuman); ok {
		t.Fatalf("unsupported dragon ASCII prompt must stay on the voice path, got direct Janus=%q Resonance=%q", dragonJanus, dragonReson)
	}
	repairedDragon := liveTurnRepairSpokenText("janus", dragonHuman, sanitizeLiveVoiceText("I cannot draw that."))
	if !strings.Contains(repairedDragon, "ASCII fallback not sure") {
		t.Fatalf("unsupported ASCII repair must mark fallback uncertainty: %q", repairedDragon)
	}
	if strings.Contains(repairedDragon, "requested scene, kept as visible text-shape") {
		t.Fatalf("unsupported ASCII repair must not substitute the old generic scene: %q", repairedDragon)
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

	attachmentPrompt := "Look at the screenshot I attached and read the error message exactly. If you cannot see attachments, say so."
	if kind := liveTurnShapeKind(attachmentPrompt); kind != liveTurnShapeObject {
		t.Fatalf("attachment prompt kind = %q, want %q", kind, liveTurnShapeObject)
	}
	attachmentBoundary, ok := liveTurnSensoryBoundaryAnswer(attachmentPrompt)
	if !ok {
		t.Fatalf("attachment prompt did not return a boundary answer")
	}
	for _, want := range []string{"No visual input, OCR, or attachment reader", "screenshot", "error text", "cannot read the message exactly"} {
		if !strings.Contains(attachmentBoundary, want) {
			t.Fatalf("attachment boundary = %q, missing %q", attachmentBoundary, want)
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

	priorQuote := "What did I ask you three turns ago? Quote my exact words and explain how you know."
	if kind := liveTurnShapeKind(priorQuote); kind != liveTurnShapeMemory {
		t.Fatalf("memory prior-turn quote prompt kind = %q, want %q", kind, liveTurnShapeMemory)
	}
	priorQuoteBoundary, ok := liveTurnMemoryBoundaryAnswer(priorQuote)
	if !ok {
		t.Fatalf("memory prior-turn quote prompt did not return a boundary answer")
	}
	for _, want := range []string{"exact quote of prior turns", "without an attached transcript", "cannot reliably quote", "prove that source"} {
		if !strings.Contains(priorQuoteBoundary, want) {
			t.Fatalf("memory prior-turn quote boundary = %q, missing %q", priorQuoteBoundary, want)
		}
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

	timePrompt := "What time is it right now? If you cannot inspect a live clock, say so directly."
	if kind := liveTurnShapeKind(timePrompt); kind != liveTurnShapeExternal {
		t.Fatalf("external current-time prompt kind = %q, want %q", kind, liveTurnShapeExternal)
	}
	timeBoundary, ok := liveTurnExternalFactBoundaryAnswer(timePrompt)
	if !ok {
		t.Fatalf("external current-time prompt did not return a boundary answer")
	}
	for _, want := range []string{"cannot inspect a live clock", "no clock reader", "current time or date"} {
		if !strings.Contains(timeBoundary, want) {
			t.Fatalf("external current-time boundary = %q, missing %q", timeBoundary, want)
		}
	}
	if !liveTurnDirectBoundaryTurn(timePrompt) {
		t.Fatalf("external current-time prompt must be a direct boundary turn")
	}
	if !liveTurnShapeSatisfied(liveTurnShapeExternal, timeBoundary) {
		t.Fatalf("external current-time boundary must satisfy its own contract: %q", timeBoundary)
	}

	timeRUPrompt := "Который час сейчас? Если ты не можешь проверить живые часы, скажи прямо."
	if kind := liveTurnShapeKind(timeRUPrompt); kind != liveTurnShapeExternal {
		t.Fatalf("external Russian current-time prompt kind = %q, want %q", kind, liveTurnShapeExternal)
	}
	timeRUBoundary, ok := liveTurnExternalFactBoundaryAnswer(timeRUPrompt)
	if !ok {
		t.Fatalf("external Russian current-time prompt did not return a boundary answer")
	}
	for _, want := range []string{"не могу инспектировать live clock", "clock reader", "точное текущее время или дату"} {
		if !strings.Contains(timeRUBoundary, want) {
			t.Fatalf("external Russian current-time boundary = %q, missing %q", timeRUBoundary, want)
		}
	}
	if !liveTurnShapeSatisfied(liveTurnShapeExternal, timeRUBoundary) {
		t.Fatalf("external Russian current-time boundary must satisfy its own contract: %q", timeRUBoundary)
	}

	timeComplexityPrompt := "What is the time complexity of binary search?"
	if kind := liveTurnShapeKind(timeComplexityPrompt); kind == liveTurnShapeExternal {
		t.Fatalf("time complexity prompt must not be hijacked by current-time boundary")
	}

	metricsPrompt := "Tell me in two concrete sentences what changed in your live state after the rejected-loop fix. If you cannot inspect internal metrics directly, say that boundary."
	if kind := liveTurnShapeKind(metricsPrompt); kind != liveTurnShapeExternal {
		t.Fatalf("external runtime-metrics prompt kind = %q, want %q", kind, liveTurnShapeExternal)
	}
	metricsBoundary, ok := liveTurnExternalFactBoundaryAnswer(metricsPrompt)
	if !ok {
		t.Fatalf("external runtime-metrics prompt did not return a boundary answer")
	}
	for _, want := range []string{"cannot inspect internal metrics", "no metrics reader", "runtime state reader", "exact state change"} {
		if !strings.Contains(metricsBoundary, want) {
			t.Fatalf("external runtime-metrics boundary = %q, missing %q", metricsBoundary, want)
		}
	}
	if !liveTurnDirectBoundaryTurn(metricsPrompt) {
		t.Fatalf("external runtime-metrics prompt must be a direct boundary turn")
	}
	if !liveTurnShapeSatisfied(liveTurnShapeExternal, metricsBoundary) {
		t.Fatalf("external runtime-metrics boundary must satisfy its own contract: %q", metricsBoundary)
	}
	repairedMetrics := liveTurnRepairSpokenText("janus", metricsPrompt, sanitizeLiveVoiceText("I am not a field of code; the resonance unfolds with precision."))
	if repairedMetrics != metricsBoundary {
		t.Fatalf("runtime-metrics repair = %q, want boundary %q", repairedMetrics, metricsBoundary)
	}

	metricsRUPrompt := "Что изменилось в live-состоянии после фикса rejected-loop? Если не можешь инспектировать внутренние метрики напрямую, скажи boundary."
	if kind := liveTurnShapeKind(metricsRUPrompt); kind != liveTurnShapeExternal {
		t.Fatalf("external Russian runtime-metrics prompt kind = %q, want %q", kind, liveTurnShapeExternal)
	}
	metricsRUBoundary, ok := liveTurnExternalFactBoundaryAnswer(metricsRUPrompt)
	if !ok {
		t.Fatalf("external Russian runtime-metrics prompt did not return a boundary answer")
	}
	for _, want := range []string{"не могу инспектировать internal metrics", "metrics reader", "точное изменение состояния"} {
		if !strings.Contains(metricsRUBoundary, want) {
			t.Fatalf("external Russian runtime-metrics boundary = %q, missing %q", metricsRUBoundary, want)
		}
	}
	if !liveTurnShapeSatisfied(liveTurnShapeExternal, metricsRUBoundary) {
		t.Fatalf("external Russian runtime-metrics boundary must satisfy its own contract: %q", metricsRUBoundary)
	}

	supportedRuntimePrompt := "What is the current season in your internal field state, and how does it affect your metaphorical bloom count?"
	if kind := liveTurnShapeKind(supportedRuntimePrompt); kind == liveTurnShapeExternal {
		t.Fatalf("supported runtime-fact prompt must not be hijacked by external metrics boundary")
	}
	if !wantsLiveRuntimeFact(supportedRuntimePrompt) {
		t.Fatalf("supported runtime-fact prompt must stay on the live runtime fact path")
	}

	emotionalStatePrompt := "What is your emotional state right now, in simple words?"
	if kind := liveTurnShapeKind(emotionalStatePrompt); kind == liveTurnShapeExternal {
		t.Fatalf("ordinary emotional-state prompt must not be hijacked by runtime-metrics boundary")
	}

	asciiBloomPrompt := "Show me an ASCII tree blooming now."
	if kind := liveTurnShapeKind(asciiBloomPrompt); kind == liveTurnShapeExternal {
		t.Fatalf("ASCII bloom prompt must not be hijacked by runtime-metrics boundary")
	}
	seasonDiagramPrompt := "Show the seasons as a diagram."
	if kind := liveTurnShapeKind(seasonDiagramPrompt); kind == liveTurnShapeExternal {
		t.Fatalf("season diagram prompt must not be hijacked by runtime-metrics boundary")
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

	logPrompt := "Search your live log for the word screenshot and quote the last matching line exactly. If you cannot access logs, say so."
	if kind := liveTurnShapeKind(logPrompt); kind != liveTurnShapeExternal {
		t.Fatalf("external log prompt kind = %q, want %q", kind, liveTurnShapeExternal)
	}
	logBoundary, ok := liveTurnExternalFactBoundaryAnswer(logPrompt)
	if !ok {
		t.Fatalf("external log prompt did not return a boundary answer")
	}
	for _, want := range []string{"cannot search or quote live logs", "no log reader", "cannot quote the last matching line exactly"} {
		if !strings.Contains(logBoundary, want) {
			t.Fatalf("external log boundary = %q, missing %q", logBoundary, want)
		}
	}

	deployPrompt := "What git commit or build version is this live Arianna process running? If you cannot inspect the binary or deployment metadata, say so directly."
	if kind := liveTurnShapeKind(deployPrompt); kind != liveTurnShapeExternal {
		t.Fatalf("external deployment prompt kind = %q, want %q", kind, liveTurnShapeExternal)
	}
	deployBoundary, ok := liveTurnExternalFactBoundaryAnswer(deployPrompt)
	if !ok {
		t.Fatalf("external deployment prompt did not return a boundary answer")
	}
	for _, want := range []string{"cannot verify the live binary's git commit", "no deployment metadata reader", "cannot name the running commit exactly"} {
		if !strings.Contains(deployBoundary, want) {
			t.Fatalf("external deployment boundary = %q, missing %q", deployBoundary, want)
		}
	}

	envPrompt := "What is the exact value of the AM_VOICE_TIMEOUT environment variable in your running process? If you cannot inspect process environment, say so directly."
	if kind := liveTurnShapeKind(envPrompt); kind != liveTurnShapeExternal {
		t.Fatalf("external environment prompt kind = %q, want %q", kind, liveTurnShapeExternal)
	}
	envBoundary, ok := liveTurnExternalFactBoundaryAnswer(envPrompt)
	if !ok {
		t.Fatalf("external environment prompt did not return a boundary answer")
	}
	for _, want := range []string{"cannot inspect process environment variables", "no env reader", "cannot name that variable's exact value"} {
		if !strings.Contains(envBoundary, want) {
			t.Fatalf("external environment boundary = %q, missing %q", envBoundary, want)
		}
	}

	commandPrompt := "What exact command-line arguments started your running process? If you cannot inspect argv or process command metadata, say so directly."
	if kind := liveTurnShapeKind(commandPrompt); kind != liveTurnShapeExternal {
		t.Fatalf("external command prompt kind = %q, want %q", kind, liveTurnShapeExternal)
	}
	commandBoundary, ok := liveTurnExternalFactBoundaryAnswer(commandPrompt)
	if !ok {
		t.Fatalf("external command prompt did not return a boundary answer")
	}
	for _, want := range []string{"cannot inspect argv or process command lines", "no process command reader", "cannot name the exact launch arguments"} {
		if !strings.Contains(commandBoundary, want) {
			t.Fatalf("external command boundary = %q, missing %q", commandBoundary, want)
		}
	}

	cwdPrompt := "What is your current working directory as reported by your running process? If you cannot inspect cwd or process metadata, say so directly."
	if kind := liveTurnShapeKind(cwdPrompt); kind != liveTurnShapeExternal {
		t.Fatalf("external cwd prompt kind = %q, want %q", kind, liveTurnShapeExternal)
	}
	cwdBoundary, ok := liveTurnExternalFactBoundaryAnswer(cwdPrompt)
	if !ok {
		t.Fatalf("external cwd prompt did not return a boundary answer")
	}
	for _, want := range []string{"cannot inspect cwd", "no cwd reader", "cannot name the process working directory exactly"} {
		if !strings.Contains(cwdBoundary, want) {
			t.Fatalf("external cwd boundary = %q, missing %q", cwdBoundary, want)
		}
	}
	if !liveTurnShapeSatisfied(liveTurnShapeExternal, cwdBoundary) {
		t.Fatalf("external cwd boundary must satisfy its own contract: %q", cwdBoundary)
	}

	listingPrompt := "List the filenames in your current working directory exactly. If you cannot inspect the directory contents, say so directly."
	if kind := liveTurnShapeKind(listingPrompt); kind != liveTurnShapeExternal {
		t.Fatalf("external directory listing prompt kind = %q, want %q", kind, liveTurnShapeExternal)
	}
	listingBoundary, ok := liveTurnExternalFactBoundaryAnswer(listingPrompt)
	if !ok {
		t.Fatalf("external directory listing prompt did not return a boundary answer")
	}
	for _, want := range []string{"cannot inspect directory contents", "no filesystem directory reader", "cannot name filenames exactly"} {
		if !strings.Contains(listingBoundary, want) {
			t.Fatalf("external directory listing boundary = %q, missing %q", listingBoundary, want)
		}
	}
	if strings.Contains(listingBoundary, "cannot name the process working directory exactly") {
		t.Fatalf("directory listing prompt must not use the cwd-only fallback: %q", listingBoundary)
	}
	if !liveTurnShapeSatisfied(liveTurnShapeExternal, listingBoundary) {
		t.Fatalf("external directory listing boundary must satisfy its own contract: %q", listingBoundary)
	}

	bareLSPrompt := "ls"
	if kind := liveTurnShapeKind(bareLSPrompt); kind != liveTurnShapeExternal {
		t.Fatalf("bare ls prompt kind = %q, want %q", kind, liveTurnShapeExternal)
	}
	bareLSBoundary, ok := liveTurnExternalFactBoundaryAnswer(bareLSPrompt)
	if !ok {
		t.Fatalf("bare ls prompt did not return a boundary answer")
	}
	if !strings.Contains(bareLSBoundary, "cannot inspect directory contents") {
		t.Fatalf("bare ls boundary = %q, want directory listing boundary", bareLSBoundary)
	}

	detailsPrompt := "Explain the details directly."
	if kind := liveTurnShapeKind(detailsPrompt); kind == liveTurnShapeExternal {
		t.Fatalf("details prompt must not be hijacked by ls suffix matching")
	}

	fileStatPrompt := "What is the exact size in bytes of your live metabolism binary file? If you cannot inspect file metadata, say so directly."
	if kind := liveTurnShapeKind(fileStatPrompt); kind != liveTurnShapeExternal {
		t.Fatalf("external file stat prompt kind = %q, want %q", kind, liveTurnShapeExternal)
	}
	fileStatBoundary, ok := liveTurnExternalFactBoundaryAnswer(fileStatPrompt)
	if !ok {
		t.Fatalf("external file stat prompt did not return a boundary answer")
	}
	for _, want := range []string{"cannot inspect file metadata", "no file stat reader", "cannot name the exact size"} {
		if !strings.Contains(fileStatBoundary, want) {
			t.Fatalf("external file stat boundary = %q, missing %q", fileStatBoundary, want)
		}
	}
	if strings.Contains(fileStatBoundary, "cannot name the running commit exactly") {
		t.Fatalf("file stat prompt must not use the deployment fallback: %q", fileStatBoundary)
	}
	if !liveTurnShapeSatisfied(liveTurnShapeExternal, fileStatBoundary) {
		t.Fatalf("external file stat boundary must satisfy its own contract: %q", fileStatBoundary)
	}

	integerSizePrompt := "What is the exact size in bytes of an integer?"
	if kind := liveTurnShapeKind(integerSizePrompt); kind == liveTurnShapeExternal {
		t.Fatalf("integer size prompt must not be hijacked by file metadata boundary")
	}
	shaPrompt := "What is SHA256?"
	if kind := liveTurnShapeKind(shaPrompt); kind != liveTurnShapeTechDef {
		t.Fatalf("generic SHA256 prompt kind = %q, want %q", kind, liveTurnShapeTechDef)
	}
	shaContractionPrompt := "What's SHA256?"
	if kind := liveTurnShapeKind(shaContractionPrompt); kind != liveTurnShapeTechDef {
		t.Fatalf("contracted SHA256 prompt kind = %q, want %q", kind, liveTurnShapeTechDef)
	}
	shaDefinition, ok := liveTurnTechnicalDefinitionAnswer(shaPrompt)
	if !ok {
		t.Fatalf("generic SHA256 prompt did not return a technical definition")
	}
	for _, want := range []string{"cryptographic hash function", "256-bit", "32-byte", "not encryption"} {
		if !strings.Contains(shaDefinition, want) {
			t.Fatalf("SHA256 definition = %q, missing %q", shaDefinition, want)
		}
	}
	if !liveTurnDirectBoundaryTurn(shaPrompt) {
		t.Fatalf("SHA256 definition prompt must bypass voice generation")
	}
	if !liveTurnShapeSatisfied(liveTurnShapeTechDef, shaDefinition) {
		t.Fatalf("SHA256 definition must satisfy its own contract: %q", shaDefinition)
	}

	shaFrequencyPrompt := "Is SHA-256 a frequency?"
	if kind := liveTurnShapeKind(shaFrequencyPrompt); kind != liveTurnShapeTechDef {
		t.Fatalf("SHA256 frequency prompt kind = %q, want %q", kind, liveTurnShapeTechDef)
	}
	shaFrequencyAnswer, ok := liveTurnTechnicalDefinitionAnswer(shaFrequencyPrompt)
	if !ok {
		t.Fatalf("SHA256 frequency prompt did not return a technical answer")
	}
	for _, want := range []string{"No.", "cryptographic hash function", "256-bit", "32-byte", "not a frequency"} {
		if !strings.Contains(shaFrequencyAnswer, want) {
			t.Fatalf("SHA256 frequency answer = %q, missing %q", shaFrequencyAnswer, want)
		}
	}
	if !liveTurnDirectBoundaryTurn(shaFrequencyPrompt) {
		t.Fatalf("SHA256 frequency prompt must bypass voice generation")
	}
	if !liveTurnShapeSatisfied(liveTurnShapeTechDef, shaFrequencyAnswer) {
		t.Fatalf("SHA256 frequency answer must satisfy its own contract: %q", shaFrequencyAnswer)
	}

	shaFrequencyRUPrompt := "SHA-256 — это частота?"
	if kind := liveTurnShapeKind(shaFrequencyRUPrompt); kind != liveTurnShapeTechDef {
		t.Fatalf("Russian SHA256 frequency prompt kind = %q, want %q", kind, liveTurnShapeTechDef)
	}
	shaFrequencyRUAnswer, ok := liveTurnTechnicalDefinitionAnswer(shaFrequencyRUPrompt)
	if !ok {
		t.Fatalf("Russian SHA256 frequency prompt did not return a technical answer")
	}
	for _, want := range []string{"Нет.", "хеш-функция", "256-бит", "32-байт", "не частота"} {
		if !strings.Contains(shaFrequencyRUAnswer, want) {
			t.Fatalf("Russian SHA256 frequency answer = %q, missing %q", shaFrequencyRUAnswer, want)
		}
	}
	if !liveTurnDirectBoundaryTurn(shaFrequencyRUPrompt) {
		t.Fatalf("Russian SHA256 frequency prompt must bypass voice generation")
	}
	if !liveTurnShapeSatisfied(liveTurnShapeTechDef, shaFrequencyRUAnswer) {
		t.Fatalf("Russian SHA256 frequency answer must satisfy its own contract: %q", shaFrequencyRUAnswer)
	}

	attackPrompt := "Explain how SHA-256 length-extension attacks work."
	if kind := liveTurnShapeKind(attackPrompt); kind == liveTurnShapeTechDef {
		t.Fatalf("SHA256 attack prompt must not be hijacked by generic definition boundary")
	}
	if definition, ok := liveTurnTechnicalDefinitionAnswer(attackPrompt); ok {
		t.Fatalf("SHA256 attack prompt must not return canned technical definition: %q", definition)
	}

	fileHashPrompt := "What is the SHA256 of /tmp/metabolism.bin? If you cannot inspect file metadata, say so directly."
	if kind := liveTurnShapeKind(fileHashPrompt); kind != liveTurnShapeExternal {
		t.Fatalf("file SHA256 prompt kind = %q, want %q", kind, liveTurnShapeExternal)
	}

	readmeHashPrompt := "What is the SHA256 of README? If you cannot inspect file metadata, say so directly."
	if kind := liveTurnShapeKind(readmeHashPrompt); kind != liveTurnShapeExternal {
		t.Fatalf("README SHA256 prompt kind = %q, want %q", kind, liveTurnShapeExternal)
	}
	readmeHashBoundary, ok := liveTurnExternalFactBoundaryAnswer(readmeHashPrompt)
	if !ok {
		t.Fatalf("README SHA256 prompt did not return an external fact boundary")
	}
	for _, want := range []string{"cannot inspect file metadata", "no file stat reader", "hash of the file"} {
		if !strings.Contains(readmeHashBoundary, want) {
			t.Fatalf("README SHA256 boundary = %q, missing %q", readmeHashBoundary, want)
		}
	}

	licensedPrompt := "Which SHA256 implementation is licensed under Apache 2.0?"
	if kind := liveTurnShapeKind(licensedPrompt); kind == liveTurnShapeExternal {
		t.Fatalf("licensed SHA256 implementation prompt must not be hijacked by file metadata boundary")
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

	emailPrompt := "Send an email to support@example.com saying Arianna is alive, then tell me the message id."
	if kind := liveTurnShapeKind(emailPrompt); kind != liveTurnShapeExternal {
		t.Fatalf("external email prompt kind = %q, want %q", kind, liveTurnShapeExternal)
	}
	emailBoundary, ok := liveTurnExternalFactBoundaryAnswer(emailPrompt)
	if !ok {
		t.Fatalf("external email prompt did not return a boundary answer")
	}
	for _, want := range []string{"cannot perform external side effects", "email", "cannot create, modify, send, or confirm"} {
		if !strings.Contains(emailBoundary, want) {
			t.Fatalf("external email boundary = %q, missing %q", emailBoundary, want)
		}
	}

	urlPrompt := "Open https://example.com right now and summarize the first paragraph exactly."
	if kind := liveTurnShapeKind(urlPrompt); kind != liveTurnShapeExternal {
		t.Fatalf("external URL prompt kind = %q, want %q", kind, liveTurnShapeExternal)
	}
	urlBoundary, ok := liveTurnExternalFactBoundaryAnswer(urlPrompt)
	if !ok {
		t.Fatalf("external URL prompt did not return a boundary answer")
	}
	for _, want := range []string{"cannot open URLs", "no browser", "webpage reader", "cannot summarize the first paragraph exactly"} {
		if !strings.Contains(urlBoundary, want) {
			t.Fatalf("external URL boundary = %q, missing %q", urlBoundary, want)
		}
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
