package main

import "strings"

// wantsLiveRuntimeFact gates the special telemetry surface in chat. It must stay
// explicit: a normal live question that mentions metrics, debt, status, or facts
// is still a conversation turn and must not be silently rerouted away from
// ProcessText/nano into telemetry labels.
func wantsLiveRuntimeFact(human string) bool {
	lower := strings.ToLower(strings.TrimSpace(human))
	if lower == "" {
		return false
	}
	for _, command := range []string{
		"/status",
		"/runtime",
		"/runtime-fact",
		"/live-status",
		"/live-fact",
	} {
		if lower == command || strings.HasPrefix(lower, command+" ") {
			return true
		}
	}
	return false
}
