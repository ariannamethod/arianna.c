package main

import (
	"fmt"
	"strings"
)

// wantsLiveRuntimeFact gates the special telemetry surface in chat. It stays
// narrow, but live probing showed that humans and GPT probes ask metric questions
// in ordinary language ("why does bloom_counts disagree with bloom=1?"). Those
// must be answered from the mmap/log semantics, not hallucinated by Janus,
// Resonance, or nano as if the prompt were a normal poetic turn.
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
	return asksNaturalLiveRuntimeFact(lower)
}

func asksNaturalLiveRuntimeFact(lower string) bool {
	return containsAny(lower, liveRuntimeMetricCues) && containsAny(lower, liveRuntimeInquiryCues)
}

var liveRuntimeMetricCues = []string{
	"bloom_counts",
	"bloom count",
	"blossom count",
	"printed field",
	"visible bloom",
	"bloom=",
	"bloom remain",
	"bloom remains",
	"bloom stable",
	"bloom value",
	"bloom parameter",
	"current season",
	"internal field state",
	"field state",
	"field status",
	"field debt",
	"temporal_debt",
	"temporal debt",
	"debt_last",
	"debt_min",
	"debt_max",
	"debt=",
	"debt decreases",
	"debt increases",
	"debt changes",
	"debt remains",
	"debt stay",
	"phase 0.",
	"field_ticks",
	"janus_turns",
	"resonance_turns",
	"nano_turns",
	"live metric",
	"live metrics",
	"metrics log",
	"internal log",
	"live log",
	"telemetry",
	"runtime",
	"gait=",
	"состояние поля",
	"поле сейчас",
	"текущий сезон",
	"текущая телеметрия",
	"метрик",
	"телеметр",
	"лог",
	"счётчик",
	"счетчик",
	"долг поля",
}

var liveRuntimeInquiryCues = []string{
	"?",
	"what is",
	"what exact",
	"why",
	"how does",
	"how are",
	"explain",
	"clarify",
	"discrepancy",
	"mismatch",
	"shifted",
	"caused",
	"cause",
	"condition",
	"increment",
	"updated",
	"shows",
	"showed",
	"current",
	"now",
	"exact",
	"что",
	"почему",
	"как",
	"какой",
	"какая",
	"объясни",
	"поясни",
	"расхожд",
	"несовпад",
	"причин",
	"из-за",
	"сейчас",
	"текущ",
	"точно",
	"факт",
	"давай",
	"дай",
}

func containsAny(text string, patterns []string) bool {
	for _, pattern := range patterns {
		if strings.Contains(text, pattern) {
			return true
		}
	}
	return false
}

func formatLiveRuntimeFact(fs fieldSnapshot, voices, pid int) string {
	if voices < 0 {
		voices = 0
	}
	if fs.valid {
		_, _, bloom := fs.modulate()
		return fmt.Sprintf(
			"visible field: %s temporal_debt=%.1f phase=%.2f intensity=%.2f energies spring=%.2f summer=%.2f autumn=%.2f winter=%.2f; visible bloom=%d (derived from gait/season/debt modulation); bloom_counts is a per-log histogram of observed printed bloom values, so a key like \"1\":5 means five sampled field lines showed bloom=1, not current bloom=5; metric counters are scoped to the current live log and can reset when arianna-live is hot-swapped into a new log; telemetry replies bypass voice generation, so janus_turns/resonance_turns count generated voice lines, not live-fact echoes; restart command/signal cause is not encoded in the field mmap; inspect process/log audit for that; pid=%d voices=%d.",
			fs.describe(),
			fs.temporalDebt,
			fs.seasonPhase,
			fs.seasonIntensity,
			fs.spring,
			fs.summer,
			fs.autumn,
			fs.winter,
			bloom,
			pid,
			voices,
		)
	}
	return fmt.Sprintf("pid=%d; field mmap not available; voices=%d.", pid, voices)
}
