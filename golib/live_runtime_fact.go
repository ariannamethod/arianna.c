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
	"current visible field",
	"visible field",
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
	"current field context",
	"meaning of debt",
	"meaning of \"debt\"",
	"debt value",
	"debt jump",
	"debt value jump",
	"recovery state",
	"recovery states",
	"bloom state",
	"bloom states",
	"spring field",
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
	"active voices",
	"voice lines",
	"voice ids",
	"voice id",
	"generated voice",
	"generated text",
	"fell silent",
	"silent — revived",
	"silent - revived",
	"silence and revival",
	"silence and renewal",
	"voice fell silent",
	"janus fell silent",
	"janus's silence",
	"janus’ silence",
	"janus silence",
	"revived",
	"revival",
	"service line",
	"runtime event",
	"turn increments",
	"turn increment",
	"cpu",
	"cpu usage",
	"memory usage",
	"mem usage",
	"rss",
	"resident memory",
	"process memory",
	"uptime",
	"system uptime",
	"process uptime",
	"elapsed process time",
	"etime",
	"realtime cpu",
	"real-time cpu",
	"associated process",
	"daemon process",
	"arianna daemon",
	"arianna process",
	"process usage",
	"current log",
	"timestamps",
	"timestamp",
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
	"лог ",
	"лог.",
	"лог?",
	"лог,",
	"лог:",
	"лога",
	"логе",
	"логов",
	"логах",
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
			"visible field: %s temporal_debt=%.1f phase=%.2f intensity=%.2f energies spring=%.2f summer=%.2f autumn=%.2f winter=%.2f; visible bloom=%d (derived from gait/season/debt modulation); bloom_counts is a per-log histogram of observed printed bloom values, so a key like \"1\":5 means five sampled field lines showed bloom=1, not current bloom=5; metric counters are scoped to the current live log and can reset when arianna-live is hot-swapped into a new log; telemetry replies bypass voice generation, so janus_turns/resonance_turns count generated voice lines, not live-fact echoes; Janus fell silent — revived is a runtime service event: a voice daemon missed the END frame and was respawned; it is not biography, memory, witness metaphysics, or an inner dimension; exact CPU%%/RSS/uptime and prior voice lines, timestamps, commands, and signals are not in the field mmap and must come from the log/probe archive or process audit, never from voice memory; pid=%d voices=%d.",
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
	return fmt.Sprintf("pid=%d; field mmap not available; voices=%d; exact CPU%%/RSS/uptime must come from an external process snapshot, not voice memory; Janus fell silent — revived is a runtime service event, not biography or an inner dimension.", pid, voices)
}
