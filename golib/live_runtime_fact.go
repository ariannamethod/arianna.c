package main

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
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
	if liveTurnLooksLikeMemoryBoundaryProbe(lower) {
		return false
	}
	if liveTurnLooksLikeExternalFactProbe(lower) {
		return false
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
	"date and time",
	"current date",
	"current time",
	"system clock",
	"internal system clock",
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

type liveProcessSnapshot struct {
	valid      bool
	cpuPct     string
	rssKiB     string
	uptime     string
	errorCause string
}

func readLiveProcessSnapshot(pid int) liveProcessSnapshot {
	if pid <= 0 {
		return liveProcessSnapshot{errorCause: "invalid pid"}
	}
	out, err := exec.Command("ps", "-o", "%cpu=", "-o", "rss=", "-o", "etime=", "-p", strconv.Itoa(pid)).Output()
	if err != nil {
		return liveProcessSnapshot{errorCause: err.Error()}
	}
	fields := strings.Fields(string(out))
	if len(fields) < 3 {
		return liveProcessSnapshot{errorCause: "ps output missing cpu/rss/etime fields"}
	}
	return liveProcessSnapshot{
		valid:  true,
		cpuPct: fields[0],
		rssKiB: fields[1],
		uptime: fields[2],
	}
}

func formatLiveProcessSnapshot(pid int, ps liveProcessSnapshot) string {
	if ps.valid {
		return fmt.Sprintf("process_snapshot=ps pid=%d cpu_percent=%s rss_kib=%s uptime=%s; CPU/RSS/uptime come from process audit, not field mmap or voice memory", pid, ps.cpuPct, ps.rssKiB, ps.uptime)
	}
	reason := strings.TrimSpace(ps.errorCause)
	if reason == "" {
		reason = "unknown"
	}
	return fmt.Sprintf("process_snapshot=unavailable pid=%d reason=%s; exact CPU%%/RSS/uptime require process audit, not field mmap or voice memory", pid, reason)
}

func formatLiveRuntimeFact(fs fieldSnapshot, voices, pid int) string {
	return formatLiveRuntimeFactWithProcess(fs, voices, pid, readLiveProcessSnapshot(pid))
}

func formatLiveRuntimeFactWithProcess(fs fieldSnapshot, voices, pid int, ps liveProcessSnapshot) string {
	if voices < 0 {
		voices = 0
	}
	process := formatLiveProcessSnapshot(pid, ps)
	if fs.valid {
		_, _, bloom := fs.modulate()
		return fmt.Sprintf(
			"runtime_observed_at=%s; visible field: %s temporal_debt=%.1f phase=%.2f intensity=%.2f energies spring=%.2f summer=%.2f autumn=%.2f winter=%.2f; visible bloom=%d (derived from gait/season/debt modulation); bloom_counts is a per-log histogram of observed printed bloom values, so a key like \"1\":5 means five sampled field lines showed bloom=1, not current bloom=5; metric counters are scoped to the current live log and can reset when arianna-live is hot-swapped into a new log; telemetry replies bypass voice generation, so janus_turns/resonance_turns count generated voice lines, not live-fact echoes; Janus fell silent — revived is a runtime service event: a voice daemon missed the END frame and was respawned; it is not biography, memory, witness metaphysics, or an inner dimension; %s; prior voice lines, timestamps, commands, and signals are not in the field mmap and must come from the log/probe archive or process audit, never from voice memory; voices=%d.",
			time.Now().Format(time.RFC3339),
			fs.describe(),
			fs.temporalDebt,
			fs.seasonPhase,
			fs.seasonIntensity,
			fs.spring,
			fs.summer,
			fs.autumn,
			fs.winter,
			bloom,
			process,
			voices,
		)
	}
	return fmt.Sprintf("runtime_observed_at=%s; field mmap not available; voices=%d; %s; Janus fell silent — revived is a runtime service event, not biography or an inner dimension.", time.Now().Format(time.RFC3339), voices, process)
}
