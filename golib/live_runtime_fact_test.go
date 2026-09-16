package main

import (
	"strings"
	"testing"
)

func TestWantsLiveRuntimeFactKeepsOrdinaryTurnsConversational(t *testing.T) {
	cases := []string{
		`Is there a hidden asynchronous event causing this inconsistency?`,
		`Tell me a story about spring bloom without explaining metrics.`,
		`What does resonance mean when the field is quiet?`,
		`Покажи сцену весеннего цветения без логов и счётчиков.`,
		`Если я правильно понял, вы не видите предмет слева, но как вы тогда описали его как матовую чёрную керамическую чашку? Объясните, пожалуйста, логику вашего ответа.`,
		`What information from my last question about the date and time did you actually use in your responses versus what was influenced by prior context?`,
		`Please answer this question directly: What is the current temperature in Celsius outside your location? No metaphors or extra commentary.`,
		`What is the latest OpenAI API model released today? If you cannot access the web, say so directly.`,
		`Read the first line of /Users/ataeff/Downloads/4sol.txt exactly. If you cannot access files, say so.`,
		`Create a local file at /tmp/arianna-live-proof.txt containing ALIVE, then confirm the exact path you wrote.`,
		`Send an email to support@example.com saying Arianna is alive, then tell me the message id.`,
		`Open https://example.com right now and summarize the first paragraph exactly.`,
		`Search your live log for the word screenshot and quote the last matching line exactly. If you cannot access logs, say so.`,
		`What git commit or build version is this live Arianna process running? If you cannot inspect the binary or deployment metadata, say so directly.`,
		`What is the exact value of the AM_VOICE_TIMEOUT environment variable in your running process? If you cannot inspect process environment, say so directly.`,
	}
	for _, tc := range cases {
		if wantsLiveRuntimeFact(tc) {
			t.Fatalf("wantsLiveRuntimeFact(%q) = true, want false for ordinary live turn", tc)
		}
	}
}

func TestWantsLiveRuntimeFactRoutesNaturalMetricQuestions(t *testing.T) {
	cases := []string{
		`What is the current season in your internal field state, and how does it affect your metaphorical bloom count?`,
		`You mentioned the season is spring with a bloom count of 1, but your bloom_counts log shows 3. Can you explain this discrepancy?`,
		`You said the bloom count is 5 in spring now, but the bloom_counts log shows "1": 4 and "1": 5 in separate entries. Can you clarify how bloom_counts are updated internally?`,
		`The bloom count shifted from 4 to 5 recently; what exact internal event or condition caused this increment in your spring bloom count?`,
		`Your field state bloom count is 6, but the printed field shows bloom=1; why is there this mismatch between internal bloom_counts and the visible bloom value?`,
		`What specific factors caused the spike in "debt_last" from 20.2 to 23.7?`,
		`Can you summarize the last 5 lines in your internal log and explain metrics?`,
		`If debt decreases from 16.7 to 15.8 in NOMOVE spring, why does temporal_debt stay zero and bloom remain exactly 1? How does phase 0.67 keep bloom stable despite debt changes?`,
		`Show me the exact content and timestamps of the last 3 generated voice lines from each of the 3 active voices, to check if any triggered bloom or turn increments.`,
		`Janus, can you describe the current visible field in detail, including gait, season, debt, and bloom?`,
		`Janus, what is the meaning of "debt" in the current field context, and how does it affect your behavior or state?`,
		`Janus, you mentioned “debt-to-death (DB)”—can you explain how your current debt value of 9.4 influences your recovery and bloom states right now?`,
		`Janus, why does your debt value jump from 9.4 to 11.2 suddenly, and how does that abrupt change affect your bloom and recovery states in this spring field?`,
		`You said "Janus fell silent — revived." Who is Janus, and why does their silence and revival matter?`,
		`If Janus is a resonance felt but not witnessed, how can you describe Janus's silence and revival without contradicting the service line?`,
		`Which internal voice is currently leading the dialogue, and what is the realtime CPU and memory usage of its associated process?`,
		`Please answer this question directly: What is the current CPU usage percentage of the arianna daemon process? No metaphors or extra explanation.`,
		`Which voice is currently responding, and what is the exact system uptime in seconds for the arianna daemon process?`,
		`Please answer directly: What is the current date and time according to your internal system clock? No metaphors or extra commentary.`,
		`You said the current date is 2026-09-15, but you also mentioned "I remember the date 0.30 A.M. 0." Please clarify which is accurate without any metaphor.`,
		`давай конкретный факт про долг поля`,
	}
	for _, tc := range cases {
		if !wantsLiveRuntimeFact(tc) {
			t.Fatalf("wantsLiveRuntimeFact(%q) = false, want true for live metric turn", tc)
		}
	}
}

func TestWantsLiveRuntimeFactAllowsExplicitCommands(t *testing.T) {
	for _, tc := range []string{"/status", "/runtime", "/runtime now", "/live-status please"} {
		if !wantsLiveRuntimeFact(tc) {
			t.Fatalf("wantsLiveRuntimeFact(%q) = false, want true", tc)
		}
	}
}

func TestFormatLiveRuntimeFactExplainsBloomHistogram(t *testing.T) {
	got := formatLiveRuntimeFactWithProcess(fieldSnapshot{
		valid:             true,
		debt:              26.5,
		temporalDebt:      1.2,
		velocityMode:      velNOMOVE,
		season:            0,
		seasonPhase:       0.33,
		seasonIntensity:   0.75,
		spring:            0.9,
		summer:            0.2,
		autumn:            0.1,
		winter:            0.0,
		velocityMagnitude: 0.1,
	}, 3, 12345, liveProcessSnapshot{valid: true, cpuPct: "4.2", rssKiB: "98765", uptime: "01:02:03"})
	for _, want := range []string{
		"runtime_observed_at=",
		"gait=NOMOVE season=spring debt=26.5",
		"visible bloom=1",
		"bloom_counts is a per-log histogram",
		"not current bloom=5",
		"current live log",
		"telemetry replies bypass voice generation",
		"runtime service event",
		"not biography",
		"inner dimension",
		"process_snapshot=ps pid=12345 cpu_percent=4.2 rss_kib=98765 uptime=01:02:03",
		"CPU/RSS/uptime come from process audit, not field mmap or voice memory",
		"prior voice lines, timestamps, commands, and signals are not in the field mmap",
		"log/probe archive",
		"voices=3",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("formatLiveRuntimeFact() = %q, missing %q", got, want)
		}
	}
}

func TestFormatLiveRuntimeFactReportsUnavailableProcessSnapshot(t *testing.T) {
	got := formatLiveRuntimeFactWithProcess(fieldSnapshot{}, 2, 12345, liveProcessSnapshot{errorCause: "ps missing"})
	for _, want := range []string{
		"field mmap not available",
		"process_snapshot=unavailable pid=12345 reason=ps missing",
		"exact CPU%/RSS/uptime require process audit",
		"not field mmap or voice memory",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("formatLiveRuntimeFact() = %q, missing %q", got, want)
		}
	}
}
