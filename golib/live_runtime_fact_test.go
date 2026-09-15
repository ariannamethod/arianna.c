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
	got := formatLiveRuntimeFact(fieldSnapshot{
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
	}, 3, 12345)
	for _, want := range []string{
		"gait=NOMOVE season=spring debt=26.5",
		"visible bloom=1",
		"bloom_counts is a histogram",
		"not current bloom=5",
		"voices=3",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("formatLiveRuntimeFact() = %q, missing %q", got, want)
		}
	}
}
