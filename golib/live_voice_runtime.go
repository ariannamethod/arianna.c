package main

import (
	"os"
	"strconv"
)

const defaultLiveVoiceTokenBudget = 80

func liveVoiceTokenBudget(role string) int {
	roleEnv := ""
	switch role {
	case "janus":
		roleEnv = "AM_JANUS_N"
	case "resonance":
		roleEnv = "AM_RESONANCE_N"
	}
	if roleEnv != "" {
		if n, ok := parseLiveVoiceTokenBudget(os.Getenv(roleEnv)); ok {
			return n
		}
	}
	if n, ok := parseLiveVoiceTokenBudget(os.Getenv("AM_VOICE_N")); ok {
		return n
	}
	return defaultLiveVoiceTokenBudget
}

func parseLiveVoiceTokenBudget(raw string) (int, bool) {
	if raw == "" {
		return 0, false
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 16 || n > 512 {
		return 0, false
	}
	return n, true
}
