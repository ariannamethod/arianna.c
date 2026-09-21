package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync/atomic"
	"time"
)

func liveMetricAdd(counter *int64, delta int64) int64 {
	return atomic.AddInt64(counter, delta)
}

func liveMetricLoad(counter *int64) int64 {
	return atomic.LoadInt64(counter)
}

func recordLiveMetric(log string, tc *trioCtx, fs fieldSnapshot, extra map[string]any) {
	path := os.Getenv("AM_LIVE_METRICS_LOG")
	if path == "" || tc == nil || tc.iw == nil {
		return
	}
	dir := filepath.Dir(path)
	if dir != "." && dir != "" {
		_ = os.MkdirAll(dir, 0o755)
	}
	s := tc.iw.GetSnapshot()
	obj := map[string]any{
		"iso":             time.Now().Format(time.RFC3339Nano),
		"log":             log,
		"pid":             os.Getpid(),
		"human_turns":     liveMetricLoad(&tc.humanTurns),
		"janus_turns":     liveMetricLoad(&tc.janusTurns),
		"resonance_turns": liveMetricLoad(&tc.resonanceTurns),
		"nano_turns":      liveMetricLoad(&tc.nanoTurns),
		"dreams":          liveMetricLoad(&tc.dreams),
		"chorus_dreams":   liveMetricLoad(&tc.chorusDreams),
		"inner_lines":     liveMetricLoad(&tc.innerLines),
		"field_ticks":     liveMetricLoad(&tc.fieldTicks),
		"arousal":         s.Arousal,
		"valence":         s.Valence,
		"entropy":         s.Entropy,
		"coherence":       s.Coherence,
		"trauma":          s.TraumaLevel,
		"wander":          s.WanderPull,
		"drift_speed":     s.DriftSpeed,
		"memory_pressure": s.MemoryPressure,
		"moved":           tc.lastMoved,
		"viability":       viability(s, tc.janusD != nil && tc.janusD.dead, tc.resonD != nil && tc.resonD.dead),
		"janus_dead":      tc.janusD != nil && tc.janusD.dead,
		"resonance_dead":  tc.resonD != nil && tc.resonD.dead,
		"field_valid":     fs.valid,
	}
	if fs.valid {
		obj["debt_last"] = fs.debt
		obj["debt_min"] = fs.debt
		obj["debt_max"] = fs.debt
		obj["temporal_debt"] = fs.temporalDebt
		obj["gait"] = gaitName[fs.velocityMode]
		obj["velocity_mode"] = fs.velocityMode
		obj["velocity_magnitude"] = fs.velocityMagnitude
		if fs.season >= 0 && int(fs.season) < len(seasonName) {
			obj["season"] = seasonName[fs.season]
		}
		obj["season_phase"] = fs.seasonPhase
		obj["season_intensity"] = fs.seasonIntensity
		obj["spring"] = fs.spring
		obj["summer"] = fs.summer
		obj["autumn"] = fs.autumn
		obj["winter"] = fs.winter
	}
	for k, v := range extra {
		obj[k] = v
	}
	b, err := json.Marshal(obj)
	if err != nil {
		return
	}
	b = append(b, '\n')
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return
	}
	_, _ = f.Write(b)
	_ = f.Close()
}
