package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

func admissionLiveRouteTurnCandidateExecutionRunnerDryRun() bool {
	return dreamAdmissionBoolEnv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_RUNNER_DRY_RUN")
}

func admissionLiveRouteTurnCandidateExecutionRunnerName() string {
	raw := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_RUNNER"))
	if raw == "" {
		return admissionLiveRouteTurnCandidateExecutionRunnerSelfEmit
	}
	return raw
}

func admissionLiveRouteTurnCandidateExecutionForShellViaRunner(shell admissionLiveRouteTurnCandidateShell, text string) admissionLiveRouteTurnCandidateExecution {
	runner := admissionLiveRouteTurnCandidateExecutionRunnerName()
	pendingRuntime := admissionLiveRouteTurnCandidateExecutionRuntime{
		Runner: runner,
		Status: admissionLiveRouteTurnCandidateExecutionStatusSucceeded,
	}
	if execution, ok := admissionLiveRouteTurnCandidateExecutionPreflight(shell, pendingRuntime); !ok {
		return execution
	}
	if runner != admissionLiveRouteTurnCandidateExecutionRunnerSelfEmit {
		return admissionLiveRouteTurnCandidateExecutionForShellWithRuntime(shell, "", admissionLiveRouteTurnCandidateExecutionRuntime{
			Runner:        runner,
			Status:        admissionLiveRouteTurnCandidateExecutionStatusFailed,
			ExitCode:      -1,
			FailureReason: "unknown candidate execution runner " + runner,
		})
	}
	timeoutMS := admissionLiveRouteTurnCandidateExecutionTimeoutMS()
	if timeoutMS <= 0 || timeoutMS > admissionLiveRouteTurnCandidateExecutionMaxTimeoutMS {
		return admissionLiveRouteTurnCandidateExecutionForShellWithRuntime(shell, "", admissionLiveRouteTurnCandidateExecutionRuntime{
			Runner:        runner,
			Status:        admissionLiveRouteTurnCandidateExecutionStatusFailed,
			ExitCode:      -1,
			FailureReason: "candidate execution timeout out of bounds",
		})
	}
	exe, err := os.Executable()
	if err != nil {
		return admissionLiveRouteTurnCandidateExecutionForShellWithRuntime(shell, "", admissionLiveRouteTurnCandidateExecutionRuntime{
			Runner:        runner,
			Status:        admissionLiveRouteTurnCandidateExecutionStatusFailed,
			ExitCode:      -1,
			FailureReason: "candidate runner executable unavailable",
		})
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeoutMS)*time.Millisecond)
	defer cancel()
	cmd := exec.CommandContext(ctx, exe, "--admission-live-route-turn-candidate-runner-emit")
	cmd.Env = append(os.Environ(), "AM_LIVE_ROUTE_TURN_CANDIDATE_RUNNER_EMIT_TEXT="+text)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	started := time.Now()
	err = cmd.Run()
	durationMS := time.Since(started).Milliseconds()
	outText := strings.TrimSpace(stdout.String())
	errText := strings.TrimSpace(stderr.String())
	runtime := admissionLiveRouteTurnCandidateExecutionRuntime{
		Runner:     runner,
		Status:     admissionLiveRouteTurnCandidateExecutionStatusSucceeded,
		ExitCode:   0,
		DurationMS: durationMS,
		StdoutHash: admissionLiveRouteTurnCandidateExecutionOutputHash(outText),
		StderrHash: admissionLiveRouteTurnCandidateExecutionOutputHash(errText),
	}
	if cmd.ProcessState != nil {
		runtime.ExitCode = cmd.ProcessState.ExitCode()
	}
	if ctx.Err() == context.DeadlineExceeded {
		runtime.Status = admissionLiveRouteTurnCandidateExecutionStatusTimedOut
		runtime.TimedOut = true
		runtime.ExitCode = -1
		return admissionLiveRouteTurnCandidateExecutionForShellWithRuntime(shell, outText, runtime)
	}
	if err != nil {
		runtime.Status = admissionLiveRouteTurnCandidateExecutionStatusFailed
		if runtime.ExitCode == 0 {
			runtime.ExitCode = -1
		}
		runtime.FailureReason = "candidate runner failed for shell " + shell.ShellID
		return admissionLiveRouteTurnCandidateExecutionForShellWithRuntime(shell, outText, runtime)
	}
	return admissionLiveRouteTurnCandidateExecutionForShellWithRuntime(shell, outText, runtime)
}

func admissionLiveRouteTurnCandidateExecutionOutputHash(text string) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}
	return hashJSON(text)
}

func runAdmissionLiveRouteTurnCandidateRunnerEmit() error {
	if raw := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_RUNNER_EMIT_SLEEP_MS")); raw != "" {
		sleepMS, err := strconv.Atoi(raw)
		if err != nil || sleepMS < 0 || sleepMS > admissionLiveRouteTurnCandidateExecutionMaxTimeoutMS {
			return fmt.Errorf("bad AM_LIVE_ROUTE_TURN_CANDIDATE_RUNNER_EMIT_SLEEP_MS")
		}
		time.Sleep(time.Duration(sleepMS) * time.Millisecond)
	}
	text := strings.TrimSpace(os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_RUNNER_EMIT_TEXT"))
	if text == "" {
		return fmt.Errorf("AM_LIVE_ROUTE_TURN_CANDIDATE_RUNNER_EMIT_TEXT is required")
	}
	fmt.Println(text)
	return nil
}
