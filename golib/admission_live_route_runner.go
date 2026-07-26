package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	admissionLiveRouteTurnNanoDirectDefaultBin       = "./nano-arianna"
	admissionLiveRouteTurnNanoDirectDefaultModel     = "weights/nano_arianna_f16.gguf"
	admissionLiveRouteTurnNanoDirectDefaultMaxTokens = "32"
	admissionLiveRouteTurnNanoDirectDefaultTemp      = "0.9"
	admissionLiveRouteTurnNanoDirectDefaultTopP      = "0.92"
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
	switch runner {
	case admissionLiveRouteTurnCandidateExecutionRunnerSelfEmit:
		return admissionLiveRouteTurnCandidateExecutionForShellViaSelfEmitRunner(shell, text)
	case admissionLiveRouteTurnCandidateExecutionRunnerNanoDirect:
		return admissionLiveRouteTurnCandidateExecutionForShellViaNanoDirectRunner(shell, text)
	default:
		return admissionLiveRouteTurnCandidateExecutionForShellWithRuntime(shell, "", admissionLiveRouteTurnCandidateExecutionRuntime{
			Runner:        runner,
			Status:        admissionLiveRouteTurnCandidateExecutionStatusFailed,
			ExitCode:      -1,
			FailureReason: "unknown candidate execution runner " + runner,
		})
	}
}

func admissionLiveRouteTurnCandidateExecutionForShellViaSelfEmitRunner(shell admissionLiveRouteTurnCandidateShell, text string) admissionLiveRouteTurnCandidateExecution {
	runner := admissionLiveRouteTurnCandidateExecutionRunnerSelfEmit
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

func admissionLiveRouteTurnCandidateExecutionForShellViaNanoDirectRunner(shell admissionLiveRouteTurnCandidateShell, text string) admissionLiveRouteTurnCandidateExecution {
	runner := admissionLiveRouteTurnCandidateExecutionRunnerNanoDirect
	if reason := admissionLiveRouteTurnCandidateNanoDirectShellFailureReason(shell); reason != "" {
		return admissionLiveRouteTurnCandidateExecutionForShellWithRuntime(shell, "", admissionLiveRouteTurnCandidateExecutionRuntime{
			Runner:        runner,
			Status:        admissionLiveRouteTurnCandidateExecutionStatusFailed,
			ExitCode:      -1,
			FailureReason: reason,
		})
	}
	prompt := strings.TrimSpace(text)
	if prompt == "" {
		return admissionLiveRouteTurnCandidateExecutionForShellWithRuntime(shell, "", admissionLiveRouteTurnCandidateExecutionRuntime{
			Runner:        runner,
			Status:        admissionLiveRouteTurnCandidateExecutionStatusFailed,
			ExitCode:      -1,
			FailureReason: "candidate nano-direct runner missing prompt for shell " + shell.ShellID,
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
	bin := admissionLiveRouteTurnNanoDirectBin()
	if !admissionLiveRouteTurnCandidateRunnerExecutableExists(bin) {
		return admissionLiveRouteTurnCandidateExecutionForShellWithRuntime(shell, "", admissionLiveRouteTurnCandidateExecutionRuntime{
			Runner:        runner,
			Status:        admissionLiveRouteTurnCandidateExecutionStatusFailed,
			ExitCode:      -1,
			FailureReason: "candidate nano-direct runner missing binary " + bin,
		})
	}
	model := admissionLiveRouteTurnNanoDirectModel()
	if !admissionLiveRouteTurnCandidateRunnerFileExists(model) {
		return admissionLiveRouteTurnCandidateExecutionForShellWithRuntime(shell, "", admissionLiveRouteTurnCandidateExecutionRuntime{
			Runner:        runner,
			Status:        admissionLiveRouteTurnCandidateExecutionStatusFailed,
			ExitCode:      -1,
			FailureReason: "candidate nano-direct runner missing model " + model,
		})
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeoutMS)*time.Millisecond)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin,
		"--model", model,
		"--prompt", "Q: "+prompt+"\nA:",
		"--max-tokens", admissionLiveRouteTurnNanoDirectMaxTokens(),
		"--temp", admissionLiveRouteTurnNanoDirectTemp(),
		"--top-p", admissionLiveRouteTurnNanoDirectTopP(),
	)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	started := time.Now()
	err := cmd.Run()
	durationMS := time.Since(started).Milliseconds()
	outText := cleanDream(stdout.String())
	errText := strings.TrimSpace(strings.ToValidUTF8(stderr.String(), ""))
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
		runtime.FailureReason = "candidate nano-direct runner failed for shell " + shell.ShellID
		return admissionLiveRouteTurnCandidateExecutionForShellWithRuntime(shell, outText, runtime)
	}
	if strings.TrimSpace(outText) == "" {
		runtime.Status = admissionLiveRouteTurnCandidateExecutionStatusFailed
		runtime.ExitCode = -1
		runtime.FailureReason = "candidate nano-direct runner produced no generated text for shell " + shell.ShellID
		return admissionLiveRouteTurnCandidateExecutionForShellWithRuntime(shell, "", runtime)
	}
	return admissionLiveRouteTurnCandidateExecutionForShellWithRuntime(shell, outText, runtime)
}

func admissionLiveRouteTurnCandidateNanoDirectShellFailureReason(shell admissionLiveRouteTurnCandidateShell) string {
	route, ok := admissionLiveRouteGenerationRouteFor("direct")
	if !ok {
		return "candidate nano-direct runner direct route unavailable"
	}
	if shell.Route != "direct" {
		return "candidate nano-direct runner only supports direct route, got " + shell.Route
	}
	if shell.Source != admissionLiveRouteSource("direct") {
		return "candidate nano-direct runner only supports source " + admissionLiveRouteSource("direct") + ", got " + shell.Source
	}
	if shell.Backend != route.Backend || shell.Entrypoint != route.Entrypoint || shell.PromptFrame != route.PromptFrame {
		return "candidate nano-direct runner executor mismatch for shell " + shell.ShellID
	}
	return ""
}

func admissionLiveRouteTurnNanoDirectBin() string {
	return admissionLiveRouteTurnCandidateRunnerResolveRepoPath(admissionLiveRouteTurnCandidateRunnerEnv(
		"AM_LIVE_ROUTE_TURN_NANO_DIRECT_BIN",
		"AM_LIVE_ROUTE_TURN_NANO_BIN",
		admissionLiveRouteTurnNanoDirectDefaultBin,
	))
}

func admissionLiveRouteTurnNanoDirectModel() string {
	return admissionLiveRouteTurnCandidateRunnerResolveRepoPath(admissionLiveRouteTurnCandidateRunnerEnv(
		"AM_LIVE_ROUTE_TURN_NANO_DIRECT_MODEL",
		"AM_LIVE_ROUTE_TURN_NANO_MODEL",
		admissionLiveRouteTurnNanoDirectDefaultModel,
	))
}

func admissionLiveRouteTurnNanoDirectMaxTokens() string {
	return admissionLiveRouteTurnCandidateRunnerEnv(
		"AM_LIVE_ROUTE_TURN_NANO_DIRECT_MAX_TOKENS",
		"AM_LIVE_ROUTE_TURN_NANO_MAX_TOKENS",
		admissionLiveRouteTurnNanoDirectDefaultMaxTokens,
	)
}

func admissionLiveRouteTurnNanoDirectTemp() string {
	return admissionLiveRouteTurnCandidateRunnerEnv(
		"AM_LIVE_ROUTE_TURN_NANO_DIRECT_TEMP",
		"AM_LIVE_ROUTE_TURN_NANO_TEMP",
		admissionLiveRouteTurnNanoDirectDefaultTemp,
	)
}

func admissionLiveRouteTurnNanoDirectTopP() string {
	return admissionLiveRouteTurnCandidateRunnerEnv(
		"AM_LIVE_ROUTE_TURN_NANO_DIRECT_TOP_P",
		"AM_LIVE_ROUTE_TURN_NANO_TOP_P",
		admissionLiveRouteTurnNanoDirectDefaultTopP,
	)
}

func admissionLiveRouteTurnCandidateRunnerEnv(primary, secondary, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(primary)); value != "" {
		return value
	}
	if value := strings.TrimSpace(os.Getenv(secondary)); value != "" {
		return value
	}
	return fallback
}

func admissionLiveRouteTurnCandidateRunnerResolveRepoPath(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" || filepath.IsAbs(raw) {
		return raw
	}
	root := admissionLiveRouteTurnCandidateRunnerExecutableRoot()
	if root == "" {
		return raw
	}
	local := filepath.Join(root, raw)
	if admissionLiveRouteTurnCandidateRunnerFileExists(local) {
		return local
	}
	if mainRoot := admissionLiveRouteTurnCandidateRunnerMainRoot(root); mainRoot != "" {
		alt := filepath.Join(mainRoot, raw)
		if admissionLiveRouteTurnCandidateRunnerFileExists(alt) {
			return alt
		}
	}
	return local
}

func admissionLiveRouteTurnCandidateRunnerExecutableRoot() string {
	exe, err := os.Executable()
	if err != nil {
		return ""
	}
	return filepath.Dir(exe)
}

func admissionLiveRouteTurnCandidateRunnerMainRoot(root string) string {
	marker := string(os.PathSeparator) + ".worktrees" + string(os.PathSeparator)
	idx := strings.Index(root, marker)
	if idx < 0 {
		return ""
	}
	return root[:idx]
}

func admissionLiveRouteTurnCandidateRunnerFileExists(path string) bool {
	if strings.TrimSpace(path) == "" {
		return false
	}
	st, err := os.Stat(path)
	return err == nil && !st.IsDir()
}

func admissionLiveRouteTurnCandidateRunnerExecutableExists(path string) bool {
	if !admissionLiveRouteTurnCandidateRunnerFileExists(path) {
		return false
	}
	st, err := os.Stat(path)
	return err == nil && st.Mode().Perm()&0111 != 0
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
