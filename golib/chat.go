package main

// chat.go — the live trio. A human converses with the organism turn by turn:
// Janus is the face (the spoken answer), Resonance is the inner voice, the nano
// is the subconscious that dreams a turn behind. The inner world keeps living on
// its ticker while it waits for the human to type (the mind drifts between
// replies), and its mood + the last dream persist across sessions.
//
// Run: ./metabolism --chat   (needs ./arianna, ./arianna_resonance, and — for the
// subconscious — ./nano-arianna + weights/nano_arianna_f16.gguf + weights/nano.kk.db)

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

const innerStatePath = "weights/arianna.inner.state"

// Resonance's δ-harvest parameters — her GGUF embedding tensor and dimensions
// (V=16384, E=768), the co-occurrence and δ sidecars she reads.
const (
	resonGGUF  = "weights/arianna_resonance_v3_f16.gguf"
	resonWTE   = "tok_emb"
	resonVocab = "16384"
	resonDim   = "768"
	resonCooc  = "weights/arianna.cooc.r"
	resonDelta = "weights/arianna.delta.r"
)

func runChat() {
	tc, err := startTrio()
	if err != nil {
		fmt.Println("metabolism:", err)
		return
	}

	lastDream := tc.iw.RestoreMood(innerStatePath) // restore mood + last murmur, atomically vs the ticker

	fmt.Println("┌─ arianna — the trio (Janus · Resonance · the nano).  speak, /quit to leave.")
	if tc.nan != nil {
		fmt.Println("│  the subconscious is present (nano 88M, async — it dreams a turn behind)")
		fmt.Println("│  she breathes on her own — between your words she dreams, and the inner voice answers.")
		if tc.nan.doeBin != "" { // #3: the nano dreams through doe (the parliament engine)
			if f, err := strconv.ParseFloat(tc.nan.doeAlpha, 64); err == nil && f == 0 {
				fmt.Printf("│  she dreams notorch-native through doe — the parliament is silenced (α=%s; unset AM_LORA_ALPHA to seat it)\n", tc.nan.doeAlpha)
			} else {
				fmt.Printf("│  the parliament is seated on her dreams (doe, α=%s; AM_LORA_ALPHA=0 to silence it)\n", tc.nan.doeAlpha)
			}
			if tc.nan.doeTrain == "1" { // step-3: online expert learning opted in
				fmt.Println("│  the parliament learns from her dreams (--train: the experts grow toward what surfaces)")
			}
			if tc.nan.doeD != nil { // persistent: one model load, the field evolves across dreams
				fmt.Println("│  the parliament stays awake between dreams (one load — the field carries over; AM_DOE_DAEMON=0 to reload per dream)")
			}
		}
	}
	if lastDream != "" {
		fmt.Printf("│  (she returns carrying a dream: %s)\n", ellipsize(lastDream, 70))
	}

	// voiceMu serializes ALL voice-daemon access — a human turn and the autonomous
	// breathing must never ask a single-stream daemon at the same time. It also
	// guards the shared lastDream. The breathing goroutine lives by itself until
	// breathStop closes.
	var voiceMu sync.Mutex
	breathStop := make(chan struct{})
	breathDone := make(chan struct{})
	go runBreathing(tc, &voiceMu, &lastDream, breathStop, breathDone)

	// the human turn reads the live field too (its OWN reader — the breathing
	// goroutine owns a separate one, so attach() never races): when the field is
	// expressive, the inner dream lightly surfaces to Janus's face.
	faceFR := newFieldReader(fieldPath)
	defer faceFR.close()

	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	prevReson := ""
	fmt.Print("│\n└▶ ")
	for sc.Scan() {
		human := strings.TrimSpace(sc.Text())
		if human == "" {
			fmt.Print("└▶ ")
			continue
		}
		if human == "/quit" || human == "/exit" {
			break
		}
		voiceMu.Lock() // the human turn owns the voices for its duration
		tc.iw.ProcessText(human)
		turnRouteObs := admissionLiveRouteTurnObservation{}
		if admissionLiveRouteTurnObservationDryRunNeeded() {
			turnRouteObs = admissionLiveRouteTurnObservationForHuman(human)
			if err := recordAdmissionLiveRouteTurnObservation(turnRouteObs); err != nil {
				fmt.Println("│  · live-route turn dry-run log failed:", err)
			}
		}
		// F-2: the direct human→nano channel — the raw words hit the subconscious
		// before the face has formed (the async nano may dream on them while the
		// voices answer); turn() then re-seeds with the turn's context for the next.
		if tc.nan != nil {
			sendLatest(tc.seedCh, human)
		}

		janus, reson, dr, hasDream := tc.turn(human, prevReson, lastDream, faceFR.read().surfaces(), turnRouteObs)
		fmt.Printf("│  ◐ Janus: %s\n", janus)
		fmt.Printf("│  ◑ Resonance: %s\n", reson)
		prevReson = reson
		if line := chatLiveRouteTurnDryRunLine(turnRouteObs); line != "" {
			fmt.Println(line)
		}
		if line := chatLiveRouteTurnChoiceDryRunLine(turnRouteObs); line != "" {
			fmt.Println(line)
		}
		if line := chatLiveRouteTurnRequestDryRunLine(turnRouteObs); line != "" {
			fmt.Println(line)
		}
		if line := chatLiveRouteTurnGenerationJobDryRunLine(turnRouteObs); line != "" {
			fmt.Println(line)
		}
		if line := chatLiveRouteTurnCandidateShellDryRunLine(turnRouteObs); line != "" {
			fmt.Println(line)
		}
		if line := chatLiveRouteTurnCandidateExecutionDryRunLine(turnRouteObs); line != "" {
			fmt.Println(line)
		}
		if line := chatLiveRouteTurnGeneratorAdapterDryRunLine(turnRouteObs); line != "" {
			fmt.Println(line)
		}
		if line := chatLiveRouteTurnCandidateDraftDryRunLine(turnRouteObs); line != "" {
			fmt.Println(line)
		}
		if line := chatLiveRouteTurnCandidateAdmissionDryRunLine(turnRouteObs); line != "" {
			fmt.Println(line)
		}
		if line := chatLiveRouteTurnCandidateAdmissionAdapterDryRunLine(turnRouteObs); line != "" {
			fmt.Println(line)
		}
		if line := chatLiveRouteTurnCandidateAdmissionShadowDryRunLine(turnRouteObs); line != "" {
			fmt.Println(line)
		}
		if hasDream {
			if dr.admitted() {
				lastDream = dr.dream
				if dr.frag != "" {
					fmt.Printf("│  ◌ from the books: %s\n", ellipsize(dr.frag, 90))
				}
				fmt.Printf("│  ◓ nano (subconscious): %s\n", dr.dream)
			} else {
				fmt.Printf("│  ◓ nano candidate (%s): %s\n", dr.admissionLabel(), ellipsize(dr.dream, 90))
			}
			if line := chatLiveRouteChoiceDryRunLine(dr.candidate); line != "" {
				fmt.Println(line)
			}
			if line := chatLiveRouteTurnCandidateReviewLine(turnRouteObs, dr.candidate); line != "" {
				fmt.Println(line)
			}
		}
		// A hot voice daemon can fall silent after a turn or two (it stops framing <END>).
		// Revive it in place instead of ending the session — the trio survives one voice's
		// death and the conversation goes on. Only a failed revival stops the loop.
		if tc.janusD.dead {
			if err := tc.janusD.respawn(); err != nil {
				fmt.Println("│  · Janus fell silent and could not be revived:", err)
				voiceMu.Unlock()
				break
			}
			fmt.Println("│  · Janus fell silent — revived.")
		}
		if tc.resonD.dead {
			if err := tc.resonD.respawn(); err != nil {
				fmt.Println("│  · Resonance fell silent and could not be revived:", err)
				voiceMu.Unlock()
				break
			}
			fmt.Println("│  · Resonance fell silent — revived.")
		}
		voiceMu.Unlock()

		fmt.Print("│\n└▶ ")
	}

	close(breathStop) // stop the autonomous breathing before tearing the voices down
	// budget the FULL in-flight breathing cycle (kkRetrieve THEN the dream) so an
	// already-running fallback dream finishes — or hits its own ctx-kill — before
	// SaveState/stop touch the voices. doe dreams run up to doeDreamTimeout, longer
	// than the nanollama dreamTimeout.
	breathJoin := dreamTimeout
	if tc.nan != nil && tc.nan.doeBin != "" {
		breathJoin = doeDreamTimeout
	}
	select {
	case <-breathDone:
	case <-time.After(breathJoin + kkTimeout + 5*time.Second):
	}

	fmt.Println()
	if err := tc.iw.SaveState(innerStatePath, lastDream); err != nil {
		fmt.Println("(could not save the inner state:", err, ")")
	} else {
		fmt.Println("(she will remember.)")
	}
	tc.stop()      // close the voices — Resonance saves her co-occurrence sidecar
	harvestField() // Phase 2 (A): fold what surfaced into δ; report the growth
}

func admissionLiveRouteTurnObservationDryRunNeeded() bool {
	return dreamAdmissionLiveRouteChoiceDryRun() ||
		admissionLiveRouteTurnChoiceDryRun() ||
		admissionLiveRouteTurnRequestDryRun() ||
		admissionLiveRouteTurnGenerationJobDryRun() ||
		admissionLiveRouteTurnCandidateShellDryRun() ||
		admissionLiveRouteTurnCandidateExecutionDryRun() ||
		admissionLiveRouteTurnGeneratorAdapterDryRun() ||
		admissionLiveRouteTurnCandidateDraftDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionAdapterDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionShadowDryRun()
}

func chatLiveRouteChoiceDryRunLine(c dreamCandidate) string {
	if !dreamAdmissionLiveRouteChoiceDryRun() || c.Admission == nil || c.Admission.LiveRouteChoice == nil {
		return ""
	}
	choice := c.Admission.LiveRouteChoice
	reason := ""
	if choice.Reason != "" {
		reason = " reason=" + choice.Reason
	}
	return fmt.Sprintf("│  · live-route dry-run: class=%s route=%s source=%s expected=%s passed=%t%s",
		choice.PromptClass, choice.Route, choice.Source, choice.ExpectedSource, choice.Passed, reason)
}

func chatLiveRouteTurnDryRunLine(obs admissionLiveRouteTurnObservation) string {
	if !dreamAdmissionLiveRouteChoiceDryRun() || obs.Schema == "" {
		return ""
	}
	reason := ""
	if obs.Reason != "" {
		reason = " reason=" + obs.Reason
	}
	return fmt.Sprintf("│  · live-route turn dry-run: class=%s route=%s expected=%s passed=%t score=%d%s",
		obs.PromptClass, obs.Route, obs.ExpectedSource, obs.Passed, obs.ClassScore, reason)
}

func chatLiveRouteTurnChoiceDryRunLine(obs admissionLiveRouteTurnObservation) string {
	if !admissionLiveRouteTurnChoiceDryRun() || obs.Schema == "" {
		return ""
	}
	choice := admissionLiveRouteTurnChoiceForObservation(obs)
	if err := recordAdmissionLiveRouteTurnChoice(choice); err != nil {
		return fmt.Sprintf("│  · live-route turn choice dry-run log failed: %v", err)
	}
	reason := ""
	if choice.Reason != "" {
		reason = " reason=" + choice.Reason
	}
	return fmt.Sprintf("│  · live-route turn choice dry-run: class=%s route=%s source=%s trigger=%s passed=%t%s",
		choice.PromptClass, choice.Route, choice.Source, choice.CandidateTrigger, choice.Passed, reason)
}

func chatLiveRouteTurnRequestDryRunLine(obs admissionLiveRouteTurnObservation) string {
	if !admissionLiveRouteTurnRequestDryRun() || obs.Schema == "" {
		return ""
	}
	choice := admissionLiveRouteTurnChoiceForObservation(obs)
	request := admissionLiveRouteTurnRequestForChoice(choice)
	if err := recordAdmissionLiveRouteTurnRequest(request); err != nil {
		return fmt.Sprintf("│  · live-route turn request dry-run log failed: %v", err)
	}
	reason := ""
	if request.Reason != "" {
		reason = " reason=" + request.Reason
	}
	return fmt.Sprintf("│  · live-route turn request dry-run: class=%s route=%s source=%s trigger=%s seed=%s passed=%t%s",
		request.PromptClass, request.Route, request.Source, request.CandidateTrigger, request.CandidateSeed, request.Passed, reason)
}

func chatLiveRouteTurnGenerationJobDryRunLine(obs admissionLiveRouteTurnObservation) string {
	if !admissionLiveRouteTurnGenerationJobDryRun() || obs.Schema == "" {
		return ""
	}
	choice := admissionLiveRouteTurnChoiceForObservation(obs)
	request := admissionLiveRouteTurnRequestForChoice(choice)
	job := admissionLiveRouteTurnGenerationJobForRequest(request)
	if err := recordAdmissionLiveRouteTurnGenerationJob(job); err != nil {
		return fmt.Sprintf("│  · live-route generation job dry-run log failed: %v", err)
	}
	reason := ""
	if job.Reason != "" {
		reason = " reason=" + job.Reason
	}
	return fmt.Sprintf("│  · live-route generation job dry-run: class=%s route=%s backend=%s entry=%s trigger=%s seed=%s job=%s passed=%t%s",
		job.PromptClass, job.Route, job.Backend, job.Entrypoint, job.CandidateTrigger, job.CandidateSeed, job.JobID, job.Passed, reason)
}

func chatLiveRouteTurnCandidateShellDryRunLine(obs admissionLiveRouteTurnObservation) string {
	if !admissionLiveRouteTurnCandidateShellDryRun() || obs.Schema == "" {
		return ""
	}
	choice := admissionLiveRouteTurnChoiceForObservation(obs)
	request := admissionLiveRouteTurnRequestForChoice(choice)
	job := admissionLiveRouteTurnGenerationJobForRequest(request)
	shell := admissionLiveRouteTurnCandidateShellForJob(job)
	if err := recordAdmissionLiveRouteTurnCandidateShell(shell); err != nil {
		return fmt.Sprintf("│  · live-route candidate shell dry-run log failed: %v", err)
	}
	reason := ""
	if shell.Reason != "" {
		reason = " reason=" + shell.Reason
	}
	return fmt.Sprintf("│  · live-route candidate shell dry-run: class=%s route=%s source=%s trigger=%s seed=%s job=%s shell=%s text=%s passed=%t%s",
		shell.PromptClass, shell.Route, shell.Source, shell.CandidateTrigger, shell.CandidateSeed, shell.JobID, shell.ShellID, shell.CandidateTextStatus, shell.Passed, reason)
}

func chatLiveRouteTurnCandidateExecutionDryRunLine(obs admissionLiveRouteTurnObservation) string {
	return chatLiveRouteTurnCandidateExecutionDryRunLineForText(obs, os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_TEXT"))
}

func chatLiveRouteTurnCandidateExecutionForText(obs admissionLiveRouteTurnObservation, text string) admissionLiveRouteTurnCandidateExecution {
	choice := admissionLiveRouteTurnChoiceForObservation(obs)
	request := admissionLiveRouteTurnRequestForChoice(choice)
	job := admissionLiveRouteTurnGenerationJobForRequest(request)
	shell := admissionLiveRouteTurnCandidateShellForJob(job)
	return admissionLiveRouteTurnCandidateExecutionForShell(shell, text)
}

func chatLiveRouteTurnCandidateExecutionDryRunLineForText(obs admissionLiveRouteTurnObservation, text string) string {
	if !admissionLiveRouteTurnCandidateExecutionDryRun() || obs.Schema == "" {
		return ""
	}
	execution := chatLiveRouteTurnCandidateExecutionForText(obs, text)
	if err := recordAdmissionLiveRouteTurnCandidateExecution(execution); err != nil {
		return fmt.Sprintf("│  · live-route candidate execution dry-run log failed: %v", err)
	}
	reason := ""
	if execution.Reason != "" {
		reason = " reason=" + execution.Reason
	}
	return fmt.Sprintf("│  · live-route candidate execution dry-run: class=%s route=%s backend=%s entry=%s frame=%s executor=%s timeout_ms=%d shell=%s execution=%s text=%s passed=%t%s",
		execution.PromptClass, execution.Route, execution.Backend, execution.Entrypoint, execution.PromptFrame,
		execution.Executor, execution.TimeoutMS, execution.ShellID, execution.ExecutionID, execution.GeneratedTextStatus, execution.Passed, reason)
}

func chatLiveRouteTurnGeneratorAdapterDryRunLine(obs admissionLiveRouteTurnObservation) string {
	text := os.Getenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_TEXT")
	if strings.TrimSpace(text) == "" && admissionLiveRouteTurnCandidateExecutionDryRun() {
		text = os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_TEXT")
	}
	return chatLiveRouteTurnGeneratorAdapterDryRunLineForText(obs, text)
}

func chatLiveRouteTurnGeneratorAdapterForText(obs admissionLiveRouteTurnObservation, text string) admissionLiveRouteTurnGeneratorAdapter {
	choice := admissionLiveRouteTurnChoiceForObservation(obs)
	request := admissionLiveRouteTurnRequestForChoice(choice)
	job := admissionLiveRouteTurnGenerationJobForRequest(request)
	shell := admissionLiveRouteTurnCandidateShellForJob(job)
	if admissionLiveRouteTurnCandidateExecutionDryRun() {
		execution := admissionLiveRouteTurnCandidateExecutionForShell(shell, text)
		return admissionLiveRouteTurnGeneratorAdapterForExecution(execution)
	}
	return admissionLiveRouteTurnGeneratorAdapterForShell(shell, text)
}

func chatLiveRouteTurnGeneratorAdapterDryRunLineForText(obs admissionLiveRouteTurnObservation, text string) string {
	if !admissionLiveRouteTurnGeneratorAdapterDryRun() || obs.Schema == "" {
		return ""
	}
	adapter := chatLiveRouteTurnGeneratorAdapterForText(obs, text)
	if err := recordAdmissionLiveRouteTurnGeneratorAdapter(adapter); err != nil {
		return fmt.Sprintf("│  · live-route generator adapter dry-run log failed: %v", err)
	}
	reason := ""
	if adapter.Reason != "" {
		reason = " reason=" + adapter.Reason
	}
	return fmt.Sprintf("│  · live-route generator adapter dry-run: class=%s route=%s backend=%s entry=%s frame=%s shell=%s execution=%s adapter=%s text=%s passed=%t%s",
		adapter.PromptClass, adapter.Route, adapter.Backend, adapter.Entrypoint, adapter.PromptFrame,
		adapter.ShellID, adapter.CandidateExecutionID, adapter.AdapterID, adapter.GeneratedTextStatus, adapter.Passed, reason)
}

func chatLiveRouteTurnCandidateDraftDryRunLine(obs admissionLiveRouteTurnObservation) string {
	text := os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_TEXT")
	if strings.TrimSpace(text) == "" && admissionLiveRouteTurnCandidateExecutionDryRun() {
		text = os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_TEXT")
	}
	return chatLiveRouteTurnCandidateDraftDryRunLineForText(obs, text)
}

func chatLiveRouteTurnCandidateDraftForText(obs admissionLiveRouteTurnObservation, text string) admissionLiveRouteTurnCandidateDraft {
	adapter := chatLiveRouteTurnGeneratorAdapterForText(obs, text)
	return admissionLiveRouteTurnCandidateDraftForAdapter(adapter)
}

func chatLiveRouteTurnCandidateDraftDryRunLineForText(obs admissionLiveRouteTurnObservation, text string) string {
	if !admissionLiveRouteTurnCandidateDraftDryRun() || obs.Schema == "" {
		return ""
	}
	draft := chatLiveRouteTurnCandidateDraftForText(obs, text)
	if err := recordAdmissionLiveRouteTurnCandidateDraft(draft); err != nil {
		return fmt.Sprintf("│  · live-route candidate draft dry-run log failed: %v", err)
	}
	reason := ""
	if draft.Reason != "" {
		reason = " reason=" + draft.Reason
	}
	return fmt.Sprintf("│  · live-route candidate draft dry-run: class=%s route=%s source=%s trigger=%s seed=%s shell=%s execution=%s adapter=%s draft=%s run=%s text=%s passed=%t%s",
		draft.PromptClass, draft.Route, draft.Source, draft.CandidateTrigger, draft.CandidateSeed,
		draft.ShellID, draft.CandidateExecutionID, draft.GeneratorAdapterID, draft.DraftID, draft.CandidateRunID, draft.CandidateTextStatus, draft.Passed, reason)
}

func chatLiveRouteTurnCandidateAdmissionDryRunLine(obs admissionLiveRouteTurnObservation) string {
	return chatLiveRouteTurnCandidateAdmissionDryRunLineForText(obs, os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_TEXT"))
}

func chatLiveRouteTurnCandidateAdmissionDryRunLineForText(obs admissionLiveRouteTurnObservation, text string) string {
	if !admissionLiveRouteTurnCandidateAdmissionDryRun() || !admissionLiveRouteTurnCandidateDraftDryRun() || obs.Schema == "" {
		return ""
	}
	draft := chatLiveRouteTurnCandidateDraftForText(obs, text)
	review := admissionLiveRouteTurnCandidateReviewForDraft(obs, draft)
	if err := recordAdmissionLiveRouteTurnCandidateReview(review); err != nil {
		return fmt.Sprintf("│  · live-route candidate admission handoff dry-run review log failed: %v", err)
	}
	admission := admissionLiveRouteTurnCandidateAdmissionForDraftReview(obs, draft, review)
	if err := recordAdmissionLiveRouteTurnCandidateAdmission(admission); err != nil {
		return fmt.Sprintf("│  · live-route candidate admission handoff dry-run log failed: %v", err)
	}
	reason := ""
	if admission.Reason != "" {
		reason = " reason=" + admission.Reason
	}
	return fmt.Sprintf("│  · live-route candidate admission handoff dry-run: class=%s route=%s source=%s draft=%s adapter=%s handoff=%s review=%t passed=%t%s",
		admission.PromptClass, admission.Route, admission.Source, admission.CandidateDraftID,
		admission.GeneratorAdapterID, admission.HandoffID, admission.ReviewMatched, admission.Passed, reason)
}

func chatLiveRouteTurnCandidateAdmissionAdapterDryRunLine(obs admissionLiveRouteTurnObservation) string {
	return chatLiveRouteTurnCandidateAdmissionAdapterDryRunLineForText(obs, os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_TEXT"))
}

func chatLiveRouteTurnCandidateAdmissionAdapterDryRunLineForText(obs admissionLiveRouteTurnObservation, text string) string {
	if !admissionLiveRouteTurnCandidateAdmissionAdapterDryRun() ||
		!admissionLiveRouteTurnCandidateAdmissionDryRun() ||
		!admissionLiveRouteTurnCandidateDraftDryRun() ||
		obs.Schema == "" {
		return ""
	}
	draft := chatLiveRouteTurnCandidateDraftForText(obs, text)
	review := admissionLiveRouteTurnCandidateReviewForDraft(obs, draft)
	admission := admissionLiveRouteTurnCandidateAdmissionForDraftReview(obs, draft, review)
	adapter := admissionLiveRouteTurnCandidateAdmissionAdapterForDraft(admission, draft)
	if err := recordAdmissionLiveRouteTurnCandidateAdmissionAdapter(adapter); err != nil {
		return fmt.Sprintf("│  · live-route candidate admission adapter dry-run log failed: %v", err)
	}
	reason := ""
	if adapter.Reason != "" {
		reason = " reason=" + adapter.Reason
	}
	return fmt.Sprintf("│  · live-route candidate admission adapter dry-run: class=%s route=%s source=%s handoff=%s admission_adapter=%s run=%s passed=%t%s",
		adapter.PromptClass, adapter.Route, adapter.Source, adapter.HandoffID,
		adapter.AdmissionAdapterID, adapter.DreamCandidateRunID, adapter.Passed, reason)
}

func chatLiveRouteTurnCandidateAdmissionShadowDryRunLine(obs admissionLiveRouteTurnObservation) string {
	return chatLiveRouteTurnCandidateAdmissionShadowDryRunLineForText(obs, os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_TEXT"))
}

func chatLiveRouteTurnCandidateAdmissionShadowDryRunLineForText(obs admissionLiveRouteTurnObservation, text string) string {
	if !admissionLiveRouteTurnCandidateAdmissionShadowDryRun() ||
		!admissionLiveRouteTurnCandidateAdmissionAdapterDryRun() ||
		!admissionLiveRouteTurnCandidateAdmissionDryRun() ||
		!admissionLiveRouteTurnCandidateDraftDryRun() ||
		obs.Schema == "" {
		return ""
	}
	draft := chatLiveRouteTurnCandidateDraftForText(obs, text)
	review := admissionLiveRouteTurnCandidateReviewForDraft(obs, draft)
	admission := admissionLiveRouteTurnCandidateAdmissionForDraftReview(obs, draft, review)
	adapter := admissionLiveRouteTurnCandidateAdmissionAdapterForDraft(admission, draft)
	passed := false
	accepted := false
	policyPassed := false
	reason := ""
	if dreamAdmissionMode() != dreamAdmissionShadow {
		reason = "AM_DREAM_ADMISSION must be shadow"
	} else if !dreamAdmissionRequireLiveRoutePlan() {
		reason = "AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN is required"
	} else {
		candidate := admissionLiveRouteTurnCandidateForAdmissionAdapter(draft, adapter)
		if candidate.Schema == "" {
			reason = "candidate_admission_adapter_failed"
			if adapter.Reason != "" {
				reason += ": " + adapter.Reason
			}
		} else {
			candidate = prepareDreamCandidateForAdmissionWithTurnObservation(NewInnerWorld(), candidate, obs)
			accepted = candidate.Accepted
			policyPassed = candidate.Admission != nil && candidate.Admission.Checked && candidate.Admission.Passed
			passed = candidate.Schema == "arianna.dream_candidate.v1" &&
				candidate.LiveRouteCandidateAdmission != nil &&
				candidate.LiveRouteCandidateAdmission.AdmissionAdapterID == adapter.AdmissionAdapterID &&
				!candidate.Accepted &&
				policyPassed
			reason = candidate.Reason
			if !policyPassed && candidate.Admission != nil && len(candidate.Admission.Reasons) > 0 {
				reason = "admission policy failed: " + strings.Join(candidate.Admission.Reasons, "; ")
			}
		}
	}
	reasonSuffix := ""
	if reason != "" {
		reasonSuffix = " reason=" + reason
	}
	return fmt.Sprintf("│  · live-route candidate admission shadow dry-run: class=%s route=%s source=%s handoff=%s admission_adapter=%s run=%s policy=%t accepted=%t passed=%t%s",
		adapter.PromptClass, adapter.Route, adapter.Source, adapter.HandoffID,
		adapter.AdmissionAdapterID, adapter.DreamCandidateRunID, policyPassed, accepted, passed, reasonSuffix)
}

func chatLiveRouteTurnCandidateReviewLine(obs admissionLiveRouteTurnObservation, c dreamCandidate) string {
	if !dreamAdmissionLiveRouteChoiceDryRun() || obs.Schema == "" || c.Schema == "" {
		return ""
	}
	review := admissionLiveRouteTurnCandidateReviewForDream(obs, c)
	if err := recordAdmissionLiveRouteTurnCandidateReview(review); err != nil {
		return fmt.Sprintf("│  · live-route turn/candidate review log failed: %v", err)
	}
	reason := ""
	if review.Reason != "" {
		reason = " reason=" + review.Reason
	}
	bridge := ""
	if review.CandidateBridgeApplied {
		bridge = " bridge=" + review.CandidateBridgeTrigger
	}
	return fmt.Sprintf("│  · live-route turn/candidate review: turn_class=%s expected=%s candidate_source=%s candidate_class=%s candidate_route=%s matched=%t%s%s",
		review.TurnPromptClass, review.TurnExpectedSource, review.CandidateSource,
		review.CandidatePromptClass, review.CandidateRoute, review.Matched, bridge, reason)
}

// harvestField is Phase 2 (A): the organism learns from the subconscious. The
// whole conversation was tinted by the subconscious surfacing into Resonance's
// inject (1d), so her grown co-occurrence carries the subconscious's influence.
// At session end this folds that cooc into her δ via the notorch Hebbian
// (harvest_delta / am_cooc_learn_delta) so next session her voice is shaped by
// what the subconscious taught — async between turns, never mid-sentence. The
// harvest reports |B|, the learning made visible. Absent tool or empty cooc =>
// skipped (nothing was learned). The voice applies the δ only when its blend is
// non-zero, so this is dormant by default — it grows memory without forcing it.
func harvestField() {
	if _, err := os.Stat("./harvest_delta"); err != nil {
		return
	}
	// bound the consolidation like every other subprocess — a wedged harvest must
	// not hang the exit after the voices are already stopped.
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "./harvest_delta",
		resonGGUF, resonWTE, resonCooc, resonDelta, resonVocab, resonDim, "8").CombinedOutput()
	for _, line := range strings.Split(string(out), "\n") {
		if i := strings.Index(line, "|B|="); i >= 0 {
			fmt.Printf("│  the organism consolidated what surfaced — δ %s\n", strings.TrimSpace(line[i:]))
			return
		}
	}
	// F-6: no |B| line — the consolidation did not happen (empty cooc, dim mismatch,
	// a crash). Don't swallow it; surface the reason.
	reason := "nothing surfaced to consolidate"
	if lines := strings.Split(strings.TrimSpace(string(out)), "\n"); err != nil && len(lines) > 0 {
		reason = lines[len(lines)-1]
	}
	fmt.Printf("│  (she could not consolidate — %s)\n", reason)
}
