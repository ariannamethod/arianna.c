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
		for _, line := range chatLiveRouteTurnCandidateChainDryRunLines(turnRouteObs) {
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
		admissionLiveRouteTurnCandidateAdmissionShadowDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionFinalGateDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceIntentDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceReceiverDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceObservationDryRun()
}

type chatLiveRouteTurnCandidateChain struct {
	Execution        admissionLiveRouteTurnCandidateExecution
	Adapter          admissionLiveRouteTurnGeneratorAdapter
	Draft            admissionLiveRouteTurnCandidateDraft
	Review           admissionLiveRouteTurnCandidateReview
	Admission        admissionLiveRouteTurnCandidateAdmission
	AdmissionAdapter admissionLiveRouteTurnCandidateAdmissionAdapter
}

type chatLiveRouteTurnCandidateAdmissionShadowResult struct {
	Candidate    dreamCandidate
	PolicyPassed bool
	Accepted     bool
	Passed       bool
	Reason       string
}

func chatLiveRouteTurnCandidateChainDryRunNeeded() bool {
	return admissionLiveRouteTurnCandidateExecutionDryRun() ||
		admissionLiveRouteTurnGeneratorAdapterDryRun() ||
		admissionLiveRouteTurnCandidateDraftDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionAdapterDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionShadowDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionFinalGateDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceIntentDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceReceiverDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceObservationDryRun()
}

func chatLiveRouteTurnCandidateChainText() string {
	if admissionLiveRouteTurnCandidateExecutionDryRun() {
		if text := os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_TEXT"); strings.TrimSpace(text) != "" {
			return text
		}
	}
	if admissionLiveRouteTurnGeneratorAdapterDryRun() {
		if text := os.Getenv("AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_TEXT"); strings.TrimSpace(text) != "" {
			return text
		}
	}
	if admissionLiveRouteTurnCandidateDraftDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionAdapterDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionShadowDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionFinalGateDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceIntentDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionResonanceReceiverDryRun() {
		if text := os.Getenv("AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_TEXT"); strings.TrimSpace(text) != "" {
			return text
		}
	}
	return ""
}

func chatLiveRouteTurnCandidateChainForText(obs admissionLiveRouteTurnObservation, text string) chatLiveRouteTurnCandidateChain {
	choice := admissionLiveRouteTurnChoiceForObservation(obs)
	request := admissionLiveRouteTurnRequestForChoice(choice)
	job := admissionLiveRouteTurnGenerationJobForRequest(request)
	shell := admissionLiveRouteTurnCandidateShellForJob(job)
	chain := chatLiveRouteTurnCandidateChain{}
	if admissionLiveRouteTurnCandidateExecutionDryRun() {
		chain.Execution = admissionLiveRouteTurnCandidateExecutionForShellViaRunner(shell, text)
		chain.Adapter = admissionLiveRouteTurnGeneratorAdapterForExecution(chain.Execution)
	} else {
		chain.Adapter = admissionLiveRouteTurnGeneratorAdapterForShell(shell, text)
	}
	chain.Draft = admissionLiveRouteTurnCandidateDraftForAdapter(chain.Adapter)
	chain.Review = admissionLiveRouteTurnCandidateReviewForDraft(obs, chain.Draft)
	chain.Admission = admissionLiveRouteTurnCandidateAdmissionForDraftReview(obs, chain.Draft, chain.Review)
	chain.AdmissionAdapter = admissionLiveRouteTurnCandidateAdmissionAdapterForDraft(chain.Admission, chain.Draft)
	return chain
}

func chatLiveRouteReasonSuffix(reason string) string {
	if reason == "" {
		return ""
	}
	return " reason=" + reason
}

func chatLiveRouteTurnCandidateExecutionLine(execution admissionLiveRouteTurnCandidateExecution) string {
	return fmt.Sprintf("│  · live-route candidate execution dry-run: class=%s route=%s backend=%s entry=%s frame=%s executor=%s timeout_ms=%d shell=%s execution=%s text=%s runner=%s runner_status=%s passed=%t%s",
		execution.PromptClass, execution.Route, execution.Backend, execution.Entrypoint, execution.PromptFrame,
		execution.Executor, execution.TimeoutMS, execution.ShellID, execution.ExecutionID, execution.GeneratedTextStatus,
		execution.Runner, execution.RunnerStatus, execution.Passed, chatLiveRouteReasonSuffix(execution.Reason))
}

func chatLiveRouteTurnGeneratorAdapterLine(adapter admissionLiveRouteTurnGeneratorAdapter) string {
	return fmt.Sprintf("│  · live-route generator adapter dry-run: class=%s route=%s backend=%s entry=%s frame=%s shell=%s execution=%s adapter=%s text=%s passed=%t%s",
		adapter.PromptClass, adapter.Route, adapter.Backend, adapter.Entrypoint, adapter.PromptFrame,
		adapter.ShellID, adapter.CandidateExecutionID, adapter.AdapterID, adapter.GeneratedTextStatus, adapter.Passed, chatLiveRouteReasonSuffix(adapter.Reason))
}

func chatLiveRouteTurnCandidateDraftLine(draft admissionLiveRouteTurnCandidateDraft) string {
	return fmt.Sprintf("│  · live-route candidate draft dry-run: class=%s route=%s source=%s trigger=%s seed=%s shell=%s execution=%s adapter=%s draft=%s run=%s text=%s passed=%t%s",
		draft.PromptClass, draft.Route, draft.Source, draft.CandidateTrigger, draft.CandidateSeed,
		draft.ShellID, draft.CandidateExecutionID, draft.GeneratorAdapterID, draft.DraftID, draft.CandidateRunID, draft.CandidateTextStatus, draft.Passed, chatLiveRouteReasonSuffix(draft.Reason))
}

func chatLiveRouteTurnCandidateAdmissionLine(admission admissionLiveRouteTurnCandidateAdmission) string {
	return fmt.Sprintf("│  · live-route candidate admission handoff dry-run: class=%s route=%s source=%s draft=%s adapter=%s handoff=%s review=%t passed=%t%s",
		admission.PromptClass, admission.Route, admission.Source, admission.CandidateDraftID,
		admission.GeneratorAdapterID, admission.HandoffID, admission.ReviewMatched, admission.Passed, chatLiveRouteReasonSuffix(admission.Reason))
}

func chatLiveRouteTurnCandidateAdmissionAdapterLine(adapter admissionLiveRouteTurnCandidateAdmissionAdapter) string {
	return fmt.Sprintf("│  · live-route candidate admission adapter dry-run: class=%s route=%s source=%s handoff=%s admission_adapter=%s run=%s passed=%t%s",
		adapter.PromptClass, adapter.Route, adapter.Source, adapter.HandoffID,
		adapter.AdmissionAdapterID, adapter.DreamCandidateRunID, adapter.Passed, chatLiveRouteReasonSuffix(adapter.Reason))
}

func chatLiveRouteTurnCandidateAdmissionShadowResultForChain(obs admissionLiveRouteTurnObservation, chain chatLiveRouteTurnCandidateChain) chatLiveRouteTurnCandidateAdmissionShadowResult {
	adapter := chain.AdmissionAdapter
	result := chatLiveRouteTurnCandidateAdmissionShadowResult{}
	if dreamAdmissionMode() != dreamAdmissionShadow {
		result.Reason = "AM_DREAM_ADMISSION must be shadow"
	} else if !dreamAdmissionRequireLiveRoutePlan() {
		result.Reason = "AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN is required"
	} else {
		candidate := admissionLiveRouteTurnCandidateForAdmissionAdapter(chain.Draft, adapter)
		if candidate.Schema == "" {
			result.Reason = "candidate_admission_adapter_failed"
			if adapter.Reason != "" {
				result.Reason += ": " + adapter.Reason
			}
		} else {
			candidate = prepareDreamCandidateForAdmissionWithTurnObservation(NewInnerWorld(), candidate, obs)
			result.Candidate = candidate
			result.Accepted = candidate.Accepted
			result.PolicyPassed = candidate.Admission != nil && candidate.Admission.Checked && candidate.Admission.Passed
			result.Passed = candidate.Schema == "arianna.dream_candidate.v1" &&
				candidate.LiveRouteCandidateAdmission != nil &&
				candidate.LiveRouteCandidateAdmission.AdmissionAdapterID == adapter.AdmissionAdapterID &&
				!candidate.Accepted &&
				result.PolicyPassed
			result.Reason = candidate.Reason
			if !result.PolicyPassed && candidate.Admission != nil && len(candidate.Admission.Reasons) > 0 {
				result.Reason = "admission policy failed: " + strings.Join(candidate.Admission.Reasons, "; ")
			}
		}
	}
	return result
}

func chatLiveRouteTurnCandidateAdmissionShadowLineForResult(chain chatLiveRouteTurnCandidateChain, result chatLiveRouteTurnCandidateAdmissionShadowResult) string {
	adapter := chain.AdmissionAdapter
	return fmt.Sprintf("│  · live-route candidate admission shadow dry-run: class=%s route=%s source=%s handoff=%s admission_adapter=%s run=%s policy=%t accepted=%t passed=%t%s",
		adapter.PromptClass, adapter.Route, adapter.Source, adapter.HandoffID,
		adapter.AdmissionAdapterID, adapter.DreamCandidateRunID, result.PolicyPassed, result.Accepted, result.Passed, chatLiveRouteReasonSuffix(result.Reason))
}

func chatLiveRouteTurnCandidateAdmissionShadowLine(obs admissionLiveRouteTurnObservation, chain chatLiveRouteTurnCandidateChain) string {
	return chatLiveRouteTurnCandidateAdmissionShadowLineForResult(chain, chatLiveRouteTurnCandidateAdmissionShadowResultForChain(obs, chain))
}

func chatLiveRouteTurnCandidateAdmissionDecisionLine(decision admissionLiveRouteTurnCandidateAdmissionDecision) string {
	return fmt.Sprintf("│  · live-route candidate admission decision dry-run: class=%s route=%s source=%s handoff=%s admission_adapter=%s decision=%s decision_id=%s live_ready=%t mutates=%t passed=%t%s",
		decision.PromptClass, decision.Route, decision.Source, decision.HandoffID,
		decision.AdmissionAdapterID, decision.Decision, decision.DecisionID,
		decision.LiveReady, decision.MutatesState, decision.Passed, chatLiveRouteReasonSuffix(decision.Reason))
}

func chatLiveRouteTurnCandidateAdmissionPromotionLine(promotion admissionLiveRouteTurnCandidateAdmissionPromotion) string {
	return fmt.Sprintf("│  · live-route candidate admission promotion dry-run: class=%s route=%s source=%s decision=%s decision_id=%s promotion=%s promotion_id=%s live_ready=%t live_enabled=%t mutates=%t passed=%t%s",
		promotion.PromptClass, promotion.Route, promotion.Source,
		promotion.AdmissionDecision, promotion.AdmissionDecisionID,
		promotion.Promotion, promotion.PromotionID,
		promotion.LiveReady, promotion.LiveAdmissionEnabled, promotion.MutatesState, promotion.Passed,
		chatLiveRouteReasonSuffix(promotion.Reason))
}

func chatLiveRouteTurnCandidateAdmissionSwitchLine(sw admissionLiveRouteTurnCandidateAdmissionSwitch) string {
	return fmt.Sprintf("│  · live-route candidate admission switch dry-run: class=%s route=%s source=%s promotion=%s promotion_id=%s switch=%s switch_action=%s switch_id=%s admission_allowed=%t live_ready=%t live_enabled=%t mutates=%t passed=%t%s",
		sw.PromptClass, sw.Route, sw.Source,
		sw.AdmissionPromotion, sw.AdmissionPromotionID,
		sw.SwitchState, sw.SwitchAction, sw.SwitchID,
		sw.AdmissionAllowed, sw.LiveReady, sw.LiveAdmissionEnabled, sw.MutatesState, sw.Passed,
		chatLiveRouteReasonSuffix(sw.Reason))
}

func chatLiveRouteTurnCandidateAdmissionEnableGateLine(gate admissionLiveRouteTurnCandidateAdmissionEnableGate) string {
	return fmt.Sprintf("│  · live-route candidate admission enable gate dry-run: class=%s route=%s source=%s switch=%s switch_id=%s enable=%s enable_action=%s enable_id=%s admission_allowed=%t manual_enable=%t key_matched=%t live_ready=%t live_enabled=%t mutates=%t passed=%t%s",
		gate.PromptClass, gate.Route, gate.Source,
		gate.SwitchState, gate.AdmissionSwitchID,
		gate.EnableState, gate.EnableAction, gate.EnableGateID,
		gate.AdmissionAllowed, gate.ManualEnableRequested, gate.EnableKeyMatched,
		gate.LiveReady, gate.LiveAdmissionEnabled, gate.MutatesState, gate.Passed,
		chatLiveRouteReasonSuffix(gate.Reason))
}

func chatLiveRouteTurnCandidateAdmissionLiveStageLine(stage admissionLiveRouteTurnCandidateAdmissionLiveStage) string {
	return fmt.Sprintf("│  · live-route candidate admission live stage dry-run: class=%s route=%s source=%s enable=%s enable_id=%s stage=%s stage_action=%s stage_id=%s admission_allowed=%t writer_ready=%t rollback_ready=%t live_ready=%t live_enabled=%t mutates=%t passed=%t%s",
		stage.PromptClass, stage.Route, stage.Source,
		stage.EnableState, stage.AdmissionEnableGateID,
		stage.StageState, stage.StageAction, stage.LiveStageID,
		stage.AdmissionAllowed, stage.WriterReady, stage.RollbackReady,
		stage.LiveReady, stage.LiveAdmissionEnabled, stage.MutatesState, stage.Passed,
		chatLiveRouteReasonSuffix(stage.Reason))
}

func chatLiveRouteTurnCandidateAdmissionWriterPreflightLine(preflight admissionLiveRouteTurnCandidateAdmissionWriterPreflight) string {
	return fmt.Sprintf("│  · live-route candidate admission writer preflight dry-run: class=%s route=%s source=%s stage=%s stage_id=%s writer=%s writer_action=%s rollback=%s rollback_action=%s writer_preflight_id=%s write_allowed=%t admission_allowed=%t live_ready=%t live_enabled=%t mutates=%t passed=%t%s",
		preflight.PromptClass, preflight.Route, preflight.Source,
		preflight.StageState, preflight.AdmissionLiveStageID,
		preflight.WriterState, preflight.WriterAction,
		preflight.RollbackState, preflight.RollbackAction,
		preflight.WriterPreflightID,
		preflight.WriteAllowed, preflight.AdmissionAllowed,
		preflight.LiveReady, preflight.LiveAdmissionEnabled, preflight.MutatesState, preflight.Passed,
		chatLiveRouteReasonSuffix(preflight.Reason))
}

func chatLiveRouteTurnCandidateAdmissionWriterInventoryLine(inventory admissionLiveRouteTurnCandidateAdmissionWriterInventory) string {
	return fmt.Sprintf("│  · live-route candidate admission writer inventory dry-run: class=%s route=%s source=%s writer_preflight=%s inventory=%s inventory_action=%s writer_contract=%s rollback_contract=%s ledger_contract=%s contracts_ready=%t write_allowed=%t admission_allowed=%t live_ready=%t live_enabled=%t mutates=%t writer_inventory_id=%s passed=%t%s",
		inventory.PromptClass, inventory.Route, inventory.Source,
		inventory.AdmissionWriterPreflightID,
		inventory.InventoryState, inventory.InventoryAction,
		inventory.WriterContract, inventory.RollbackContract, inventory.AdmissionLedgerContract,
		inventory.ContractsReady, inventory.WriteAllowed, inventory.AdmissionAllowed,
		inventory.LiveReady, inventory.LiveAdmissionEnabled, inventory.MutatesState,
		inventory.WriterInventoryID, inventory.Passed,
		chatLiveRouteReasonSuffix(inventory.Reason))
}

func chatLiveRouteTurnCandidateAdmissionWriterContractLine(contract admissionLiveRouteTurnCandidateAdmissionWriterContract) string {
	return fmt.Sprintf("│  · live-route candidate admission writer contract dry-run: class=%s route=%s source=%s writer_inventory=%s contract=%s contract_action=%s writer_contract=%s rollback_contract=%s ledger_contract=%s writer_shape=%s rollback_shape=%s ledger_shape=%s shape_ready=%t writer_impl=%t rollback_impl=%t ledger_impl=%t contracts_ready=%t write_allowed=%t admission_allowed=%t live_ready=%t live_enabled=%t mutates=%t writer_contract_id=%s passed=%t%s",
		contract.PromptClass, contract.Route, contract.Source,
		contract.AdmissionWriterInventoryID,
		contract.ContractState, contract.ContractAction,
		contract.WriterContract, contract.RollbackContract, contract.AdmissionLedgerContract,
		contract.WriterContractShape, contract.RollbackContractShape, contract.LedgerContractShape,
		contract.ContractShapeReady,
		contract.WriterImplementationReady, contract.RollbackImplementationReady, contract.LedgerImplementationReady,
		contract.ContractsReady, contract.WriteAllowed, contract.AdmissionAllowed,
		contract.LiveReady, contract.LiveAdmissionEnabled, contract.MutatesState,
		contract.WriterContractID, contract.Passed,
		chatLiveRouteReasonSuffix(contract.Reason))
}

func chatLiveRouteTurnCandidateAdmissionLedgerLine(ledger admissionLiveRouteTurnCandidateAdmissionLedger) string {
	return fmt.Sprintf("│  · live-route candidate admission ledger dry-run: class=%s route=%s source=%s writer_contract=%s ledger=%s ledger_action=%s ledger_contract=%s ledger_mode=%s ledger_entry=%s entry_status=%s receipt_shape=%s append_ready=%t persisted=%t ledger_impl=%t contracts_ready=%t write_allowed=%t admission_allowed=%t live_ready=%t live_enabled=%t mutates=%t admission_ledger_id=%s passed=%t%s",
		ledger.PromptClass, ledger.Route, ledger.Source,
		ledger.AdmissionWriterContractID,
		ledger.LedgerState, ledger.LedgerAction,
		ledger.LedgerContract, ledger.LedgerMode,
		ledger.LedgerEntryKind, ledger.LedgerEntryStatus, ledger.LedgerReceiptShape,
		ledger.LedgerAppendReady, ledger.LedgerReceiptPersisted, ledger.LedgerImplementationReady,
		ledger.ContractsReady, ledger.WriteAllowed, ledger.AdmissionAllowed,
		ledger.LiveReady, ledger.LiveAdmissionEnabled, ledger.MutatesState,
		ledger.AdmissionLedgerID, ledger.Passed,
		chatLiveRouteReasonSuffix(ledger.Reason))
}

func chatLiveRouteTurnCandidateAdmissionWriterImplementationLine(impl admissionLiveRouteTurnCandidateAdmissionWriterImplementation) string {
	return fmt.Sprintf("│  · live-route candidate admission writer implementation dry-run: class=%s route=%s source=%s ledger=%s implementation=%s implementation_action=%s writer_entrypoint=%s ledger_entrypoint=%s rollback_entrypoint=%s write_target=%s body_target=%s append_only=%t rollback_required=%t implementation_contract=%t writer_impl=%t ledger_impl=%t rollback_impl=%t contracts_ready=%t write_allowed=%t admission_allowed=%t live_ready=%t live_enabled=%t mutates=%t writer_implementation_id=%s passed=%t%s",
		impl.PromptClass, impl.Route, impl.Source,
		impl.AdmissionLedgerID,
		impl.ImplementationState, impl.ImplementationAction,
		impl.WriterEntrypoint, impl.LedgerEntrypoint, impl.RollbackEntrypoint,
		impl.WriteTarget, impl.BodyTarget,
		impl.AppendOnly, impl.RollbackRequired, impl.ImplementationContractReady,
		impl.WriterImplementationReady, impl.LedgerImplementationReady, impl.RollbackImplementationReady,
		impl.ContractsReady, impl.WriteAllowed, impl.AdmissionAllowed,
		impl.LiveReady, impl.LiveAdmissionEnabled, impl.MutatesState,
		impl.WriterImplementationID, impl.Passed,
		chatLiveRouteReasonSuffix(impl.Reason))
}

func chatLiveRouteTurnCandidateAdmissionWriterReceiptLine(receipt admissionLiveRouteTurnCandidateAdmissionWriterReceipt) string {
	return fmt.Sprintf("│  · live-route candidate admission writer receipt dry-run: class=%s route=%s source=%s writer_implementation=%s writer_receipt=%s receipt_action=%s receipt_kind=%s receipt_target=%s receipt_mode=%s receipt_shape=%s receipt_persisted=%t shadow_write_allowed=%t body_target=%s append_only=%t rollback_required=%t writer_ready=%t writer_impl=%t ledger_impl=%t rollback_impl=%t contracts_ready=%t write_allowed=%t admission_allowed=%t live_ready=%t live_enabled=%t mutates=%t writer_receipt_id=%s passed=%t%s",
		receipt.PromptClass, receipt.Route, receipt.Source,
		receipt.WriterImplementationID,
		receipt.WriterReceiptState, receipt.WriterReceiptAction,
		receipt.WriterReceiptKind, receipt.WriterReceiptTarget, receipt.WriterReceiptMode, receipt.WriterReceiptShape,
		receipt.WriterReceiptPersisted, receipt.ShadowWriteAllowed,
		receipt.BodyTarget, receipt.AppendOnly, receipt.RollbackRequired,
		receipt.WriterReady, receipt.WriterImplementationReady, receipt.LedgerImplementationReady, receipt.RollbackImplementationReady,
		receipt.ContractsReady, receipt.WriteAllowed, receipt.AdmissionAllowed,
		receipt.LiveReady, receipt.LiveAdmissionEnabled, receipt.MutatesState,
		receipt.WriterReceiptID, receipt.Passed,
		chatLiveRouteReasonSuffix(receipt.Reason))
}

func chatLiveRouteTurnCandidateAdmissionRollbackImplementationLine(rollback admissionLiveRouteTurnCandidateAdmissionRollbackImplementation) string {
	return fmt.Sprintf("│  · live-route candidate admission rollback implementation dry-run: class=%s route=%s source=%s writer_receipt=%s rollback=%s rollback_action=%s rollback_entrypoint=%s rollback_target=%s rollback_target_kind=%s rollback_target_id=%s rollback_mode=%s exact_match=%t dry_run_only=%t receipt_removed=%t writer_ready=%t rollback_ready=%t writer_impl=%t rollback_impl=%t ledger_impl=%t contracts_ready=%t write_allowed=%t admission_allowed=%t live_ready=%t live_enabled=%t mutates=%t rollback_implementation_id=%s passed=%t%s",
		rollback.PromptClass, rollback.Route, rollback.Source,
		rollback.WriterReceiptID,
		rollback.RollbackImplementationState, rollback.RollbackImplementationAction,
		rollback.RollbackEntrypointResolved, rollback.RollbackTarget, rollback.RollbackTargetKind,
		rollback.RollbackTargetID, rollback.RollbackMode,
		rollback.ExactReceiptMatchRequired, rollback.RollbackDryRunOnly, rollback.RollbackReceiptRemoved,
		rollback.WriterReady, rollback.RollbackReady,
		rollback.WriterImplementationReady, rollback.RollbackImplementationReady, rollback.LedgerImplementationReady,
		rollback.ContractsReady, rollback.WriteAllowed, rollback.AdmissionAllowed,
		rollback.LiveReady, rollback.LiveAdmissionEnabled, rollback.MutatesState,
		rollback.RollbackImplementationID, rollback.Passed,
		chatLiveRouteReasonSuffix(rollback.Reason))
}

func chatLiveRouteTurnCandidateAdmissionLedgerImplementationLine(ledger admissionLiveRouteTurnCandidateAdmissionLedgerImplementation) string {
	return fmt.Sprintf("│  · live-route candidate admission ledger implementation dry-run: class=%s route=%s source=%s rollback_implementation=%s ledger=%s ledger_action=%s ledger_entrypoint=%s ledger_target=%s ledger_target_kind=%s ledger_target_mode=%s append_only=%t dry_run_only=%t receipt_persisted=%t writer_ready=%t rollback_ready=%t writer_impl=%t rollback_impl=%t ledger_impl=%t contracts_ready=%t write_allowed=%t admission_allowed=%t live_ready=%t live_enabled=%t mutates=%t ledger_implementation_id=%s passed=%t%s",
		ledger.PromptClass, ledger.Route, ledger.Source,
		ledger.RollbackImplementationID,
		ledger.LedgerImplementationState, ledger.LedgerImplementationAction,
		ledger.LedgerEntrypointResolved, ledger.LedgerImplementationTarget,
		ledger.LedgerImplementationTargetKind, ledger.LedgerImplementationTargetMode,
		ledger.LedgerImplementationAppendOnly, ledger.LedgerImplementationDryRunOnly,
		ledger.LedgerImplementationReceiptPersisted,
		ledger.WriterReady, ledger.RollbackReady,
		ledger.WriterImplementationReady, ledger.RollbackImplementationReady, ledger.LedgerImplementationReady,
		ledger.ContractsReady, ledger.WriteAllowed, ledger.AdmissionAllowed,
		ledger.LiveReady, ledger.LiveAdmissionEnabled, ledger.MutatesState,
		ledger.LedgerImplementationID, ledger.Passed,
		chatLiveRouteReasonSuffix(ledger.Reason))
}

func chatLiveRouteTurnCandidateAdmissionLedgerPersistenceLine(persistence admissionLiveRouteTurnCandidateAdmissionLedgerPersistence) string {
	return fmt.Sprintf("│  · live-route candidate admission ledger persistence dry-run: class=%s route=%s source=%s ledger_implementation=%s admission_ledger=%s writer_receipt=%s rollback_implementation=%s persistence=%s persistence_action=%s persistence_target=%s persistence_target_kind=%s persistence_target_mode=%s receipt_shape=%s append_only=%t dry_run_only=%t receipt_persisted=%t persistence_ready=%t writer_ready=%t rollback_ready=%t writer_impl=%t rollback_impl=%t ledger_impl=%t contracts_ready=%t write_allowed=%t admission_allowed=%t live_ready=%t live_enabled=%t mutates=%t ledger_persistence_id=%s passed=%t%s",
		persistence.PromptClass, persistence.Route, persistence.Source,
		persistence.LedgerImplementationID, persistence.AdmissionLedgerID,
		persistence.WriterReceiptID, persistence.RollbackImplementationID,
		persistence.LedgerPersistenceState, persistence.LedgerPersistenceAction,
		persistence.LedgerPersistenceTarget, persistence.LedgerPersistenceTargetKind,
		persistence.LedgerPersistenceTargetMode, persistence.LedgerPersistenceReceiptShape,
		persistence.LedgerPersistenceAppendOnly, persistence.LedgerPersistenceDryRunOnly,
		persistence.LedgerPersistenceReceiptPersisted, persistence.LedgerPersistenceReady,
		persistence.WriterReady, persistence.RollbackReady,
		persistence.WriterImplementationReady, persistence.RollbackImplementationReady, persistence.LedgerImplementationReady,
		persistence.ContractsReady, persistence.WriteAllowed, persistence.AdmissionAllowed,
		persistence.LiveReady, persistence.LiveAdmissionEnabled, persistence.MutatesState,
		persistence.LedgerPersistenceID, persistence.Passed,
		chatLiveRouteReasonSuffix(persistence.Reason))
}

func chatLiveRouteTurnCandidateAdmissionLedgerVerificationLine(verification admissionLiveRouteTurnCandidateAdmissionLedgerVerification) string {
	return fmt.Sprintf("│  · live-route candidate admission ledger verification dry-run: class=%s route=%s source=%s ledger_persistence=%s ledger_implementation=%s admission_ledger=%s writer_receipt=%s rollback_implementation=%s verification=%s verification_action=%s verification_target=%s verification_target_kind=%s verification_target_mode=%s receipt_shape=%s append_only=%t dry_run_only=%t read_back=%t receipt_verified=%t verification_ready=%t persistence_ready=%t writer_ready=%t rollback_ready=%t writer_impl=%t rollback_impl=%t ledger_impl=%t contracts_ready=%t write_allowed=%t admission_allowed=%t live_ready=%t live_enabled=%t mutates=%t ledger_verification_id=%s passed=%t%s",
		verification.PromptClass, verification.Route, verification.Source,
		verification.LedgerPersistenceID, verification.LedgerImplementationID,
		verification.AdmissionLedgerID, verification.WriterReceiptID, verification.RollbackImplementationID,
		verification.LedgerVerificationState, verification.LedgerVerificationAction,
		verification.LedgerVerificationTarget, verification.LedgerVerificationTargetKind,
		verification.LedgerVerificationTargetMode, verification.LedgerVerificationReceiptShape,
		verification.LedgerVerificationAppendOnly, verification.LedgerVerificationDryRunOnly,
		verification.LedgerVerificationReceiptReadBack, verification.LedgerVerificationReceiptVerified,
		verification.LedgerVerificationReady, verification.LedgerPersistenceReady,
		verification.WriterReady, verification.RollbackReady,
		verification.WriterImplementationReady, verification.RollbackImplementationReady, verification.LedgerImplementationReady,
		verification.ContractsReady, verification.WriteAllowed, verification.AdmissionAllowed,
		verification.LiveReady, verification.LiveAdmissionEnabled, verification.MutatesState,
		verification.LedgerVerificationID, verification.Passed,
		chatLiveRouteReasonSuffix(verification.Reason))
}

func chatLiveRouteTurnCandidateAdmissionReadinessLine(readiness admissionLiveRouteTurnCandidateAdmissionReadiness) string {
	return fmt.Sprintf("│  · live-route candidate admission readiness dry-run: class=%s route=%s source=%s ledger_verification=%s ledger_persistence=%s ledger_implementation=%s admission_ledger=%s writer_receipt=%s rollback_implementation=%s readiness=%s readiness_action=%s readiness_target=%s readiness_target_kind=%s readiness_target_mode=%s dry_run_only=%t ledger_verified=%t writer_ready=%t rollback_ready=%t ledger_ready=%t readiness_ready=%t verification_ready=%t persistence_ready=%t writer_impl=%t rollback_impl=%t ledger_impl=%t contracts_ready=%t write_allowed=%t admission_allowed=%t live_ready=%t live_enabled=%t mutates=%t admission_readiness_id=%s passed=%t%s",
		readiness.PromptClass, readiness.Route, readiness.Source,
		readiness.LedgerVerificationID, readiness.LedgerPersistenceID,
		readiness.LedgerImplementationID, readiness.AdmissionLedgerID,
		readiness.WriterReceiptID, readiness.RollbackImplementationID,
		readiness.AdmissionReadinessState, readiness.AdmissionReadinessAction,
		readiness.AdmissionReadinessTarget, readiness.AdmissionReadinessTargetKind,
		readiness.AdmissionReadinessTargetMode, readiness.AdmissionReadinessDryRunOnly,
		readiness.AdmissionReadinessLedgerVerified, readiness.AdmissionReadinessWriterReady,
		readiness.AdmissionReadinessRollbackReady, readiness.AdmissionReadinessLedgerReady,
		readiness.AdmissionReadinessReady, readiness.LedgerVerificationReady,
		readiness.LedgerPersistenceReady, readiness.WriterImplementationReady,
		readiness.RollbackImplementationReady, readiness.LedgerImplementationReady,
		readiness.ContractsReady, readiness.WriteAllowed, readiness.AdmissionAllowed,
		readiness.LiveReady, readiness.LiveAdmissionEnabled, readiness.MutatesState,
		readiness.AdmissionReadinessID, readiness.Passed,
		chatLiveRouteReasonSuffix(readiness.Reason))
}

func chatLiveRouteTurnCandidateAdmissionPermitLine(permit admissionLiveRouteTurnCandidateAdmissionPermit) string {
	return fmt.Sprintf("│  · live-route candidate admission permit dry-run: class=%s route=%s source=%s readiness=%s ledger_verification=%s ledger_persistence=%s ledger_implementation=%s admission_ledger=%s writer_receipt=%s rollback_implementation=%s permit=%s permit_action=%s permit_target=%s permit_target_kind=%s permit_target_mode=%s dry_run_only=%t readiness_verified=%t ledger_verified=%t writer_ready=%t rollback_ready=%t ledger_ready=%t permit_ready=%t manual_requested=%t key_matched=%t readiness_ready=%t verification_ready=%t persistence_ready=%t writer_impl=%t rollback_impl=%t ledger_impl=%t contracts_ready=%t write_allowed=%t admission_allowed=%t live_ready=%t live_enabled=%t mutates=%t admission_permit_id=%s passed=%t%s",
		permit.PromptClass, permit.Route, permit.Source,
		permit.AdmissionReadinessID, permit.LedgerVerificationID,
		permit.LedgerPersistenceID, permit.LedgerImplementationID,
		permit.AdmissionLedgerID, permit.WriterReceiptID, permit.RollbackImplementationID,
		permit.AdmissionPermitState, permit.AdmissionPermitAction,
		permit.AdmissionPermitTarget, permit.AdmissionPermitTargetKind,
		permit.AdmissionPermitTargetMode, permit.AdmissionPermitDryRunOnly,
		permit.AdmissionPermitReadinessVerified, permit.AdmissionPermitLedgerVerified,
		permit.AdmissionPermitWriterReady, permit.AdmissionPermitRollbackReady,
		permit.AdmissionPermitLedgerReady, permit.AdmissionPermitReady,
		permit.ManualPermitRequested, permit.PermitKeyMatched,
		permit.AdmissionReadinessReady, permit.LedgerVerificationReady,
		permit.LedgerPersistenceReady, permit.WriterImplementationReady,
		permit.RollbackImplementationReady, permit.LedgerImplementationReady,
		permit.ContractsReady, permit.WriteAllowed, permit.AdmissionAllowed,
		permit.LiveReady, permit.LiveAdmissionEnabled, permit.MutatesState,
		permit.AdmissionPermitID, permit.Passed,
		chatLiveRouteReasonSuffix(permit.Reason))
}

func chatLiveRouteTurnCandidateAdmissionSealLine(seal admissionLiveRouteTurnCandidateAdmissionSeal) string {
	return fmt.Sprintf("│  · live-route candidate admission seal dry-run: class=%s route=%s source=%s permit=%s readiness=%s ledger_verification=%s ledger_persistence=%s ledger_implementation=%s admission_ledger=%s writer_receipt=%s rollback_implementation=%s seal=%s seal_action=%s seal_target=%s seal_target_kind=%s seal_target_mode=%s receipt_shape=%s dry_run_only=%t permit_verified=%t readiness_verified=%t ledger_verified=%t writer_ready=%t rollback_ready=%t ledger_ready=%t seal_ready=%t permit_ready=%t key_matched=%t readiness_ready=%t verification_ready=%t persistence_ready=%t writer_impl=%t rollback_impl=%t ledger_impl=%t contracts_ready=%t write_allowed=%t admission_allowed=%t live_ready=%t live_enabled=%t mutates=%t admission_seal_id=%s passed=%t%s",
		seal.PromptClass, seal.Route, seal.Source,
		seal.AdmissionPermitID, seal.AdmissionReadinessID,
		seal.LedgerVerificationID, seal.LedgerPersistenceID,
		seal.LedgerImplementationID, seal.AdmissionLedgerID,
		seal.WriterReceiptID, seal.RollbackImplementationID,
		seal.AdmissionSealState, seal.AdmissionSealAction,
		seal.AdmissionSealTarget, seal.AdmissionSealTargetKind,
		seal.AdmissionSealTargetMode, seal.AdmissionSealReceiptShape,
		seal.AdmissionSealDryRunOnly, seal.AdmissionSealPermitVerified,
		seal.AdmissionSealReadinessVerified, seal.AdmissionSealLedgerVerified,
		seal.AdmissionSealWriterReady, seal.AdmissionSealRollbackReady,
		seal.AdmissionSealLedgerReady, seal.AdmissionSealReady,
		seal.AdmissionPermitReady, seal.PermitKeyMatched,
		seal.AdmissionReadinessReady, seal.LedgerVerificationReady,
		seal.LedgerPersistenceReady, seal.WriterImplementationReady,
		seal.RollbackImplementationReady, seal.LedgerImplementationReady,
		seal.ContractsReady, seal.WriteAllowed, seal.AdmissionAllowed,
		seal.LiveReady, seal.LiveAdmissionEnabled, seal.MutatesState,
		seal.AdmissionSealID, seal.Passed,
		chatLiveRouteReasonSuffix(seal.Reason))
}

func chatLiveRouteTurnCandidateAdmissionFinalGateLine(finalGate admissionLiveRouteTurnCandidateAdmissionFinalGate) string {
	return fmt.Sprintf("│  · live-route candidate admission final gate dry-run: class=%s route=%s source=%s seal=%s permit=%s readiness=%s ledger_verification=%s ledger_persistence=%s ledger_implementation=%s admission_ledger=%s writer_receipt=%s rollback_implementation=%s final_gate=%s final_gate_action=%s final_gate_target=%s final_gate_target_kind=%s final_gate_target_mode=%s receipt_shape=%s dry_run_only=%t seal_verified=%t permit_verified=%t readiness_verified=%t ledger_verified=%t writer_ready=%t rollback_ready=%t ledger_ready=%t final_gate_ready=%t seal_ready=%t permit_ready=%t key_matched=%t readiness_ready=%t verification_ready=%t persistence_ready=%t writer_impl=%t rollback_impl=%t ledger_impl=%t contracts_ready=%t write_allowed=%t admission_allowed=%t live_ready=%t live_enabled=%t mutates=%t admission_final_gate_id=%s passed=%t%s",
		finalGate.PromptClass, finalGate.Route, finalGate.Source,
		finalGate.AdmissionSealID, finalGate.AdmissionPermitID, finalGate.AdmissionReadinessID,
		finalGate.LedgerVerificationID, finalGate.LedgerPersistenceID,
		finalGate.LedgerImplementationID, finalGate.AdmissionLedgerID,
		finalGate.WriterReceiptID, finalGate.RollbackImplementationID,
		finalGate.AdmissionFinalGateState, finalGate.AdmissionFinalGateAction,
		finalGate.AdmissionFinalGateTarget, finalGate.AdmissionFinalGateTargetKind,
		finalGate.AdmissionFinalGateTargetMode, finalGate.AdmissionFinalGateReceiptShape,
		finalGate.AdmissionFinalGateDryRunOnly, finalGate.AdmissionFinalGateSealVerified,
		finalGate.AdmissionFinalGatePermitVerified, finalGate.AdmissionFinalGateReadinessVerified,
		finalGate.AdmissionFinalGateLedgerVerified, finalGate.AdmissionFinalGateWriterReady,
		finalGate.AdmissionFinalGateRollbackReady, finalGate.AdmissionFinalGateLedgerReady,
		finalGate.AdmissionFinalGateReady, finalGate.AdmissionSealReady,
		finalGate.AdmissionPermitReady, finalGate.PermitKeyMatched,
		finalGate.AdmissionReadinessReady, finalGate.LedgerVerificationReady,
		finalGate.LedgerPersistenceReady, finalGate.WriterImplementationReady,
		finalGate.RollbackImplementationReady, finalGate.LedgerImplementationReady,
		finalGate.ContractsReady, finalGate.WriteAllowed, finalGate.AdmissionAllowed,
		finalGate.LiveReady, finalGate.LiveAdmissionEnabled, finalGate.MutatesState,
		finalGate.AdmissionFinalGateID, finalGate.Passed,
		chatLiveRouteReasonSuffix(finalGate.Reason))
}

func chatLiveRouteTurnCandidateAdmissionResonanceIntentLine(intent admissionLiveRouteTurnCandidateAdmissionResonanceIntent) string {
	return fmt.Sprintf("│  · live-route candidate admission resonance intent dry-run: class=%s route=%s source=%s final_gate=%s seal=%s permit=%s readiness=%s ledger_verification=%s receiver=%s receiver_kind=%s influence_kind=%s max_influence=%.2f ttl_turns=%d causal_id=%s raw_text_allowed=%t janus_surface_allowed=%t cooc_learning_allowed=%t delta_harvest_allowed=%t rollback_required=%t pre_hash_required=%t post_hash_required=%t intent=%s intent_action=%s intent_target=%s intent_target_kind=%s intent_target_mode=%s receipt_shape=%s dry_run_only=%t final_gate_verified=%t seal_verified=%t permit_verified=%t readiness_verified=%t ledger_verified=%t writer_ready=%t rollback_ready=%t ledger_ready=%t intent_ready=%t contracts_ready=%t write_allowed=%t admission_allowed=%t live_ready=%t live_enabled=%t mutates=%t admission_resonance_intent_id=%s passed=%t%s",
		intent.PromptClass, intent.Route, intent.Source,
		intent.AdmissionFinalGateID, intent.AdmissionSealID, intent.AdmissionPermitID,
		intent.AdmissionReadinessID, intent.LedgerVerificationID,
		intent.AdmissionResonanceIntentReceiver, intent.AdmissionResonanceIntentReceiverKind,
		intent.AdmissionResonanceIntentInfluenceKind, intent.AdmissionResonanceIntentMaxInfluence,
		intent.AdmissionResonanceIntentTTLTurns, intent.AdmissionResonanceIntentCausalID,
		intent.AdmissionResonanceIntentRawDreamTextAllowed,
		intent.AdmissionResonanceIntentJanusSurfaceAllowed,
		intent.AdmissionResonanceIntentCoocLearningAllowed,
		intent.AdmissionResonanceIntentDeltaHarvestAllowed,
		intent.AdmissionResonanceIntentRollbackRequired,
		intent.AdmissionResonanceIntentPreStateHashRequired,
		intent.AdmissionResonanceIntentPostStateHashRequired,
		intent.AdmissionResonanceIntentState, intent.AdmissionResonanceIntentAction,
		intent.AdmissionResonanceIntentTarget, intent.AdmissionResonanceIntentTargetKind,
		intent.AdmissionResonanceIntentTargetMode, intent.AdmissionResonanceIntentReceiptShape,
		intent.AdmissionResonanceIntentDryRunOnly,
		intent.AdmissionResonanceIntentFinalGateVerified,
		intent.AdmissionResonanceIntentSealVerified,
		intent.AdmissionResonanceIntentPermitVerified,
		intent.AdmissionResonanceIntentReadinessVerified,
		intent.AdmissionResonanceIntentLedgerVerified,
		intent.AdmissionResonanceIntentWriterReady,
		intent.AdmissionResonanceIntentRollbackReady,
		intent.AdmissionResonanceIntentLedgerReady,
		intent.AdmissionResonanceIntentReady,
		intent.ContractsReady, intent.WriteAllowed, intent.AdmissionAllowed,
		intent.LiveReady, intent.LiveAdmissionEnabled, intent.MutatesState,
		intent.AdmissionResonanceIntentID, intent.Passed,
		chatLiveRouteReasonSuffix(intent.Reason))
}

func chatLiveRouteTurnCandidateAdmissionResonanceReceiverLine(receiver admissionLiveRouteTurnCandidateAdmissionResonanceReceiver) string {
	return fmt.Sprintf("│  · live-route candidate admission resonance receiver dry-run: class=%s route=%s source=%s intent=%s final_gate=%s seal=%s permit=%s readiness=%s ledger_verification=%s receiver=%s receiver_kind=%s influence_kind=%s max_influence=%.2f ttl_turns=%d causal_id=%s source_causal_id=%s pre_state_hash=%s post_state_hash=%s delta_hash=%s state_hash_mode=%s raw_text_observed=%t raw_text_forwarded=%t janus_surface_allowed=%t cooc_learning_allowed=%t delta_harvest_allowed=%t body_mutation_allowed=%t rollback_required=%t receiver_state=%s receiver_action=%s receiver_target=%s receiver_target_kind=%s receiver_target_mode=%s receipt_shape=%s dry_run_only=%t intent_verified=%t final_gate_verified=%t seal_verified=%t permit_verified=%t readiness_verified=%t ledger_verified=%t writer_ready=%t rollback_ready=%t ledger_ready=%t receiver_ready=%t contracts_ready=%t write_allowed=%t admission_allowed=%t live_ready=%t live_enabled=%t mutates=%t admission_resonance_receiver_id=%s passed=%t%s",
		receiver.PromptClass, receiver.Route, receiver.Source,
		receiver.AdmissionResonanceIntentID,
		receiver.AdmissionFinalGateID, receiver.AdmissionSealID, receiver.AdmissionPermitID,
		receiver.AdmissionReadinessID, receiver.LedgerVerificationID,
		receiver.AdmissionResonanceReceiverReceiver, receiver.AdmissionResonanceReceiverReceiverKind,
		receiver.AdmissionResonanceReceiverInfluenceKind, receiver.AdmissionResonanceReceiverMaxInfluence,
		receiver.AdmissionResonanceReceiverTTLTurns, receiver.AdmissionResonanceReceiverCausalID,
		receiver.SourceAdmissionResonanceIntentCausalID,
		receiver.AdmissionResonanceReceiverPreStateHash,
		receiver.AdmissionResonanceReceiverPostStateHash,
		receiver.AdmissionResonanceReceiverStateDeltaHash,
		receiver.AdmissionResonanceReceiverStateHashMode,
		receiver.AdmissionResonanceReceiverRawDreamTextObserved,
		receiver.AdmissionResonanceReceiverRawDreamTextForwarded,
		receiver.AdmissionResonanceReceiverJanusSurfaceAllowed,
		receiver.AdmissionResonanceReceiverCoocLearningAllowed,
		receiver.AdmissionResonanceReceiverDeltaHarvestAllowed,
		receiver.AdmissionResonanceReceiverBodyMutationAllowed,
		receiver.AdmissionResonanceReceiverRollbackRequired,
		receiver.AdmissionResonanceReceiverState, receiver.AdmissionResonanceReceiverAction,
		receiver.AdmissionResonanceReceiverTarget, receiver.AdmissionResonanceReceiverTargetKind,
		receiver.AdmissionResonanceReceiverTargetMode, receiver.AdmissionResonanceReceiverReceiptShape,
		receiver.AdmissionResonanceReceiverDryRunOnly,
		receiver.AdmissionResonanceReceiverIntentVerified,
		receiver.AdmissionResonanceReceiverFinalGateVerified,
		receiver.AdmissionResonanceReceiverSealVerified,
		receiver.AdmissionResonanceReceiverPermitVerified,
		receiver.AdmissionResonanceReceiverReadinessVerified,
		receiver.AdmissionResonanceReceiverLedgerVerified,
		receiver.AdmissionResonanceReceiverWriterReady,
		receiver.AdmissionResonanceReceiverRollbackReady,
		receiver.AdmissionResonanceReceiverLedgerReady,
		receiver.AdmissionResonanceReceiverReady,
		receiver.ContractsReady, receiver.WriteAllowed, receiver.AdmissionAllowed,
		receiver.LiveReady, receiver.LiveAdmissionEnabled, receiver.MutatesState,
		receiver.AdmissionResonanceReceiverID, receiver.Passed,
		chatLiveRouteReasonSuffix(receiver.Reason))
}

func chatLiveRouteTurnCandidateAdmissionResonanceObservationLine(observation admissionLiveRouteTurnCandidateAdmissionResonanceObservation) string {
	return fmt.Sprintf("│  · live-route candidate admission resonance observation dry-run: class=%s route=%s source=%s receiver=%s intent=%s final_gate=%s seal=%s permit=%s readiness=%s ledger_verification=%s observer=%s observer_kind=%s observation_kind=%s observation_mode=%s causal_id=%s append_hash=%s read_back_hash=%s source_receiver_causal_id=%s source_receiver_delta_hash=%s append_only=%t read_back=%t receipt_verified=%t raw_text_observed=%t raw_text_forwarded=%t janus_surface_allowed=%t cooc_learning_allowed=%t delta_harvest_allowed=%t body_mutation_allowed=%t rollback_required=%t observation_state=%s observation_action=%s observation_target=%s observation_target_kind=%s observation_target_mode=%s receipt_shape=%s dry_run_only=%t receiver_verified=%t intent_verified=%t final_gate_verified=%t seal_verified=%t permit_verified=%t readiness_verified=%t ledger_verified=%t writer_ready=%t rollback_ready=%t ledger_ready=%t observation_ready=%t contracts_ready=%t write_allowed=%t admission_allowed=%t live_ready=%t live_enabled=%t mutates=%t admission_resonance_observation_id=%s passed=%t%s",
		observation.PromptClass, observation.Route, observation.Source,
		observation.AdmissionResonanceReceiverID, observation.AdmissionResonanceIntentID,
		observation.AdmissionFinalGateID, observation.AdmissionSealID, observation.AdmissionPermitID,
		observation.AdmissionReadinessID, observation.LedgerVerificationID,
		observation.AdmissionResonanceObservationObserver,
		observation.AdmissionResonanceObservationObserverKind,
		observation.AdmissionResonanceObservationKind,
		observation.AdmissionResonanceObservationMode,
		observation.AdmissionResonanceObservationCausalID,
		observation.AdmissionResonanceObservationAppendHash,
		observation.AdmissionResonanceObservationReadBackHash,
		observation.SourceAdmissionResonanceReceiverCausalID,
		observation.SourceAdmissionResonanceReceiverStateDeltaHash,
		observation.AdmissionResonanceObservationAppendOnly,
		observation.AdmissionResonanceObservationReadBack,
		observation.AdmissionResonanceObservationReceiptVerified,
		observation.AdmissionResonanceObservationRawDreamTextObserved,
		observation.AdmissionResonanceObservationRawDreamTextForwarded,
		observation.AdmissionResonanceObservationJanusSurfaceAllowed,
		observation.AdmissionResonanceObservationCoocLearningAllowed,
		observation.AdmissionResonanceObservationDeltaHarvestAllowed,
		observation.AdmissionResonanceObservationBodyMutationAllowed,
		observation.AdmissionResonanceObservationRollbackRequired,
		observation.AdmissionResonanceObservationState,
		observation.AdmissionResonanceObservationAction,
		observation.AdmissionResonanceObservationTarget,
		observation.AdmissionResonanceObservationTargetKind,
		observation.AdmissionResonanceObservationTargetMode,
		observation.AdmissionResonanceObservationReceiptShape,
		observation.AdmissionResonanceObservationDryRunOnly,
		observation.AdmissionResonanceObservationReceiverVerified,
		observation.AdmissionResonanceObservationIntentVerified,
		observation.AdmissionResonanceObservationFinalGateVerified,
		observation.AdmissionResonanceObservationSealVerified,
		observation.AdmissionResonanceObservationPermitVerified,
		observation.AdmissionResonanceObservationReadinessVerified,
		observation.AdmissionResonanceObservationLedgerVerified,
		observation.AdmissionResonanceObservationWriterReady,
		observation.AdmissionResonanceObservationRollbackReady,
		observation.AdmissionResonanceObservationLedgerReady,
		observation.AdmissionResonanceObservationReady,
		observation.ContractsReady, observation.WriteAllowed, observation.AdmissionAllowed,
		observation.LiveReady, observation.LiveAdmissionEnabled, observation.MutatesState,
		observation.AdmissionResonanceObservationID, observation.Passed,
		chatLiveRouteReasonSuffix(observation.Reason))
}

func chatLiveRouteTurnCandidateChainDryRunLines(obs admissionLiveRouteTurnObservation) []string {
	if !chatLiveRouteTurnCandidateChainDryRunNeeded() || obs.Schema == "" {
		return nil
	}
	finalGateDryRun := admissionLiveRouteTurnCandidateAdmissionFinalGateDryRun()
	resonanceIntentDryRun := admissionLiveRouteTurnCandidateAdmissionResonanceIntentDryRun()
	resonanceReceiverDryRun := admissionLiveRouteTurnCandidateAdmissionResonanceReceiverDryRun()
	resonanceObservationDryRun := admissionLiveRouteTurnCandidateAdmissionResonanceObservationDryRun()
	resonanceReceiverNeeded := resonanceReceiverDryRun || resonanceObservationDryRun
	resonanceIntentNeeded := resonanceIntentDryRun || resonanceReceiverNeeded
	finalGateNeeded := finalGateDryRun || resonanceIntentNeeded
	chain := chatLiveRouteTurnCandidateChainForText(obs, chatLiveRouteTurnCandidateChainText())
	lines := []string{}
	if admissionLiveRouteTurnCandidateExecutionDryRun() {
		if err := recordAdmissionLiveRouteTurnCandidateExecution(chain.Execution); err != nil {
			return append(lines, fmt.Sprintf("│  · live-route candidate execution dry-run log failed: %v", err))
		}
		lines = append(lines, chatLiveRouteTurnCandidateExecutionLine(chain.Execution))
	}
	if admissionLiveRouteTurnGeneratorAdapterDryRun() {
		if err := recordAdmissionLiveRouteTurnGeneratorAdapter(chain.Adapter); err != nil {
			return append(lines, fmt.Sprintf("│  · live-route generator adapter dry-run log failed: %v", err))
		}
		lines = append(lines, chatLiveRouteTurnGeneratorAdapterLine(chain.Adapter))
	}
	if admissionLiveRouteTurnCandidateDraftDryRun() {
		if err := recordAdmissionLiveRouteTurnCandidateDraft(chain.Draft); err != nil {
			return append(lines, fmt.Sprintf("│  · live-route candidate draft dry-run log failed: %v", err))
		}
		lines = append(lines, chatLiveRouteTurnCandidateDraftLine(chain.Draft))
	}
	if admissionLiveRouteTurnCandidateAdmissionDryRun() && admissionLiveRouteTurnCandidateDraftDryRun() {
		if err := recordAdmissionLiveRouteTurnCandidateReview(chain.Review); err != nil {
			return append(lines, fmt.Sprintf("│  · live-route candidate admission handoff dry-run review log failed: %v", err))
		}
		if err := recordAdmissionLiveRouteTurnCandidateAdmission(chain.Admission); err != nil {
			return append(lines, fmt.Sprintf("│  · live-route candidate admission handoff dry-run log failed: %v", err))
		}
		lines = append(lines, chatLiveRouteTurnCandidateAdmissionLine(chain.Admission))
	}
	if admissionLiveRouteTurnCandidateAdmissionAdapterDryRun() &&
		admissionLiveRouteTurnCandidateAdmissionDryRun() &&
		admissionLiveRouteTurnCandidateDraftDryRun() {
		if err := recordAdmissionLiveRouteTurnCandidateAdmissionAdapter(chain.AdmissionAdapter); err != nil {
			return append(lines, fmt.Sprintf("│  · live-route candidate admission adapter dry-run log failed: %v", err))
		}
		lines = append(lines, chatLiveRouteTurnCandidateAdmissionAdapterLine(chain.AdmissionAdapter))
	}
	shadow := chatLiveRouteTurnCandidateAdmissionShadowResult{}
	if (admissionLiveRouteTurnCandidateAdmissionShadowDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
		finalGateNeeded) &&
		admissionLiveRouteTurnCandidateAdmissionAdapterDryRun() &&
		admissionLiveRouteTurnCandidateAdmissionDryRun() &&
		admissionLiveRouteTurnCandidateDraftDryRun() {
		shadow = chatLiveRouteTurnCandidateAdmissionShadowResultForChain(obs, chain)
		if admissionLiveRouteTurnCandidateAdmissionShadowDryRun() {
			lines = append(lines, chatLiveRouteTurnCandidateAdmissionShadowLineForResult(chain, shadow))
		}
	}
	decision := admissionLiveRouteTurnCandidateAdmissionDecision{}
	if admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
		finalGateNeeded {
		decision = admissionLiveRouteTurnCandidateAdmissionDecisionForShadow(
			chain.Execution,
			chain.Adapter,
			chain.Draft,
			chain.Admission,
			chain.AdmissionAdapter,
			shadow.Candidate,
		)
		if admissionLiveRouteTurnCandidateAdmissionDecisionDryRun() {
			if err := recordAdmissionLiveRouteTurnCandidateAdmissionDecision(decision); err != nil {
				return append(lines, fmt.Sprintf("│  · live-route candidate admission decision dry-run log failed: %v", err))
			}
			lines = append(lines, chatLiveRouteTurnCandidateAdmissionDecisionLine(decision))
		}
	}
	promotion := admissionLiveRouteTurnCandidateAdmissionPromotion{}
	if admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
		finalGateNeeded {
		promotion = admissionLiveRouteTurnCandidateAdmissionPromotionForDecision(decision)
		if admissionLiveRouteTurnCandidateAdmissionPromotionDryRun() {
			if err := recordAdmissionLiveRouteTurnCandidateAdmissionPromotion(promotion); err != nil {
				return append(lines, fmt.Sprintf("│  · live-route candidate admission promotion dry-run log failed: %v", err))
			}
			lines = append(lines, chatLiveRouteTurnCandidateAdmissionPromotionLine(promotion))
		}
	}
	sw := admissionLiveRouteTurnCandidateAdmissionSwitch{}
	if admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
		finalGateNeeded {
		sw = admissionLiveRouteTurnCandidateAdmissionSwitchForPromotion(promotion)
	}
	if admissionLiveRouteTurnCandidateAdmissionSwitchDryRun() {
		if err := recordAdmissionLiveRouteTurnCandidateAdmissionSwitch(sw); err != nil {
			return append(lines, fmt.Sprintf("│  · live-route candidate admission switch dry-run log failed: %v", err))
		}
		lines = append(lines, chatLiveRouteTurnCandidateAdmissionSwitchLine(sw))
	}
	gate := admissionLiveRouteTurnCandidateAdmissionEnableGate{}
	if admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
		finalGateNeeded {
		gate = admissionLiveRouteTurnCandidateAdmissionEnableGateForSwitch(sw)
	}
	if admissionLiveRouteTurnCandidateAdmissionEnableGateDryRun() {
		if err := recordAdmissionLiveRouteTurnCandidateAdmissionEnableGate(gate); err != nil {
			return append(lines, fmt.Sprintf("│  · live-route candidate admission enable gate dry-run log failed: %v", err))
		}
		lines = append(lines, chatLiveRouteTurnCandidateAdmissionEnableGateLine(gate))
	}
	stage := admissionLiveRouteTurnCandidateAdmissionLiveStage{}
	if admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
		finalGateNeeded {
		stage = admissionLiveRouteTurnCandidateAdmissionLiveStageForEnableGate(gate)
	}
	if admissionLiveRouteTurnCandidateAdmissionLiveStageDryRun() {
		if err := recordAdmissionLiveRouteTurnCandidateAdmissionLiveStage(stage); err != nil {
			return append(lines, fmt.Sprintf("│  · live-route candidate admission live stage dry-run log failed: %v", err))
		}
		lines = append(lines, chatLiveRouteTurnCandidateAdmissionLiveStageLine(stage))
	}
	preflight := admissionLiveRouteTurnCandidateAdmissionWriterPreflight{}
	if admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
		finalGateNeeded {
		preflight = admissionLiveRouteTurnCandidateAdmissionWriterPreflightForLiveStage(stage)
	}
	if admissionLiveRouteTurnCandidateAdmissionWriterPreflightDryRun() {
		if err := recordAdmissionLiveRouteTurnCandidateAdmissionWriterPreflight(preflight); err != nil {
			return append(lines, fmt.Sprintf("│  · live-route candidate admission writer preflight dry-run log failed: %v", err))
		}
		lines = append(lines, chatLiveRouteTurnCandidateAdmissionWriterPreflightLine(preflight))
	}
	inventory := admissionLiveRouteTurnCandidateAdmissionWriterInventory{}
	if admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
		finalGateNeeded {
		inventory = admissionLiveRouteTurnCandidateAdmissionWriterInventoryForPreflight(preflight)
	}
	if admissionLiveRouteTurnCandidateAdmissionWriterInventoryDryRun() {
		if err := recordAdmissionLiveRouteTurnCandidateAdmissionWriterInventory(inventory); err != nil {
			return append(lines, fmt.Sprintf("│  · live-route candidate admission writer inventory dry-run log failed: %v", err))
		}
		lines = append(lines, chatLiveRouteTurnCandidateAdmissionWriterInventoryLine(inventory))
	}
	contract := admissionLiveRouteTurnCandidateAdmissionWriterContract{}
	if admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
		finalGateNeeded {
		contract = admissionLiveRouteTurnCandidateAdmissionWriterContractForInventory(inventory)
	}
	if admissionLiveRouteTurnCandidateAdmissionWriterContractDryRun() {
		if err := recordAdmissionLiveRouteTurnCandidateAdmissionWriterContract(contract); err != nil {
			return append(lines, fmt.Sprintf("│  · live-route candidate admission writer contract dry-run log failed: %v", err))
		}
		lines = append(lines, chatLiveRouteTurnCandidateAdmissionWriterContractLine(contract))
	}
	ledger := admissionLiveRouteTurnCandidateAdmissionLedger{}
	if admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
		finalGateNeeded {
		ledger = admissionLiveRouteTurnCandidateAdmissionLedgerForWriterContract(contract)
	}
	if admissionLiveRouteTurnCandidateAdmissionLedgerDryRun() {
		if err := recordAdmissionLiveRouteTurnCandidateAdmissionLedger(ledger); err != nil {
			return append(lines, fmt.Sprintf("│  · live-route candidate admission ledger dry-run log failed: %v", err))
		}
		lines = append(lines, chatLiveRouteTurnCandidateAdmissionLedgerLine(ledger))
	}
	impl := admissionLiveRouteTurnCandidateAdmissionWriterImplementation{}
	if admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
		finalGateNeeded {
		impl = admissionLiveRouteTurnCandidateAdmissionWriterImplementationForLedger(ledger)
	}
	if admissionLiveRouteTurnCandidateAdmissionWriterImplementationDryRun() {
		if err := recordAdmissionLiveRouteTurnCandidateAdmissionWriterImplementation(impl); err != nil {
			return append(lines, fmt.Sprintf("│  · live-route candidate admission writer implementation dry-run log failed: %v", err))
		}
		lines = append(lines, chatLiveRouteTurnCandidateAdmissionWriterImplementationLine(impl))
	}
	receipt := admissionLiveRouteTurnCandidateAdmissionWriterReceipt{}
	if admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
		finalGateNeeded {
		receipt = admissionLiveRouteTurnCandidateAdmissionWriterReceiptForImplementation(impl)
	}
	if admissionLiveRouteTurnCandidateAdmissionWriterReceiptDryRun() {
		if err := recordAdmissionLiveRouteTurnCandidateAdmissionWriterReceipt(receipt); err != nil {
			return append(lines, fmt.Sprintf("│  · live-route candidate admission writer receipt dry-run log failed: %v", err))
		}
		lines = append(lines, chatLiveRouteTurnCandidateAdmissionWriterReceiptLine(receipt))
	}
	rollback := admissionLiveRouteTurnCandidateAdmissionRollbackImplementation{}
	if admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
		finalGateNeeded {
		rollback = admissionLiveRouteTurnCandidateAdmissionRollbackImplementationForWriterReceipt(receipt)
	}
	if admissionLiveRouteTurnCandidateAdmissionRollbackImplementationDryRun() {
		if err := recordAdmissionLiveRouteTurnCandidateAdmissionRollbackImplementation(rollback); err != nil {
			return append(lines, fmt.Sprintf("│  · live-route candidate admission rollback implementation dry-run log failed: %v", err))
		}
		lines = append(lines, chatLiveRouteTurnCandidateAdmissionRollbackImplementationLine(rollback))
	}
	ledgerImpl := admissionLiveRouteTurnCandidateAdmissionLedgerImplementation{}
	if admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
		finalGateNeeded {
		ledgerImpl = admissionLiveRouteTurnCandidateAdmissionLedgerImplementationForRollbackImplementation(rollback)
	}
	if admissionLiveRouteTurnCandidateAdmissionLedgerImplementationDryRun() {
		if err := recordAdmissionLiveRouteTurnCandidateAdmissionLedgerImplementation(ledgerImpl); err != nil {
			return append(lines, fmt.Sprintf("│  · live-route candidate admission ledger implementation dry-run log failed: %v", err))
		}
		lines = append(lines, chatLiveRouteTurnCandidateAdmissionLedgerImplementationLine(ledgerImpl))
	}
	persistence := admissionLiveRouteTurnCandidateAdmissionLedgerPersistence{}
	if admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
		finalGateNeeded {
		persistence = admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceForLedgerImplementation(ledgerImpl)
	}
	if admissionLiveRouteTurnCandidateAdmissionLedgerPersistenceDryRun() {
		if err := recordAdmissionLiveRouteTurnCandidateAdmissionLedgerPersistence(persistence); err != nil {
			return append(lines, fmt.Sprintf("│  · live-route candidate admission ledger persistence dry-run log failed: %v", err))
		}
		lines = append(lines, chatLiveRouteTurnCandidateAdmissionLedgerPersistenceLine(persistence))
	}
	verification := admissionLiveRouteTurnCandidateAdmissionLedgerVerification{}
	if admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
		finalGateNeeded {
		verification = admissionLiveRouteTurnCandidateAdmissionLedgerVerificationForLedgerPersistence(persistence)
	}
	if admissionLiveRouteTurnCandidateAdmissionLedgerVerificationDryRun() {
		if err := recordAdmissionLiveRouteTurnCandidateAdmissionLedgerVerification(verification); err != nil {
			return append(lines, fmt.Sprintf("│  · live-route candidate admission ledger verification dry-run log failed: %v", err))
		}
		lines = append(lines, chatLiveRouteTurnCandidateAdmissionLedgerVerificationLine(verification))
	}
	if admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
		admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
		finalGateNeeded {
		readiness := admissionLiveRouteTurnCandidateAdmissionReadinessForLedgerVerification(verification)
		if admissionLiveRouteTurnCandidateAdmissionReadinessDryRun() {
			if err := recordAdmissionLiveRouteTurnCandidateAdmissionReadiness(readiness); err != nil {
				return append(lines, fmt.Sprintf("│  · live-route candidate admission readiness dry-run log failed: %v", err))
			}
			lines = append(lines, chatLiveRouteTurnCandidateAdmissionReadinessLine(readiness))
		}
		if admissionLiveRouteTurnCandidateAdmissionPermitDryRun() ||
			admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
			finalGateNeeded {
			permit := admissionLiveRouteTurnCandidateAdmissionPermitForReadiness(readiness)
			if err := recordAdmissionLiveRouteTurnCandidateAdmissionPermit(permit); err != nil {
				return append(lines, fmt.Sprintf("│  · live-route candidate admission permit dry-run log failed: %v", err))
			}
			if admissionLiveRouteTurnCandidateAdmissionPermitDryRun() {
				lines = append(lines, chatLiveRouteTurnCandidateAdmissionPermitLine(permit))
			}
			if admissionLiveRouteTurnCandidateAdmissionSealDryRun() ||
				finalGateNeeded {
				seal := admissionLiveRouteTurnCandidateAdmissionSealForPermit(permit)
				if err := recordAdmissionLiveRouteTurnCandidateAdmissionSeal(seal); err != nil {
					return append(lines, fmt.Sprintf("│  · live-route candidate admission seal dry-run log failed: %v", err))
				}
				if admissionLiveRouteTurnCandidateAdmissionSealDryRun() {
					lines = append(lines, chatLiveRouteTurnCandidateAdmissionSealLine(seal))
				}
				if finalGateNeeded {
					finalGate := admissionLiveRouteTurnCandidateAdmissionFinalGateForSeal(seal)
					if err := recordAdmissionLiveRouteTurnCandidateAdmissionFinalGate(finalGate); err != nil {
						return append(lines, fmt.Sprintf("│  · live-route candidate admission final gate dry-run log failed: %v", err))
					}
					if finalGateDryRun {
						lines = append(lines, chatLiveRouteTurnCandidateAdmissionFinalGateLine(finalGate))
					}
					if resonanceIntentNeeded {
						intent := admissionLiveRouteTurnCandidateAdmissionResonanceIntentForFinalGate(finalGate)
						if resonanceIntentDryRun {
							if err := recordAdmissionLiveRouteTurnCandidateAdmissionResonanceIntent(intent); err != nil {
								return append(lines, fmt.Sprintf("│  · live-route candidate admission resonance intent dry-run log failed: %v", err))
							}
							lines = append(lines, chatLiveRouteTurnCandidateAdmissionResonanceIntentLine(intent))
						}
						if resonanceReceiverNeeded {
							receiver := admissionLiveRouteTurnCandidateAdmissionResonanceReceiverForIntent(intent)
							if resonanceReceiverDryRun {
								if err := recordAdmissionLiveRouteTurnCandidateAdmissionResonanceReceiver(receiver); err != nil {
									return append(lines, fmt.Sprintf("│  · live-route candidate admission resonance receiver dry-run log failed: %v", err))
								}
								lines = append(lines, chatLiveRouteTurnCandidateAdmissionResonanceReceiverLine(receiver))
							}
							if resonanceObservationDryRun {
								observation := admissionLiveRouteTurnCandidateAdmissionResonanceObservationForReceiver(receiver)
								if err := recordAdmissionLiveRouteTurnCandidateAdmissionResonanceObservation(observation); err != nil {
									return append(lines, fmt.Sprintf("│  · live-route candidate admission resonance observation dry-run log failed: %v", err))
								}
								lines = append(lines, chatLiveRouteTurnCandidateAdmissionResonanceObservationLine(observation))
							}
						}
					}
				}
			}
		}
	}
	return lines
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
	if admissionLiveRouteTurnCandidateExecutionRunnerDryRun() {
		return admissionLiveRouteTurnCandidateExecutionForShellViaRunner(shell, text)
	}
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
	return fmt.Sprintf("│  · live-route candidate execution dry-run: class=%s route=%s backend=%s entry=%s frame=%s executor=%s timeout_ms=%d shell=%s execution=%s text=%s runner=%s runner_status=%s passed=%t%s",
		execution.PromptClass, execution.Route, execution.Backend, execution.Entrypoint, execution.PromptFrame,
		execution.Executor, execution.TimeoutMS, execution.ShellID, execution.ExecutionID, execution.GeneratedTextStatus,
		execution.Runner, execution.RunnerStatus, execution.Passed, reason)
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
		execution := chatLiveRouteTurnCandidateExecutionForText(obs, text)
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
