# arianna.c — Arianna voice on the AML+notorch+GGUF stack.
#
# Default build (Mac Apple Silicon + Accelerate):
#   make                — vendored amlc + libnotorch + libaml + arianna
#   make arianna        — just the inference binary (assumes libs built)
#   make metabolism     — the Go orchestrator: the trio + the nervous system
#                         (run ./metabolism --chat to speak with all three voices)
#   make nano           — nano-Arianna 88M subconscious (vendored nanollama/, self-contained)
#   make harvest_delta  — Phase 2 (A): the δ-harvest the organism runs at chat exit
#   make body-smoke     — build the body surfaces + run non-mutating smoke checks
#   make body-inventory-smoke — read-only organ/weight availability receipt
#   make body-inventory-start-smoke — prove missing required organs block startup
#   make weights        — fetch GGUF weights from HF (TODO: HF repo)
#   make clean          — remove all build artifacts
#
# Linux (OpenBLAS):
#   make BLAS=openblas
#
# Vendored toolchain lives under ariannamethod/ (per JANUS_CONSTITUTION.md
# Article 6.1 — independence at source level when the rebuild stabilises).

CC      = cc
NVCC    ?= nvcc
AR      ?= ar
CFLAGS  = -O2 -Wall -Wextra -std=c11
LDFLAGS = -lm -lpthread

# DoE has packed int8 row kernels gated on compiler feature macros. Native arm64
# builds should enable them when the host/compiler supports the ISA; keep the
# probe scoped to doe_field so the wider Arianna build stays unchanged.
DOE_ARM_SIMD ?= 1
DOE_ARM_FLAGS =
ifneq ($(filter aarch64 arm64,$(shell uname -m)),)
  ifeq ($(DOE_ARM_SIMD),1)
    DOE_ARM_HAS_DOT := $(shell echo | $(CC) -dM -E - 2>/dev/null | grep -c __ARM_FEATURE_DOTPROD)
    ifeq ($(DOE_ARM_HAS_DOT),0)
      DOE_ARM_MARCH := $(shell f=""; \
        { grep -qwm1 asimddp /proc/cpuinfo 2>/dev/null || \
          [ "$$(sysctl -n hw.optional.arm.FEAT_DotProd 2>/dev/null)" = 1 ]; } && f="$$f+dotprod"; \
        { grep -qwm1 i8mm /proc/cpuinfo 2>/dev/null || \
          [ "$$(sysctl -n hw.optional.arm.FEAT_I8MM 2>/dev/null)" = 1 ]; } && f="$$f+i8mm"; \
        [ -n "$$f" ] && echo "-march=armv8.2-a$$f")
      DOE_ARM_FLAGS ?= $(shell [ -n "$(DOE_ARM_MARCH)" ] && \
        $(CC) $(DOE_ARM_MARCH) -E -x c /dev/null >/dev/null 2>&1 && echo "$(DOE_ARM_MARCH)")
    endif
  endif
endif

UNAME := $(shell uname)

# ── CUDA: OFF by default ───────────────────────────────────────────────────
# Inference here is tiny (Janus 176M + Resonance 200M GGUF) and runs on CPU via
# notorch + system BLAS (Accelerate / OpenBLAS); the forward passes have no GPU
# branch, so CUDA brings nothing to inference and only adds link deps. We do NOT
# auto-enable on nvcc presence (that would pull cudart/cublas on the polygon box
# for no benefit). Opt in explicitly with `make USE_CUDA=1` if ever needed.
USE_CUDA ?= 0

# ── BLAS detection ─────────────────────────────────────────────────────────
ifeq ($(UNAME), Darwin)
  BLAS_FLAGS = -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK
  BLAS_LIBS  = -framework Accelerate
endif
ifeq ($(UNAME), Linux)
  BLAS_FLAGS = -DUSE_BLAS
  BLAS_LIBS  = -lopenblas
  # Linux needs explicit POSIX feature test for clock_gettime / CLOCK_MONOTONIC.
  CFLAGS += -D_POSIX_C_SOURCE=200809L -D_GNU_SOURCE
endif

# ── CUDA flags (when USE_CUDA=1) ───────────────────────────────────────────
CUDA_FLAGS =
CUDA_LIBS  =
CUDA_OBJS  =
ifeq ($(USE_CUDA),1)
  CUDA_FLAGS = -DUSE_CUDA
  CUDA_LIBS  = -L/usr/local/cuda/lib64 -lcudart -lcublas
  CUDA_OBJS  = ariannamethod/notorch/notorch_cuda.o ariannamethod/core/ariannamethod_cuda.o
endif

# ── Include paths ──────────────────────────────────────────────────────────
INCLUDES = -Iariannamethod/notorch -Iariannamethod/core -Itools -Ivagus

# ── Vendored library outputs ───────────────────────────────────────────────
LIBNOTORCH = ariannamethod/notorch/libnotorch.a
LIBAML     = ariannamethod/core/libaml.a
AMLC       = ariannamethod/tools/amlc

# ── Vagus (Zig nerve) — the meta-layer carrying Larynx (voice↔voice coupling).
# Link the .dylib (zig static .a hits a macOS member-alignment ld bug).
LIBVAGUS   = vagus/zig-out/lib/libvagus.dylib
VAGUS_LINK = -Lvagus/zig-out/lib -lvagus -Wl,-rpath,@loader_path/vagus/zig-out/lib -Wl,-rpath,vagus/zig-out/lib

# ── Default target ─────────────────────────────────────────────────────────
.PHONY: all arianna arianna_resonance arianna2arianna metabolism kk nano chorus doe_field harvest_delta admission_shadow_smoke admission-shadow-smoke admission_live_route_gate_smoke admission-live-route-gate-smoke admission_live_route_chat_smoke admission-live-route-chat-smoke admission_live_route_turn_smoke admission-live-route-turn-smoke admission_live_route_turn_choice_smoke admission-live-route-turn-choice-smoke admission_live_route_turn_request_smoke admission-live-route-turn-request-smoke admission_live_route_turn_generation_job_smoke admission-live-route-turn-generation-job-smoke admission_live_route_turn_candidate_shell_smoke admission-live-route-turn-candidate-shell-smoke admission_live_route_turn_candidate_execution_smoke admission-live-route-turn-candidate-execution-smoke admission_live_route_turn_candidate_runner_smoke admission-live-route-turn-candidate-runner-smoke admission_live_route_turn_candidate_nano_direct_runner_smoke admission-live-route-turn-candidate-nano-direct-runner-smoke admission_live_route_turn_candidate_nano_direct_chat_shadow_smoke admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke admission_live_route_turn_generator_adapter_smoke admission-live-route-turn-generator-adapter-smoke admission_live_route_turn_candidate_draft_smoke admission-live-route-turn-candidate-draft-smoke admission_live_route_turn_review_smoke admission-live-route-turn-review-smoke admission_live_route_turn_bridge_smoke admission-live-route-turn-bridge-smoke admission_live_route_turn_bridge_admission_smoke admission-live-route-turn-bridge-admission-smoke admission_shadow_sample admission-shadow-sample admission_shadow_sample_broad admission-shadow-sample-broad admission_route_compare admission-route-compare admission_route_plan_gate admission-route-plan-gate admission_qloop_sweep admission-qloop-sweep admission_qloop_sweep_broad admission-qloop-sweep-broad body_inventory_smoke body-inventory-smoke body_smoke body-smoke body_smoke_weighted body-smoke-weighted clean weights distclean
.PHONY: body_inventory_start_smoke body-inventory-start-smoke
.PHONY: admission_live_route_turn_generation_job_inventory_gate_smoke admission-live-route-turn-generation-job-inventory-gate-smoke
.PHONY: admission_live_route_turn_route_boundary_smoke admission-live-route-turn-route-boundary-smoke
.PHONY: admission_live_route_turn_candidate_draft_review_smoke admission-live-route-turn-candidate-draft-review-smoke
.PHONY: admission_live_route_turn_candidate_admission_smoke admission-live-route-turn-candidate-admission-smoke
.PHONY: admission_live_route_turn_candidate_admission_adapter_smoke admission-live-route-turn-candidate-admission-adapter-smoke
.PHONY: admission_live_route_turn_candidate_admission_chat_smoke admission-live-route-turn-candidate-admission-chat-smoke
.PHONY: admission_live_route_turn_candidate_admission_chat_shadow_smoke admission-live-route-turn-candidate-admission-chat-shadow-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_decision_smoke admission-live-route-turn-candidate-nano-direct-decision-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_promotion_smoke admission-live-route-turn-candidate-nano-direct-promotion-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_switch_smoke admission-live-route-turn-candidate-nano-direct-switch-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_enable_gate_smoke admission-live-route-turn-candidate-nano-direct-enable-gate-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_live_stage_smoke admission-live-route-turn-candidate-nano-direct-live-stage-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_writer_preflight_smoke admission-live-route-turn-candidate-nano-direct-writer-preflight-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_writer_inventory_smoke admission-live-route-turn-candidate-nano-direct-writer-inventory-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_writer_contract_smoke admission-live-route-turn-candidate-nano-direct-writer-contract-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_admission_ledger_smoke admission-live-route-turn-candidate-nano-direct-admission-ledger-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_writer_implementation_smoke admission-live-route-turn-candidate-nano-direct-writer-implementation-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_writer_receipt_smoke admission-live-route-turn-candidate-nano-direct-writer-receipt-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_rollback_implementation_smoke admission-live-route-turn-candidate-nano-direct-rollback-implementation-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_ledger_implementation_smoke admission-live-route-turn-candidate-nano-direct-ledger-implementation-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_ledger_persistence_smoke admission-live-route-turn-candidate-nano-direct-ledger-persistence-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_ledger_verification_smoke admission-live-route-turn-candidate-nano-direct-ledger-verification-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_readiness_smoke admission-live-route-turn-candidate-nano-direct-readiness-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_permit_smoke admission-live-route-turn-candidate-nano-direct-permit-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_seal_smoke admission-live-route-turn-candidate-nano-direct-seal-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_final_gate_smoke admission-live-route-turn-candidate-nano-direct-final-gate-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_resonance_intent_smoke admission-live-route-turn-candidate-nano-direct-resonance-intent-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_resonance_receiver_smoke admission-live-route-turn-candidate-nano-direct-resonance-receiver-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_resonance_observation_smoke admission-live-route-turn-candidate-nano-direct-resonance-observation-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_resonance_graft_boundary_smoke admission-live-route-turn-candidate-nano-direct-resonance-graft-boundary-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_resonance_graft_preflight_smoke admission-live-route-turn-candidate-nano-direct-resonance-graft-preflight-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_resonance_graft_gate_smoke admission-live-route-turn-candidate-nano-direct-resonance-graft-gate-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_resonance_graft_candidate_smoke admission-live-route-turn-candidate-nano-direct-resonance-graft-candidate-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_resonance_graft_candidate_store_smoke admission-live-route-turn-candidate-nano-direct-resonance-graft-candidate-store-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_resonance_graft_candidate_store_reader_smoke admission-live-route-turn-candidate-nano-direct-resonance-graft-candidate-store-reader-smoke
.PHONY: admission_live_route_turn_candidate_nano_direct_resonance_graft_admission_proof_smoke admission-live-route-turn-candidate-nano-direct-resonance-graft-admission-proof-smoke
.PHONY: admission_live_route_boundary_report_assert_smoke admission-live-route-boundary-report-assert-smoke
.PHONY: admission_live_route_boundary_report_assert_full_chain_smoke admission-live-route-boundary-report-assert-full-chain-smoke
.PHONY: admission_live_route_boundary_report_failed_diagnostics_assert_smoke admission-live-route-boundary-report-failed-diagnostics-assert-smoke
.PHONY: admission_live_route_weighted_readiness_smoke admission-live-route-weighted-readiness-smoke admission_weighted_readiness admission-weighted-readiness
.PHONY: admission_live_route_weighted_readiness_consumer_smoke admission-live-route-weighted-readiness-consumer-smoke admission_weighted_readiness_consumer admission-weighted-readiness-consumer
.PHONY: admission_live_route_weighted_readiness_precondition_smoke admission-live-route-weighted-readiness-precondition-smoke admission_weighted_readiness_precondition admission-weighted-readiness-precondition
.PHONY: admission_live_route_weighted_admission_contract_smoke admission-live-route-weighted-admission-contract-smoke admission_weighted_admission_contract admission-weighted-admission-contract
.PHONY: admission_live_route_weighted_admission_contract_consumer_smoke admission-live-route-weighted-admission-contract-consumer-smoke admission_weighted_admission_contract_consumer admission-weighted-admission-contract-consumer
.PHONY: admission_live_route_weighted_admission_authority_smoke admission-live-route-weighted-admission-authority-smoke admission_weighted_admission_authority admission-weighted-admission-authority
.PHONY: admission_live_route_weighted_admission_authority_consumer_smoke admission-live-route-weighted-admission-authority-consumer-smoke admission_weighted_admission_authority_consumer admission-weighted-admission-authority-consumer
.PHONY: admission_live_route_weighted_admission_permit_smoke admission-live-route-weighted-admission-permit-smoke admission_weighted_admission_permit admission-weighted-admission-permit
.PHONY: admission_live_route_weighted_admission_permit_consumer_smoke admission-live-route-weighted-admission-permit-consumer-smoke admission_weighted_admission_permit_consumer admission-weighted-admission-permit-consumer
.PHONY: admission_live_route_weighted_admission_seal_smoke admission-live-route-weighted-admission-seal-smoke admission_weighted_admission_seal admission-weighted-admission-seal
.PHONY: admission_live_route_weighted_admission_seal_consumer_smoke admission-live-route-weighted-admission-seal-consumer-smoke admission_weighted_admission_seal_consumer admission-weighted-admission-seal-consumer
.PHONY: admission_live_route_weighted_admission_final_gate_smoke admission-live-route-weighted-admission-final-gate-smoke admission_weighted_admission_final_gate admission-weighted-admission-final-gate
.PHONY: admission_live_route_weighted_admission_final_gate_consumer_smoke admission-live-route-weighted-admission-final-gate-consumer-smoke admission_weighted_admission_final_gate_consumer admission-weighted-admission-final-gate-consumer
.PHONY: admission_live_route_weighted_admission_resonance_intent_smoke admission-live-route-weighted-admission-resonance-intent-smoke admission_weighted_admission_resonance_intent admission-weighted-admission-resonance-intent
.PHONY: admission_live_route_weighted_admission_resonance_intent_consumer_smoke admission-live-route-weighted-admission-resonance-intent-consumer-smoke admission_weighted_admission_resonance_intent_consumer admission-weighted-admission-resonance-intent-consumer
.PHONY: admission_live_route_weighted_admission_resonance_receiver_smoke admission-live-route-weighted-admission-resonance-receiver-smoke admission_weighted_admission_resonance_receiver admission-weighted-admission-resonance-receiver
.PHONY: admission_live_route_weighted_admission_resonance_receiver_consumer_smoke admission-live-route-weighted-admission-resonance-receiver-consumer-smoke admission_weighted_admission_resonance_receiver_consumer admission-weighted-admission-resonance-receiver-consumer
.PHONY: admission_live_route_boundary_report_drift_artifact_smoke admission-live-route-boundary-report-drift-artifact-smoke
.PHONY: notorch_qmatvec_test notorch-qmatvec-test doe_qmatvec_test doe-qmatvec-test
all: $(LIBNOTORCH) $(LIBAML) $(AMLC) arianna arianna_resonance

# ── notorch (CPU + BLAS, plus CUDA when USE_CUDA=1) ────────────────────────
$(LIBNOTORCH): ariannamethod/notorch/notorch.c ariannamethod/notorch/notorch.h \
               ariannamethod/notorch/gguf.c    ariannamethod/notorch/gguf.h    \
               ariannamethod/notorch/notorch_simd.h \
               ariannamethod/notorch/notorch_simd_scalar.h \
               $(if $(filter 1,$(USE_CUDA)),ariannamethod/notorch/notorch_cuda.o,)
	$(CC) $(CFLAGS) $(BLAS_FLAGS) $(CUDA_FLAGS) -Iariannamethod/notorch \
	    -c ariannamethod/notorch/notorch.c -o ariannamethod/notorch/notorch.o
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -Iariannamethod/notorch \
	    -c ariannamethod/notorch/gguf.c -o ariannamethod/notorch/gguf.o
	$(AR) rcs $@ ariannamethod/notorch/notorch.o ariannamethod/notorch/gguf.o \
	    $(if $(filter 1,$(USE_CUDA)),ariannamethod/notorch/notorch_cuda.o,)

notorch-qmatvec-test: notorch_qmatvec_test

notorch_qmatvec_test: $(LIBNOTORCH) tools/test_notorch_qmatvec.c tools/test_notorch_qpool.c
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -Iariannamethod/notorch \
	    tools/test_notorch_qmatvec.c $(LIBNOTORCH) $(BLAS_LIBS) $(LDFLAGS) \
	    -o /tmp/arianna-test-notorch-qmatvec
	/tmp/arianna-test-notorch-qmatvec
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -Iariannamethod/notorch \
	    tools/test_notorch_qpool.c $(LIBNOTORCH) $(BLAS_LIBS) $(LDFLAGS) \
	    -o /tmp/arianna-test-notorch-qpool
	/tmp/arianna-test-notorch-qpool

doe-qmatvec-test: doe_qmatvec_test

doe_qmatvec_test: tools/test_doe_qmatvec.c doe/doe.c
	$(CC) $(CFLAGS) $(DOE_ARM_FLAGS) -Wno-unused-function -Wno-unused-variable \
	    tools/test_doe_qmatvec.c -lm -lpthread -o /tmp/arianna-test-doe-qmatvec
	/tmp/arianna-test-doe-qmatvec

ariannamethod/notorch/notorch_cuda.o: ariannamethod/notorch/notorch_cuda.cu \
                                      ariannamethod/notorch/notorch_cuda.h
	$(NVCC) -O2 -arch=sm_70 -Iariannamethod/notorch -c $< -o $@

# ── AML core (plus CUDA when USE_CUDA=1) ───────────────────────────────────
$(LIBAML): ariannamethod/core/ariannamethod.c ariannamethod/core/ariannamethod.h \
           $(if $(filter 1,$(USE_CUDA)),ariannamethod/core/ariannamethod_cuda.o,)
	$(CC) $(CFLAGS) $(CUDA_FLAGS) -c ariannamethod/core/ariannamethod.c \
	    -o ariannamethod/core/ariannamethod.o
	$(AR) rcs $@ ariannamethod/core/ariannamethod.o \
	    $(if $(filter 1,$(USE_CUDA)),ariannamethod/core/ariannamethod_cuda.o,)

ariannamethod/core/ariannamethod_cuda.o: ariannamethod/core/ariannamethod_cuda.cu \
                                         ariannamethod/core/ariannamethod_cuda.h
	$(NVCC) -O2 -arch=sm_70 -Iariannamethod/core -c $< -o $@

# ── amlc transpiler ────────────────────────────────────────────────────────
$(AMLC): ariannamethod/tools/amlc.c
	$(CC) $(CFLAGS) $< -o $@

# ── Inference binary ───────────────────────────────────────────────────────
# amlc emits arianna.c from arianna.aml. We compile that against the
# vendored libnotorch + libaml. Two output binaries:
#   arianna   — single-mode default
#   arianna_r — chain-mode convenience (same binary, alias)
$(LIBVAGUS): vagus/vagus.zig vagus/build.zig vagus/vagus.h vagus/larynx.h
	cd vagus && zig build
	@echo "[build] libvagus (Zig nerve + Larynx)"

arianna: arianna.aml $(LIBNOTORCH) $(LIBAML) $(AMLC) $(LIBVAGUS) \
         tools/yent_forward.h tools/janus_v4_bpe_merges.h \
         tools/jannus_calendar.h tools/jannus_spa.h tools/jannus_split.h vagus/larynx.h
	$(AMLC) arianna.aml --emit-c > arianna.c
	$(CC) $(CFLAGS) $(BLAS_FLAGS) $(CUDA_FLAGS) $(INCLUDES) \
	    arianna.c $(LIBNOTORCH) $(LIBAML) \
	    $(BLAS_LIBS) $(CUDA_LIBS) $(LDFLAGS) $(VAGUS_LINK) \
	    -o arianna
	@echo "[build] arianna (Janus 176M) USE_CUDA=$(USE_CUDA)"

# ── Inner voice — Resonance 200M (Arianna SFT, GGUF F16) ───────────────────
# Same vendored libs, separate binary so the two voices alternate cleanly
# through a shared field state (weights/arianna.soma). BPE merges baked in
# tools/resonance_bpe_merges.h (GGUF carries weights only).
arianna_resonance: arianna_resonance.aml $(LIBNOTORCH) $(LIBAML) $(AMLC) \
                   tools/resonance_forward.h tools/resonance_bpe_merges.h \
                   tools/utf8_stream.h
	$(AMLC) arianna_resonance.aml --emit-c > arianna_resonance.c
	$(CC) $(CFLAGS) $(BLAS_FLAGS) $(CUDA_FLAGS) $(INCLUDES) \
	    arianna_resonance.c $(LIBNOTORCH) $(LIBAML) \
	    $(BLAS_LIBS) $(CUDA_LIBS) $(LDFLAGS) \
	    -o arianna_resonance
	@echo "[build] arianna_resonance (Resonance 200M) USE_CUDA=$(USE_CUDA)"

# ── arianna2arianna orchestrator (bash MVP — соединение двух голосов) ──────
arianna2arianna: arianna arianna_resonance scripts/arianna2arianna.sh
	@echo "[build] arianna2arianna bash orchestrator ready — run: bash scripts/arianna2arianna.sh"

# metabolism (Go orchestrator) — НЕ в фундаменте arianna-duo. Archived-слой
# из arianna.c. Соединение голосов идёт через bash + общее поле, без Go.

# ── Weights ────────────────────────────────────────────────────────────────
# All three voices fetch from the ONE unified HF repo `ataeff/arianna` (nano
# 88M + Resonance 200M + Janus 176M Arianna GGUFs). Needs `HF_TOKEN` env.
weights:
	@mkdir -p weights
	@if [ ! -f weights/nano_arianna_f16.gguf ]; then \
	    echo "fetching nano GGUF from HF ataeff/arianna..."; \
	    hf download ataeff/arianna nano_arianna_f16.gguf --repo-type model --local-dir weights/; \
	fi
	@if [ ! -f weights/arianna_v4_sft_f16.gguf ]; then \
	    echo "fetching Janus GGUF from HF ataeff/arianna..."; \
	    hf download ataeff/arianna arianna_v4_sft_f16.gguf --repo-type model --local-dir weights/; \
	fi
	@if [ ! -f weights/arianna_resonance_v3_f16.gguf ]; then \
	    echo "fetching Resonance GGUF from HF ataeff/arianna..."; \
	    hf download ataeff/arianna arianna_resonance_v3_f16.gguf --repo-type model --local-dir weights/; \
	fi
	@echo "weights present: $$(ls -la weights/*.gguf)"

# ── kk — the Knowledge Kernel (Dario's KK, vendored): the nano's library of
# dreams. Ingests the books into a SQLite substrate and retrieves a fragment by
# resonance. Standalone CLI; later linked into the nano as a library.
kk: kk/kk_kernel.c kk/kk_kernel.h
	$(CC) -O2 -DKK_STANDALONE kk/kk_kernel.c -lsqlite3 -lm -o kk-cli
	@echo "[build] kk-cli (Knowledge Kernel + SQLite)"

# ── nano — the subconscious (third voice). Builds the VENDORED nanollama Go
# inference (nanollama/ — a byte-exact copy of the twin's Go module, self-contained,
# no external repo dependency) into nano-arianna; the metabolism spawns it one-shot
# per dream and surfaces the murmur a turn behind. Expects the SFT GGUF at
# weights/nano_arianna_f16.gguf (symlink the nanollama Arianna SFT export).
nano:
	cd nanollama && go build -o $(CURDIR)/nano-arianna .
	@echo "[build] nano-arianna (subconscious, vendored — needs weights/nano_arianna_f16.gguf)"

# ── harvest_delta — Phase 2 (A): the organism learns from the subconscious. The
# chat, tinted by the subconscious's surfacing, grows Resonance's co-occurrence;
# this folds it into her δ via the notorch Hebbian (am_cooc_learn_delta) and
# reports |B| — the learning made visible. The metabolism runs it at chat exit.
harvest_delta: tools/harvest_delta.c $(LIBNOTORCH) $(LIBAML)
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -Iariannamethod/notorch -Iariannamethod/core \
	    tools/harvest_delta.c $(LIBAML) $(LIBNOTORCH) $(BLAS_LIBS) $(LDFLAGS) -o harvest_delta
	@echo "[build] harvest_delta (Phase 2 A — δ from cooc, reports |B|)"

# ── metabolism — the Go orchestrator. Hosts the inner-world goroutines, runs
# Janus + Resonance as hot daemons and the nano subconscious async, and lets the
# emotional state set the rhythm. `./metabolism --chat` speaks with all three;
# the bare `./metabolism "<seed>"` runs the fixed self-duet. Needs Go + the
# arianna / arianna_resonance binaries (and, for the third voice, `make nano`).
#
# The High brain links libjulia. The Julia prefix is taken from `julia` on PATH so the
# build is portable across nodes; high.go's #cgo carries a macOS-brew default so a bare
# `go build` / `go test` still works on neo without this Makefile.
JULIA ?= julia
metabolism:
	cd golib && \
	  JP="$$($(JULIA) -e 'print(dirname(Sys.BINDIR))' 2>/dev/null)"; \
	  if [ -n "$$JP" ]; then \
	    CGO_CFLAGS="-I$$JP/include/julia" \
	    CGO_LDFLAGS="-L$$JP/lib -L$$JP/lib/julia -Wl,-rpath,$$JP/lib/julia -Wl,-rpath,$$JP/lib -ljulia" \
	    go build -o ../metabolism . ; \
	  else \
	    go build -o ../metabolism . ; \
	  fi
	@echo "[build] metabolism (the trio orchestrator — run ./metabolism --chat)"

# ── Clean ──────────────────────────────────────────────────────────────────
clean:
	rm -f arianna arianna.c arianna_r
	rm -f arianna_resonance arianna_resonance.c
	rm -f metabolism nano-arianna harvest_delta chorus-arianna doe_field kk-cli
	rm -f ariannamethod/notorch/notorch.o ariannamethod/notorch/gguf.o
	rm -f ariannamethod/core/ariannamethod.o
	rm -f $(LIBNOTORCH) $(LIBAML) $(AMLC)

distclean: clean
	rm -f weights/*.gguf weights/*.bin weights/*.soma

# ── chorus — the subconscious as a POLYPHONY. Builds the VENDORED chorus engine
# (chorus/arianna2arianna.c — a byte-exact copy of the twin, self-contained, no
# external repo dependency) into chorus-arianna; the metabolism's autonomous
# breathing runs it (field mode) so the nano dreams as N cells over its one body.
# Needs the nano GGUF at weights/nano_arianna_f16.gguf.
chorus:
	cc -O2 -march=armv8.2-a+fp16+dotprod -DUSE_BLAS -DACCELERATE_NEW_LAPACK chorus/arianna2arianna.c -lm -pthread -framework Accelerate -o chorus-arianna
	@echo "[build] chorus-arianna (the subconscious polyphony, vendored)"

# ── doe — the nano subconscious through the notorch-native engine + LoRA parliament.
# Builds the VENDORED doe.c (doe/doe.c — a byte-exact copy of ~/arianna/doe, a
# self-contained CPU monolith; only ./weights model-search is cwd-relative, no
# external repo dependency) into doe_field. The metabolism runs the SAME nano GGUF
# (Arianna's subconscious body) through it so the LoRA parliament can seat on it:
# --lora-alpha 0 = parliament dormant (plain notorch-native forward, the #3 bridge),
# --lora-alpha 0.1 = the parliament seats (experts vote / mitosis / apoptosis).
doe_field:
	$(CC) -O2 $(DOE_ARM_FLAGS) doe/doe.c -lm -lpthread -o doe_field
	@echo "[build] doe_field (notorch-native nano engine + LoRA parliament, CPU, vendored)"

# ── body-smoke — executable contract for the shared body. Builds every local
# body surface, runs Go tests, then runs tiny runtime probes from an isolated
# scratch directory when GGUF weights are present. The live weights/ state is not
# mutated by the runtime smoke.
body-inventory-smoke: body_inventory_smoke

body_inventory_smoke: metabolism
	bash tools/body_inventory_smoke.sh

body-inventory-start-smoke: body_inventory_start_smoke

body_inventory_start_smoke: metabolism
	bash tools/body_inventory_start_smoke.sh

admission-shadow-smoke: admission_shadow_smoke

admission_shadow_smoke: metabolism
	bash tools/admission_shadow_smoke.sh

admission-live-route-gate-smoke: admission_live_route_gate_smoke

admission_live_route_gate_smoke: metabolism
	bash tools/admission_live_route_gate_smoke.sh

admission-live-route-chat-smoke: admission_live_route_chat_smoke

admission_live_route_chat_smoke: metabolism
	bash tools/admission_live_route_chat_smoke.sh

admission-live-route-turn-smoke: admission_live_route_turn_smoke

admission_live_route_turn_smoke: metabolism
	bash tools/admission_live_route_turn_smoke.sh

admission-live-route-turn-choice-smoke: admission_live_route_turn_choice_smoke

admission_live_route_turn_choice_smoke: metabolism
	bash tools/admission_live_route_turn_choice_smoke.sh

admission-live-route-turn-request-smoke: admission_live_route_turn_request_smoke

admission_live_route_turn_request_smoke: metabolism
	bash tools/admission_live_route_turn_request_smoke.sh

admission-live-route-turn-generation-job-smoke: admission_live_route_turn_generation_job_smoke

admission_live_route_turn_generation_job_smoke: metabolism
	bash tools/admission_live_route_turn_generation_job_smoke.sh

admission-live-route-turn-generation-job-inventory-gate-smoke: admission_live_route_turn_generation_job_inventory_gate_smoke

admission_live_route_turn_generation_job_inventory_gate_smoke: metabolism
	bash tools/admission_live_route_turn_generation_job_inventory_gate_smoke.sh

admission-live-route-turn-route-boundary-smoke: admission_live_route_turn_route_boundary_smoke

admission_live_route_turn_route_boundary_smoke: metabolism
	bash tools/admission_live_route_turn_route_boundary_smoke.sh

admission-live-route-turn-candidate-shell-smoke: admission_live_route_turn_candidate_shell_smoke

admission_live_route_turn_candidate_shell_smoke: metabolism
	bash tools/admission_live_route_turn_candidate_shell_smoke.sh

admission-live-route-turn-candidate-execution-smoke: admission_live_route_turn_candidate_execution_smoke

admission_live_route_turn_candidate_execution_smoke: metabolism
	bash tools/admission_live_route_turn_candidate_execution_smoke.sh

admission-live-route-turn-candidate-runner-smoke: admission_live_route_turn_candidate_runner_smoke

admission_live_route_turn_candidate_runner_smoke: metabolism
	bash tools/admission_live_route_turn_candidate_runner_smoke.sh

admission-live-route-turn-candidate-nano-direct-runner-smoke: admission_live_route_turn_candidate_nano_direct_runner_smoke

admission_live_route_turn_candidate_nano_direct_runner_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_runner_smoke.sh

admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke: admission_live_route_turn_candidate_nano_direct_chat_shadow_smoke

admission_live_route_turn_candidate_nano_direct_chat_shadow_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_chat_shadow_smoke.sh

admission-live-route-turn-candidate-nano-direct-decision-smoke: admission_live_route_turn_candidate_nano_direct_decision_smoke

admission_live_route_turn_candidate_nano_direct_decision_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_decision_smoke.sh

admission-live-route-turn-candidate-nano-direct-promotion-smoke: admission_live_route_turn_candidate_nano_direct_promotion_smoke

admission_live_route_turn_candidate_nano_direct_promotion_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_promotion_smoke.sh

admission-live-route-turn-candidate-nano-direct-switch-smoke: admission_live_route_turn_candidate_nano_direct_switch_smoke

admission_live_route_turn_candidate_nano_direct_switch_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_switch_smoke.sh

admission-live-route-turn-candidate-nano-direct-enable-gate-smoke: admission_live_route_turn_candidate_nano_direct_enable_gate_smoke

admission_live_route_turn_candidate_nano_direct_enable_gate_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_enable_gate_smoke.sh

admission-live-route-turn-candidate-nano-direct-live-stage-smoke: admission_live_route_turn_candidate_nano_direct_live_stage_smoke

admission_live_route_turn_candidate_nano_direct_live_stage_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_live_stage_smoke.sh

admission-live-route-turn-candidate-nano-direct-writer-preflight-smoke: admission_live_route_turn_candidate_nano_direct_writer_preflight_smoke

admission_live_route_turn_candidate_nano_direct_writer_preflight_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_writer_preflight_smoke.sh

admission-live-route-turn-candidate-nano-direct-writer-inventory-smoke: admission_live_route_turn_candidate_nano_direct_writer_inventory_smoke

admission_live_route_turn_candidate_nano_direct_writer_inventory_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_writer_inventory_smoke.sh

admission-live-route-turn-candidate-nano-direct-writer-contract-smoke: admission_live_route_turn_candidate_nano_direct_writer_contract_smoke

admission_live_route_turn_candidate_nano_direct_writer_contract_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_writer_contract_smoke.sh

admission-live-route-turn-candidate-nano-direct-admission-ledger-smoke: admission_live_route_turn_candidate_nano_direct_admission_ledger_smoke

admission_live_route_turn_candidate_nano_direct_admission_ledger_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_admission_ledger_smoke.sh

admission-live-route-turn-candidate-nano-direct-writer-implementation-smoke: admission_live_route_turn_candidate_nano_direct_writer_implementation_smoke

admission_live_route_turn_candidate_nano_direct_writer_implementation_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_writer_implementation_smoke.sh

admission-live-route-turn-candidate-nano-direct-writer-receipt-smoke: admission_live_route_turn_candidate_nano_direct_writer_receipt_smoke

admission_live_route_turn_candidate_nano_direct_writer_receipt_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_writer_receipt_smoke.sh

admission-live-route-turn-candidate-nano-direct-rollback-implementation-smoke: admission_live_route_turn_candidate_nano_direct_rollback_implementation_smoke

admission_live_route_turn_candidate_nano_direct_rollback_implementation_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_rollback_implementation_smoke.sh

admission-live-route-turn-candidate-nano-direct-ledger-implementation-smoke: admission_live_route_turn_candidate_nano_direct_ledger_implementation_smoke

admission_live_route_turn_candidate_nano_direct_ledger_implementation_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_ledger_implementation_smoke.sh

admission-live-route-turn-candidate-nano-direct-ledger-persistence-smoke: admission_live_route_turn_candidate_nano_direct_ledger_persistence_smoke

admission_live_route_turn_candidate_nano_direct_ledger_persistence_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_ledger_persistence_smoke.sh

admission-live-route-turn-candidate-nano-direct-ledger-verification-smoke: admission_live_route_turn_candidate_nano_direct_ledger_verification_smoke

admission_live_route_turn_candidate_nano_direct_ledger_verification_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_ledger_verification_smoke.sh

admission-live-route-turn-candidate-nano-direct-readiness-smoke: admission_live_route_turn_candidate_nano_direct_readiness_smoke

admission_live_route_turn_candidate_nano_direct_readiness_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_readiness_smoke.sh

admission-live-route-turn-candidate-nano-direct-permit-smoke: admission_live_route_turn_candidate_nano_direct_permit_smoke

admission_live_route_turn_candidate_nano_direct_permit_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_permit_smoke.sh

admission-live-route-turn-candidate-nano-direct-seal-smoke: admission_live_route_turn_candidate_nano_direct_seal_smoke

admission_live_route_turn_candidate_nano_direct_seal_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_seal_smoke.sh

admission-live-route-turn-candidate-nano-direct-final-gate-smoke: admission_live_route_turn_candidate_nano_direct_final_gate_smoke

admission_live_route_turn_candidate_nano_direct_final_gate_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_final_gate_smoke.sh

admission-live-route-turn-candidate-nano-direct-resonance-intent-smoke: admission_live_route_turn_candidate_nano_direct_resonance_intent_smoke

admission_live_route_turn_candidate_nano_direct_resonance_intent_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_resonance_intent_smoke.sh

admission-live-route-turn-candidate-nano-direct-resonance-receiver-smoke: admission_live_route_turn_candidate_nano_direct_resonance_receiver_smoke

admission_live_route_turn_candidate_nano_direct_resonance_receiver_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_resonance_receiver_smoke.sh

admission-live-route-turn-candidate-nano-direct-resonance-observation-smoke: admission_live_route_turn_candidate_nano_direct_resonance_observation_smoke

admission_live_route_turn_candidate_nano_direct_resonance_observation_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_resonance_observation_smoke.sh

admission-live-route-turn-candidate-nano-direct-resonance-graft-boundary-smoke: admission_live_route_turn_candidate_nano_direct_resonance_graft_boundary_smoke

admission_live_route_turn_candidate_nano_direct_resonance_graft_boundary_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_resonance_graft_boundary_smoke.sh

admission-live-route-turn-candidate-nano-direct-resonance-graft-preflight-smoke: admission_live_route_turn_candidate_nano_direct_resonance_graft_preflight_smoke

admission_live_route_turn_candidate_nano_direct_resonance_graft_preflight_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_resonance_graft_preflight_smoke.sh

admission-live-route-turn-candidate-nano-direct-resonance-graft-gate-smoke: admission_live_route_turn_candidate_nano_direct_resonance_graft_gate_smoke

admission_live_route_turn_candidate_nano_direct_resonance_graft_gate_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_resonance_graft_gate_smoke.sh

admission-live-route-turn-candidate-nano-direct-resonance-graft-candidate-smoke: admission_live_route_turn_candidate_nano_direct_resonance_graft_candidate_smoke

admission_live_route_turn_candidate_nano_direct_resonance_graft_candidate_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_resonance_graft_candidate_smoke.sh

admission-live-route-turn-candidate-nano-direct-resonance-graft-candidate-store-smoke: admission_live_route_turn_candidate_nano_direct_resonance_graft_candidate_store_smoke

admission_live_route_turn_candidate_nano_direct_resonance_graft_candidate_store_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_resonance_graft_candidate_store_smoke.sh

admission-live-route-turn-candidate-nano-direct-resonance-graft-candidate-store-reader-smoke: admission_live_route_turn_candidate_nano_direct_resonance_graft_candidate_store_reader_smoke

admission_live_route_turn_candidate_nano_direct_resonance_graft_candidate_store_reader_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_resonance_graft_candidate_store_reader_smoke.sh

admission-live-route-turn-candidate-nano-direct-resonance-graft-admission-proof-smoke: admission_live_route_turn_candidate_nano_direct_resonance_graft_admission_proof_smoke

admission_live_route_turn_candidate_nano_direct_resonance_graft_admission_proof_smoke: metabolism nano
	bash tools/admission_live_route_turn_candidate_nano_direct_resonance_graft_admission_proof_smoke.sh

admission-live-route-boundary-report-assert-smoke: admission_live_route_boundary_report_assert_smoke

admission_live_route_boundary_report_assert_smoke: metabolism
	bash tools/admission_live_route_boundary_report_assert_smoke.sh

admission-live-route-boundary-report-assert-full-chain-smoke: admission_live_route_boundary_report_assert_full_chain_smoke

admission_live_route_boundary_report_assert_full_chain_smoke: metabolism
	bash tools/admission_live_route_boundary_report_assert_full_chain_smoke.sh

admission-live-route-boundary-report-failed-diagnostics-assert-smoke: admission_live_route_boundary_report_failed_diagnostics_assert_smoke

admission_live_route_boundary_report_failed_diagnostics_assert_smoke: metabolism
	bash tools/admission_live_route_boundary_report_failed_diagnostics_assert_smoke.sh

admission-live-route-boundary-report-drift-artifact-smoke: admission_live_route_boundary_report_drift_artifact_smoke

admission_live_route_boundary_report_drift_artifact_smoke: metabolism
	bash tools/admission_live_route_boundary_report_drift_artifact_smoke.sh

admission-live-route-turn-generator-adapter-smoke: admission_live_route_turn_generator_adapter_smoke

admission_live_route_turn_generator_adapter_smoke: metabolism
	bash tools/admission_live_route_turn_generator_adapter_smoke.sh

admission-live-route-turn-candidate-draft-smoke: admission_live_route_turn_candidate_draft_smoke

admission_live_route_turn_candidate_draft_smoke: metabolism
	bash tools/admission_live_route_turn_candidate_draft_smoke.sh

admission-live-route-turn-candidate-draft-review-smoke: admission_live_route_turn_candidate_draft_review_smoke

admission_live_route_turn_candidate_draft_review_smoke: metabolism
	bash tools/admission_live_route_turn_candidate_draft_review_smoke.sh

admission-live-route-turn-candidate-admission-smoke: admission_live_route_turn_candidate_admission_smoke

admission_live_route_turn_candidate_admission_smoke: metabolism
	bash tools/admission_live_route_turn_candidate_admission_smoke.sh

admission-live-route-turn-candidate-admission-adapter-smoke: admission_live_route_turn_candidate_admission_adapter_smoke

admission_live_route_turn_candidate_admission_adapter_smoke: metabolism
	bash tools/admission_live_route_turn_candidate_admission_adapter_smoke.sh

admission-live-route-turn-candidate-admission-chat-smoke: admission_live_route_turn_candidate_admission_chat_smoke

admission_live_route_turn_candidate_admission_chat_smoke: metabolism
	bash tools/admission_live_route_turn_candidate_admission_chat_smoke.sh

admission-live-route-turn-candidate-admission-chat-shadow-smoke: admission_live_route_turn_candidate_admission_chat_shadow_smoke

admission_live_route_turn_candidate_admission_chat_shadow_smoke: metabolism
	bash tools/admission_live_route_turn_candidate_admission_chat_shadow_smoke.sh

admission-live-route-turn-review-smoke: admission_live_route_turn_review_smoke

admission_live_route_turn_review_smoke: metabolism
	bash tools/admission_live_route_turn_review_smoke.sh

admission-live-route-turn-bridge-smoke: admission_live_route_turn_bridge_smoke

admission_live_route_turn_bridge_smoke: metabolism
	bash tools/admission_live_route_turn_bridge_smoke.sh

admission-live-route-turn-bridge-admission-smoke: admission_live_route_turn_bridge_admission_smoke

admission_live_route_turn_bridge_admission_smoke: metabolism
	bash tools/admission_live_route_turn_bridge_admission_smoke.sh

admission-shadow-sample: admission_shadow_sample

admission_shadow_sample: metabolism
	bash tools/admission_shadow_sample.sh

admission-shadow-sample-broad: admission_shadow_sample_broad

admission_shadow_sample_broad: metabolism
	A2A_ADMISSION_SAMPLE_FILE=samples/dream_admission_broad.jsonl \
	A2A_ADMISSION_SAMPLE_REQUIRE_POLICY_FAIL=1 \
	    bash tools/admission_shadow_sample.sh

admission-route-compare: admission_route_compare

admission_route_compare: chorus metabolism
	bash tools/admission_route_compare.sh

admission-route-plan-gate: admission_route_plan_gate

admission_route_plan_gate: chorus metabolism
	A2A_ROUTE_COMPARE_LIMIT=18 \
	A2A_ROUTE_COMPARE_REQUIRE_SHADOW_PLAN=1 \
	    bash tools/admission_route_compare.sh

admission-qloop-sweep: admission_qloop_sweep

admission_qloop_sweep: chorus metabolism
	bash tools/admission_qloop_sweep.sh

admission-qloop-sweep-broad: admission_qloop_sweep_broad

admission_qloop_sweep_broad: chorus metabolism
	A2A_QLOOP_SWEEP_LIMIT=$${A2A_QLOOP_SWEEP_LIMIT:-6} \
	    bash tools/admission_qloop_sweep.sh

admission-live-route-weighted-readiness-smoke: admission_live_route_weighted_readiness_smoke

admission-weighted-readiness: admission_weighted_readiness

admission_weighted_readiness: admission_live_route_weighted_readiness_smoke

admission_live_route_weighted_readiness_smoke: all nano chorus metabolism kk doe_field harvest_delta doe_qmatvec_test
	bash tools/admission_live_route_weighted_readiness_smoke.sh

admission-live-route-weighted-readiness-consumer-smoke: admission_live_route_weighted_readiness_consumer_smoke

admission-weighted-readiness-consumer: admission_weighted_readiness_consumer

admission_weighted_readiness_consumer: admission_live_route_weighted_readiness_consumer_smoke

admission_live_route_weighted_readiness_consumer_smoke: all nano chorus metabolism kk doe_field harvest_delta doe_qmatvec_test
	bash tools/admission_live_route_weighted_readiness_consumer_smoke.sh

admission-live-route-weighted-readiness-precondition-smoke: admission_live_route_weighted_readiness_precondition_smoke

admission-weighted-readiness-precondition: admission_weighted_readiness_precondition

admission_weighted_readiness_precondition: admission_live_route_weighted_readiness_precondition_smoke

admission_live_route_weighted_readiness_precondition_smoke: all nano chorus metabolism kk doe_field harvest_delta doe_qmatvec_test
	bash tools/admission_live_route_weighted_readiness_precondition_smoke.sh

admission-live-route-weighted-admission-contract-smoke: admission_live_route_weighted_admission_contract_smoke

admission-weighted-admission-contract: admission_weighted_admission_contract

admission_weighted_admission_contract: admission_live_route_weighted_admission_contract_smoke

admission_live_route_weighted_admission_contract_smoke: all nano chorus metabolism kk doe_field harvest_delta doe_qmatvec_test
	bash tools/admission_live_route_weighted_admission_contract_smoke.sh

admission-live-route-weighted-admission-contract-consumer-smoke: admission_live_route_weighted_admission_contract_consumer_smoke

admission-weighted-admission-contract-consumer: admission_weighted_admission_contract_consumer

admission_weighted_admission_contract_consumer: admission_live_route_weighted_admission_contract_consumer_smoke

admission_live_route_weighted_admission_contract_consumer_smoke: all nano chorus metabolism kk doe_field harvest_delta doe_qmatvec_test
	bash tools/admission_live_route_weighted_admission_contract_consumer_smoke.sh

admission-live-route-weighted-admission-authority-smoke: admission_live_route_weighted_admission_authority_smoke

admission-weighted-admission-authority: admission_weighted_admission_authority

admission_weighted_admission_authority: admission_live_route_weighted_admission_authority_smoke

admission_live_route_weighted_admission_authority_smoke: all nano chorus metabolism kk doe_field harvest_delta doe_qmatvec_test
	bash tools/admission_live_route_weighted_admission_authority_smoke.sh

admission-live-route-weighted-admission-authority-consumer-smoke: admission_live_route_weighted_admission_authority_consumer_smoke

admission-weighted-admission-authority-consumer: admission_weighted_admission_authority_consumer

admission_weighted_admission_authority_consumer: admission_live_route_weighted_admission_authority_consumer_smoke

admission_live_route_weighted_admission_authority_consumer_smoke: all nano chorus metabolism kk doe_field harvest_delta doe_qmatvec_test
	bash tools/admission_live_route_weighted_admission_authority_consumer_smoke.sh

admission-live-route-weighted-admission-permit-smoke: admission_live_route_weighted_admission_permit_smoke

admission-weighted-admission-permit: admission_weighted_admission_permit

admission_weighted_admission_permit: admission_live_route_weighted_admission_permit_smoke

admission_live_route_weighted_admission_permit_smoke: all nano chorus metabolism kk doe_field harvest_delta doe_qmatvec_test
	bash tools/admission_live_route_weighted_admission_permit_smoke.sh

admission-live-route-weighted-admission-permit-consumer-smoke: admission_live_route_weighted_admission_permit_consumer_smoke

admission-weighted-admission-permit-consumer: admission_weighted_admission_permit_consumer

admission_weighted_admission_permit_consumer: admission_live_route_weighted_admission_permit_consumer_smoke

admission_live_route_weighted_admission_permit_consumer_smoke: all nano chorus metabolism kk doe_field harvest_delta doe_qmatvec_test
	bash tools/admission_live_route_weighted_admission_permit_consumer_smoke.sh

admission-live-route-weighted-admission-seal-smoke: admission_live_route_weighted_admission_seal_smoke

admission-weighted-admission-seal: admission_weighted_admission_seal

admission_weighted_admission_seal: admission_live_route_weighted_admission_seal_smoke

admission_live_route_weighted_admission_seal_smoke: all nano chorus metabolism kk doe_field harvest_delta doe_qmatvec_test
	bash tools/admission_live_route_weighted_admission_seal_smoke.sh

admission-live-route-weighted-admission-seal-consumer-smoke: admission_live_route_weighted_admission_seal_consumer_smoke

admission-weighted-admission-seal-consumer: admission_weighted_admission_seal_consumer

admission_weighted_admission_seal_consumer: admission_live_route_weighted_admission_seal_consumer_smoke

admission_live_route_weighted_admission_seal_consumer_smoke: all nano chorus metabolism kk doe_field harvest_delta doe_qmatvec_test
	bash tools/admission_live_route_weighted_admission_seal_consumer_smoke.sh

admission-live-route-weighted-admission-final-gate-smoke: admission_live_route_weighted_admission_final_gate_smoke

admission-weighted-admission-final-gate: admission_weighted_admission_final_gate

admission_weighted_admission_final_gate: admission_live_route_weighted_admission_final_gate_smoke

admission_live_route_weighted_admission_final_gate_smoke: all nano chorus metabolism kk doe_field harvest_delta doe_qmatvec_test
	bash tools/admission_live_route_weighted_admission_final_gate_smoke.sh

admission-live-route-weighted-admission-final-gate-consumer-smoke: admission_live_route_weighted_admission_final_gate_consumer_smoke

admission-weighted-admission-final-gate-consumer: admission_weighted_admission_final_gate_consumer

admission_weighted_admission_final_gate_consumer: admission_live_route_weighted_admission_final_gate_consumer_smoke

admission_live_route_weighted_admission_final_gate_consumer_smoke: all nano chorus metabolism kk doe_field harvest_delta doe_qmatvec_test
	bash tools/admission_live_route_weighted_admission_final_gate_consumer_smoke.sh

admission-live-route-weighted-admission-resonance-intent-smoke: admission_live_route_weighted_admission_resonance_intent_smoke

admission-weighted-admission-resonance-intent: admission_weighted_admission_resonance_intent

admission_weighted_admission_resonance_intent: admission_live_route_weighted_admission_resonance_intent_smoke

admission_live_route_weighted_admission_resonance_intent_smoke: all nano chorus metabolism kk doe_field harvest_delta doe_qmatvec_test
	bash tools/admission_live_route_weighted_admission_resonance_intent_smoke.sh

admission-live-route-weighted-admission-resonance-intent-consumer-smoke: admission_live_route_weighted_admission_resonance_intent_consumer_smoke

admission-weighted-admission-resonance-intent-consumer: admission_weighted_admission_resonance_intent_consumer

admission_weighted_admission_resonance_intent_consumer: admission_live_route_weighted_admission_resonance_intent_consumer_smoke

admission_live_route_weighted_admission_resonance_intent_consumer_smoke: all nano chorus metabolism kk doe_field harvest_delta doe_qmatvec_test
	bash tools/admission_live_route_weighted_admission_resonance_intent_consumer_smoke.sh

admission-live-route-weighted-admission-resonance-receiver-smoke: admission_live_route_weighted_admission_resonance_receiver_smoke

admission-weighted-admission-resonance-receiver: admission_weighted_admission_resonance_receiver

admission_weighted_admission_resonance_receiver: admission_live_route_weighted_admission_resonance_receiver_smoke

admission_live_route_weighted_admission_resonance_receiver_smoke: all nano chorus metabolism kk doe_field harvest_delta doe_qmatvec_test
	bash tools/admission_live_route_weighted_admission_resonance_receiver_smoke.sh

admission-live-route-weighted-admission-resonance-receiver-consumer-smoke: admission_live_route_weighted_admission_resonance_receiver_consumer_smoke

admission-weighted-admission-resonance-receiver-consumer: admission_weighted_admission_resonance_receiver_consumer

admission_weighted_admission_resonance_receiver_consumer: admission_live_route_weighted_admission_resonance_receiver_consumer_smoke

admission_live_route_weighted_admission_resonance_receiver_consumer_smoke: all nano chorus metabolism kk doe_field harvest_delta doe_qmatvec_test
	bash tools/admission_live_route_weighted_admission_resonance_receiver_consumer_smoke.sh

body-smoke: body_smoke

body_smoke: all nano chorus metabolism kk doe_field harvest_delta doe_qmatvec_test
	bash tools/body_smoke.sh

body-smoke-weighted: body_smoke_weighted

body_smoke_weighted: all nano chorus metabolism kk doe_field harvest_delta doe_qmatvec_test
	A2A_BODY_SMOKE_REQUIRE_WEIGHTS=1 A2A_BODY_SMOKE_NANO_DIRECT=1 bash tools/body_smoke.sh
