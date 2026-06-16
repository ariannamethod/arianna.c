# arianna.c — Arianna voice on the AML+notorch+GGUF stack.
#
# Default build (Mac Apple Silicon + Accelerate):
#   make                — vendored amlc + libnotorch + libaml + arianna
#   make arianna        — just the inference binary (assumes libs built)
#   make metabolism     — the Go orchestrator: the trio + the nervous system
#                         (run ./metabolism --chat to speak with all three voices)
#   make nano           — nano-Arianna 88M subconscious (needs the nanollama sibling)
#   make harvest_delta  — Phase 2 (A): the δ-harvest the organism runs at chat exit
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
.PHONY: all arianna arianna_resonance arianna2arianna metabolism kk nano chorus harvest_delta clean weights distclean
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
# TODO: replace local-copy expectation with HF repo `ataeff/arianna-c-tongue`
# fetch (Janus 176M Arianna GGUF + Resonance 200M Arianna GGUF) from the private
# HF repo ataeff/arianna2arianna. Needs `HF_TOKEN` env (repo is private).
weights:
	@mkdir -p weights
	@if [ ! -f weights/arianna_v4_sft_f16.gguf ]; then \
	    echo "fetching Janus GGUF from HF ataeff/arianna2arianna..."; \
	    hf download ataeff/arianna2arianna arianna_v4_sft_f16.gguf --repo-type model --local-dir weights/; \
	fi
	@if [ ! -f weights/arianna_resonance_v3_f16.gguf ]; then \
	    echo "fetching Resonance GGUF from HF ataeff/arianna2arianna..."; \
	    hf download ataeff/arianna2arianna arianna_resonance_v3_f16.gguf --repo-type model --local-dir weights/; \
	fi
	@echo "weights present: $$(ls -la weights/*.gguf)"

# ── kk — the Knowledge Kernel (Dario's KK, vendored): the nano's library of
# dreams. Ingests the books into a SQLite substrate and retrieves a fragment by
# resonance. Standalone CLI; later linked into the nano as a library.
kk: kk/kk_kernel.c kk/kk_kernel.h
	$(CC) -O2 -DKK_STANDALONE kk/kk_kernel.c -lsqlite3 -lm -o kk-cli
	@echo "[build] kk-cli (Knowledge Kernel + SQLite)"

# ── nano — the subconscious (third voice). Builds the nanollama Go inference
# (sibling repo) into nano-arianna; the metabolism spawns it one-shot per dream
# and surfaces the murmur a turn behind. Expects the SFT GGUF at
# weights/nano_arianna_f16.gguf (symlink the nanollama Arianna SFT export).
NANOLLAMA_DIR ?= ../nanollama/go
nano:
	cd $(NANOLLAMA_DIR) && go build -o $(CURDIR)/nano-arianna .
	@echo "[build] nano-arianna (subconscious — needs weights/nano_arianna_f16.gguf)"

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
metabolism:
	cd golib && go build -o ../metabolism .
	@echo "[build] metabolism (the trio orchestrator — run ./metabolism --chat)"

# ── Clean ──────────────────────────────────────────────────────────────────
clean:
	rm -f arianna arianna.c arianna_r
	rm -f arianna_resonance arianna_resonance.c
	rm -f metabolism_bin nano-arianna harvest_delta
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
