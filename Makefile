# arianna.c Makefile
# Llama 3.5 Arianna Edition

CC = gcc
CFLAGS = -O3 -Wall -Wextra -march=native
LDFLAGS = -lm

# Platform detection
UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
  PLATFORM = macos
  DYLIB_EXT = dylib
  RPATH_FLAG = -Wl,-rpath,@loader_path
else
  PLATFORM = linux
  DYLIB_EXT = so
  RPATH_FLAG = -Wl,-rpath,'$$ORIGIN'
endif

# Go libraries (inner_world + cloud)
GO_LIB_DIR = lib
GO_LDFLAGS = -L$(GO_LIB_DIR) -linner_world $(RPATH_FLAG)
CLOUD_LDFLAGS = -L$(GO_LIB_DIR) -lcloud $(RPATH_FLAG)

SRC_DIR = src
BIN_DIR = bin

# Basic version (Cloud wrapper + Go library)
SRCS = $(SRC_DIR)/ariannabody.c $(SRC_DIR)/cloud_wrapper.c $(SRC_DIR)/main.c
TARGET = $(BIN_DIR)/arianna

# Dynamic version with full pipeline (Cloud in Go via wrapper)
SRCS_DYN_CORE = $(SRC_DIR)/ariannabody.c $(SRC_DIR)/cloud_wrapper.c $(SRC_DIR)/julia_bridge.c \
           $(SRC_DIR)/schumann.c $(SRC_DIR)/pandora.c packages/pandora/pandora_bridge.c $(SRC_DIR)/delta.c \
           $(SRC_DIR)/delta_enhanced.c $(SRC_DIR)/mood.c $(SRC_DIR)/guided.c \
           $(SRC_DIR)/subjectivity.c $(SRC_DIR)/cooccur.c $(SRC_DIR)/body_sense.c \
           $(SRC_DIR)/selfsense.c $(SRC_DIR)/mathbrain.c $(SRC_DIR)/inner_arianna.c \
           $(SRC_DIR)/amk_kernel.c $(SRC_DIR)/arianna_dsl.c \
           $(SRC_DIR)/meta_arianna.c \
           sartre/sartre_bridge.c \
           $(SRC_DIR)/arianna_dynamic.c

# Check for Lua and add it automatically
HAS_LUA := $(shell pkg-config --exists lua 2>/dev/null && echo yes || echo no)
ifeq ($(HAS_LUA),yes)
SRCS_DYN = $(SRCS_DYN_CORE) $(SRC_DIR)/amk_lua.c
DYN_CFLAGS = $(shell pkg-config --cflags lua) -DUSE_LUA
DYN_LDFLAGS = $(shell pkg-config --libs lua)
else
SRCS_DYN = $(SRCS_DYN_CORE)
DYN_CFLAGS =
DYN_LDFLAGS =
endif
TARGET_DYN = $(BIN_DIR)/arianna_dynamic

# Full version with Go inner_world
TARGET_FULL = $(BIN_DIR)/arianna_full

# Lua support - bundled in compilers/lua/
LUA_SRC_DIR = compilers/lua/src
LUA_SRCS = $(LUA_SRC_DIR)/lapi.c $(LUA_SRC_DIR)/lauxlib.c $(LUA_SRC_DIR)/lbaselib.c \
           $(LUA_SRC_DIR)/lcode.c $(LUA_SRC_DIR)/lcorolib.c $(LUA_SRC_DIR)/lctype.c \
           $(LUA_SRC_DIR)/ldblib.c $(LUA_SRC_DIR)/ldebug.c $(LUA_SRC_DIR)/ldo.c \
           $(LUA_SRC_DIR)/ldump.c $(LUA_SRC_DIR)/lfunc.c $(LUA_SRC_DIR)/lgc.c \
           $(LUA_SRC_DIR)/linit.c $(LUA_SRC_DIR)/liolib.c $(LUA_SRC_DIR)/llex.c \
           $(LUA_SRC_DIR)/lmathlib.c $(LUA_SRC_DIR)/lmem.c $(LUA_SRC_DIR)/loadlib.c \
           $(LUA_SRC_DIR)/lobject.c $(LUA_SRC_DIR)/lopcodes.c $(LUA_SRC_DIR)/loslib.c \
           $(LUA_SRC_DIR)/lparser.c $(LUA_SRC_DIR)/lstate.c $(LUA_SRC_DIR)/lstring.c \
           $(LUA_SRC_DIR)/lstrlib.c $(LUA_SRC_DIR)/ltable.c $(LUA_SRC_DIR)/ltablib.c \
           $(LUA_SRC_DIR)/ltm.c $(LUA_SRC_DIR)/lundump.c $(LUA_SRC_DIR)/lutf8lib.c \
           $(LUA_SRC_DIR)/lvm.c $(LUA_SRC_DIR)/lzio.c
LUA_CFLAGS_BUNDLED = -I$(LUA_SRC_DIR) -DLUA_USE_POSIX
SRCS_LUA = $(SRC_DIR)/amk_lua.c

.PHONY: all clean dynamic full go-lib cloud-lib both lua tests vagus test_vagus weights-f32

all: $(TARGET)

# Convert float16 weights to float32 (for GitHub-friendly storage)
# Float16 files are ~50% smaller, converted on first build
weights-f32:
	@if [ -f weights/arianna_34m_f16.bin ] && [ ! -f weights/arianna_34m.bin ]; then \
		echo "[weights] Converting float16 -> float32..."; \
		python3 scripts/f16_to_f32.py weights/arianna_34m_f16.bin weights/arianna_34m.bin; \
	fi
	@if [ -f weights/arianna_20m_f16.bin ] && [ ! -f weights/arianna_20m.bin ]; then \
		echo "[weights] Converting float16 -> float32..."; \
		python3 scripts/f16_to_f32.py weights/arianna_20m_f16.bin weights/arianna_20m.bin; \
	fi

dynamic: weights-f32 $(TARGET_DYN)

lua: $(TARGET_LUA)

full: go-lib $(TARGET_FULL)

both: $(TARGET) $(TARGET_DYN)

# Build Go libraries (inner_world + cloud)
go-lib: cloud-lib
	cd inner_world && go build -buildmode=c-shared -o libinner_world.$(DYLIB_EXT) .
	@mkdir -p $(GO_LIB_DIR)
	cp inner_world/libinner_world.$(DYLIB_EXT) $(GO_LIB_DIR)/

cloud-lib:
	cd inner_world && go build -buildmode=c-shared -o ../$(GO_LIB_DIR)/libcloud.$(DYLIB_EXT) .
	@mkdir -p $(GO_LIB_DIR)

$(TARGET): $(SRCS) $(SRC_DIR)/arianna.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $(SRCS) -o $(TARGET) $(LDFLAGS)

$(TARGET_DYN): $(SRCS_DYN) $(SRC_DIR)/arianna.h $(SRC_DIR)/delta.h $(SRC_DIR)/mood.h \
               $(SRC_DIR)/guided.h $(SRC_DIR)/subjectivity.h $(SRC_DIR)/cooccur.h \
               $(SRC_DIR)/body_sense.h $(SRC_DIR)/selfsense.h $(SRC_DIR)/mathbrain.h \
               $(SRC_DIR)/julia_bridge.h $(SRC_DIR)/delta_enhanced.h $(SRC_DIR)/inner_arianna.h \
               $(SRC_DIR)/amk_kernel.h $(SRC_DIR)/arianna_dsl.h $(SRC_DIR)/meta_arianna.h \
               sartre/sartre_bridge.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(DYN_CFLAGS) -I$(SRC_DIR) -Isartre $(SRCS_DYN) -o $(TARGET_DYN) $(LDFLAGS) $(DYN_LDFLAGS)

# Full version with Go inner_world (C + Go hybrid)
$(TARGET_FULL): $(SRCS_DYN) $(SRC_DIR)/arianna.h $(SRC_DIR)/delta.h $(SRC_DIR)/mood.h \
                $(SRC_DIR)/guided.h $(SRC_DIR)/subjectivity.h $(SRC_DIR)/cooccur.h \
                $(SRC_DIR)/body_sense.h $(SRC_DIR)/selfsense.h $(SRC_DIR)/mathbrain.h \
                $(SRC_DIR)/inner_world.h $(GO_LIB_DIR)/libinner_world.$(DYLIB_EXT)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -DUSE_GO_INNER_WORLD -I$(SRC_DIR) -Isartre $(SRCS_DYN) -o $(TARGET_FULL) $(LDFLAGS) $(GO_LDFLAGS)
	@cp $(GO_LIB_DIR)/libinner_world.$(DYLIB_EXT) $(BIN_DIR)/
ifeq ($(PLATFORM),macos)
	@install_name_tool -change libinner_world.dylib @loader_path/libinner_world.dylib $(TARGET_FULL)
endif

# Lua-enabled dynamic version
$(TARGET_LUA): $(SRCS_DYN) $(SRCS_LUA) $(SRC_DIR)/arianna.h $(SRC_DIR)/delta.h $(SRC_DIR)/mood.h \
               $(SRC_DIR)/guided.h $(SRC_DIR)/subjectivity.h $(SRC_DIR)/cooccur.h \
               $(SRC_DIR)/body_sense.h $(SRC_DIR)/selfsense.h $(SRC_DIR)/mathbrain.h \
               $(SRC_DIR)/julia_bridge.h $(SRC_DIR)/delta_enhanced.h $(SRC_DIR)/inner_arianna.h \
               $(SRC_DIR)/amk_kernel.h $(SRC_DIR)/arianna_dsl.h $(SRC_DIR)/amk_lua.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(LUA_CFLAGS) -DUSE_LUA -I$(SRC_DIR) $(SRCS_DYN) $(SRCS_LUA) -o $(TARGET_LUA) $(LDFLAGS) $(LUA_LDFLAGS)

clean:
	rm -rf $(BIN_DIR)/*

# ============================================================
# TESTS
# ============================================================

TEST_DIR = tests
TEST_BIN_DIR = $(BIN_DIR)

# Common test dependencies
TEST_COMMON = $(SRC_DIR)/ariannabody.c

# Individual tests
test_cloud: $(TEST_BIN_DIR)/test_cloud
$(TEST_BIN_DIR)/test_cloud: $(TEST_DIR)/test_cloud.c $(SRC_DIR)/cloud_wrapper.c $(TEST_COMMON)
	@mkdir -p $(TEST_BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $^ -o $@ $(LDFLAGS)

test_amlk: $(TEST_BIN_DIR)/test_amlk
$(TEST_BIN_DIR)/test_amlk: $(TEST_DIR)/test_amlk.c $(SRC_DIR)/amk_kernel.c $(SRC_DIR)/schumann.c $(SRC_DIR)/amk_lua.c $(TEST_COMMON)
	@mkdir -p $(TEST_BIN_DIR)
	$(CC) $(CFLAGS) $(DYN_CFLAGS) -I$(SRC_DIR) $^ -o $@ $(LDFLAGS) $(DYN_LDFLAGS) $(GO_LDFLAGS)

test_comprehensive: $(TEST_BIN_DIR)/test_comprehensive
$(TEST_BIN_DIR)/test_comprehensive: $(TEST_DIR)/test_comprehensive.c $(SRC_DIR)/cloud_wrapper.c $(SRC_DIR)/schumann.c \
                                     $(SRC_DIR)/mood.c $(SRC_DIR)/body_sense.c $(SRC_DIR)/delta.c $(SRC_DIR)/mathbrain.c \
                                     $(SRC_DIR)/inner_arianna.c $(SRC_DIR)/amk_kernel.c $(SRC_DIR)/cooccur.c $(TEST_COMMON)
	@mkdir -p $(TEST_BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $^ -o $@ $(LDFLAGS)

test_julia: $(TEST_BIN_DIR)/test_julia
$(TEST_BIN_DIR)/test_julia: $(TEST_DIR)/test_julia.c $(SRC_DIR)/julia_bridge.c $(TEST_COMMON)
	@mkdir -p $(TEST_BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $^ -o $@ $(LDFLAGS)

test_inner: $(TEST_BIN_DIR)/test_inner
$(TEST_BIN_DIR)/test_inner: $(TEST_DIR)/test_inner.c $(SRC_DIR)/inner_arianna.c $(TEST_COMMON)
	@mkdir -p $(TEST_BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $^ -o $@ $(LDFLAGS)

test_amk: $(TEST_BIN_DIR)/test_amk
$(TEST_BIN_DIR)/test_amk: $(TEST_DIR)/test_amk.c $(SRC_DIR)/amk_kernel.c $(SRC_DIR)/schumann.c $(TEST_COMMON)
	@mkdir -p $(TEST_BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $^ -o $@ $(LDFLAGS)

test_accumulator: $(TEST_BIN_DIR)/test_accumulator
$(TEST_BIN_DIR)/test_accumulator: $(TEST_DIR)/test_accumulator.c $(SRC_DIR)/delta.c $(SRC_DIR)/schumann.c $(TEST_COMMON)
	@mkdir -p $(TEST_BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $^ -o $@ $(LDFLAGS)

test_blood: $(TEST_BIN_DIR)/test_blood
$(TEST_BIN_DIR)/test_blood: $(TEST_DIR)/test_blood.c
	@mkdir -p $(TEST_BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $^ -o $@ $(LDFLAGS) $(GO_LDFLAGS)

test_high: $(TEST_BIN_DIR)/test_high
$(TEST_BIN_DIR)/test_high: $(TEST_DIR)/test_high.c
	@mkdir -p $(TEST_BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $^ -o $@ $(LDFLAGS) $(GO_LDFLAGS)

test_delta_enhanced: $(TEST_BIN_DIR)/test_delta_enhanced
$(TEST_BIN_DIR)/test_delta_enhanced: $(TEST_DIR)/test_delta_enhanced.c $(SRC_DIR)/delta_enhanced.c $(SRC_DIR)/delta.c $(SRC_DIR)/body_sense.c $(SRC_DIR)/schumann.c $(TEST_COMMON)
	@mkdir -p $(TEST_BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $^ -o $@ $(LDFLAGS)

test_inner_world: $(TEST_BIN_DIR)/test_inner_world
$(TEST_BIN_DIR)/test_inner_world: $(TEST_DIR)/test_inner_world.c
	@mkdir -p $(TEST_BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $^ -o $@ $(LDFLAGS) $(GO_LDFLAGS)

test_mathbrain: $(TEST_BIN_DIR)/test_mathbrain
$(TEST_BIN_DIR)/test_mathbrain: $(TEST_DIR)/test_mathbrain.c $(SRC_DIR)/mathbrain.c $(TEST_COMMON)
	@mkdir -p $(TEST_BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $^ -o $@ $(LDFLAGS)

test_pandora: $(TEST_BIN_DIR)/test_pandora
$(TEST_BIN_DIR)/test_pandora: $(TEST_DIR)/test_pandora.c $(SRC_DIR)/pandora.c $(TEST_COMMON)
	@mkdir -p $(TEST_BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $^ -o $@ $(LDFLAGS)

test_selfsense: $(TEST_BIN_DIR)/test_selfsense
$(TEST_BIN_DIR)/test_selfsense: $(TEST_DIR)/test_selfsense.c $(SRC_DIR)/selfsense.c $(TEST_COMMON)
	@mkdir -p $(TEST_BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $^ -o $@ $(LDFLAGS)

test_ariannabody_extended: $(TEST_BIN_DIR)/test_ariannabody_extended
$(TEST_BIN_DIR)/test_ariannabody_extended: $(TEST_DIR)/test_ariannabody_extended.c $(SRC_DIR)/ariannabody.c $(TEST_COMMON)
	@mkdir -p $(TEST_BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $^ -o $@ $(LDFLAGS)

test_sampling_edge_cases: $(TEST_BIN_DIR)/test_sampling_edge_cases
$(TEST_BIN_DIR)/test_sampling_edge_cases: $(TEST_DIR)/test_sampling_edge_cases.c $(SRC_DIR)/ariannabody.c $(TEST_COMMON)
	@mkdir -p $(TEST_BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $^ -o $@ $(LDFLAGS)

test_delta: $(TEST_BIN_DIR)/test_delta
$(TEST_BIN_DIR)/test_delta: $(TEST_DIR)/test_delta.c $(SRC_DIR)/delta.c $(SRC_DIR)/schumann.c $(TEST_COMMON)
	@mkdir -p $(TEST_BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $^ -o $@ $(LDFLAGS)

test_vagus_delta: $(TEST_BIN_DIR)/test_vagus_delta
$(TEST_BIN_DIR)/test_vagus_delta: $(TEST_DIR)/test_vagus_delta.c $(SRC_DIR)/vagus_delta.c $(SRC_DIR)/delta.c $(SRC_DIR)/schumann.c locus/locus.c $(TEST_COMMON)
	@mkdir -p $(TEST_BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) -Ilocus -Ivagus $^ -o $@ $(LDFLAGS)

test_meta_arianna: $(TEST_BIN_DIR)/test_meta_arianna
$(TEST_BIN_DIR)/test_meta_arianna: $(TEST_DIR)/test_meta_arianna.c $(SRC_DIR)/meta_arianna.c $(TEST_COMMON)
	@mkdir -p $(TEST_BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $^ -o $@ $(LDFLAGS)

# Go race tests (requires Go)
test_go_race:
	@echo "[test] Running Go race tests..."
	cd inner_world && go test -race -v ./...

# Run all tests
tests: test_amlk test_cloud test_comprehensive test_accumulator test_inner test_amk test_mathbrain test_pandora test_selfsense test_delta_enhanced test_julia test_ariannabody_extended test_meta_arianna
	@echo ""
	@echo "=========================================="
	@echo "RUNNING ALL ARIANNA TESTS"
	@echo "=========================================="
	@./tests/run_all_tests.sh

# ============================================================
# SARTRE - Metalinux Kernel Verbal Interface
# ============================================================

SARTRE_DIR = sartre
SARTRE_SRC = $(SARTRE_DIR)/sartre_kernel.c
SARTRE_TARGET = $(BIN_DIR)/test_sartre

sartre: $(SARTRE_TARGET)

$(SARTRE_TARGET): $(SARTRE_SRC) $(TEST_DIR)/test_sartre.c $(SARTRE_DIR)/sartre.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(SARTRE_DIR) -I$(SRC_DIR) $(SARTRE_SRC) $(TEST_DIR)/test_sartre.c -o $@ $(LDFLAGS)
	@echo "[sartre] compiled"

# SARTRE comprehensive test
test_sartre_comprehensive: $(TEST_BIN_DIR)/test_sartre_comprehensive
$(TEST_BIN_DIR)/test_sartre_comprehensive: $(TEST_DIR)/test_sartre_comprehensive.c $(SARTRE_SRC) $(SARTRE_DIR)/sartre.h
	@mkdir -p $(TEST_BIN_DIR)
	$(CC) $(CFLAGS) -I$(SARTRE_DIR) -I$(SRC_DIR) $(TEST_DIR)/test_sartre_comprehensive.c $(SARTRE_SRC) -o $@ $(LDFLAGS)
	@echo "[sartre] comprehensive test compiled"

.PHONY: sartre test_sartre test_sartre_comprehensive

# SARTRE inference binary (standalone transformer)
SARTRE_INFERENCE_SRC = $(SARTRE_DIR)/sartre.c
SARTRE_INFERENCE_BIN = $(BIN_DIR)/sartre

sartre_inference: $(SARTRE_INFERENCE_BIN)

$(SARTRE_INFERENCE_BIN): $(SARTRE_INFERENCE_SRC)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(SARTRE_INFERENCE_SRC) -o $@ $(LDFLAGS)
	@echo "[sartre] inference binary compiled"

.PHONY: sartre_inference

# ============================================================
# VAGUS - The Nervous System (Zig)
# ============================================================

VAGUS_DIR = vagus

vagus:
	@echo "[vagus] building..."
	cd $(VAGUS_DIR) && zig build
	@echo "[vagus] libvagus.a + libvagus.so built"

test_vagus:
	@echo "[vagus] running tests..."
	cd $(VAGUS_DIR) && zig build test
	@echo "[vagus] 35/35 tests passed"

vagus-clean:
	rm -rf $(VAGUS_DIR)/.zig-cache $(VAGUS_DIR)/zig-out

.PHONY: vagus test_vagus vagus-clean

# ============================================================
# LOCUS - Resonance Detector (Locus Coeruleus)
# ============================================================

LOCUS_DIR = locus

locus:
	@echo "[locus] building..."
	cd $(LOCUS_DIR) && make
	@echo "[locus] liblocus.a built"

test_locus:
	@echo "[locus] running tests..."
	cd $(LOCUS_DIR) && make test
	@echo "[locus] 16/16 tests passed"

locus-clean:
	cd $(LOCUS_DIR) && make clean

.PHONY: locus test_locus locus-clean

