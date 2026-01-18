# arianna.c Makefile
# Personality Weights Transformer in Pure C + Go

CC = gcc
CFLAGS = -O3 -Wall -Wextra -march=native
LDFLAGS = -lm

# Go inner_world library
GO_LIB_DIR = lib
GO_LDFLAGS = -L$(GO_LIB_DIR) -linner_world -Wl,-rpath,@loader_path

SRC_DIR = src
BIN_DIR = bin

# Basic version (with Cloud pre-processing)
SRCS = $(SRC_DIR)/model.c $(SRC_DIR)/cloud.c $(SRC_DIR)/main.c
TARGET = $(BIN_DIR)/arianna

# Dynamic version with full Stanley-style architecture (includes Cloud)
SRCS_DYN = $(SRC_DIR)/model.c $(SRC_DIR)/cloud.c $(SRC_DIR)/delta.c $(SRC_DIR)/delta_enhanced.c $(SRC_DIR)/mood.c $(SRC_DIR)/guided.c $(SRC_DIR)/subjectivity.c $(SRC_DIR)/cooccur.c $(SRC_DIR)/body_sense.c $(SRC_DIR)/selfsense.c $(SRC_DIR)/mathbrain.c $(SRC_DIR)/arianna_dynamic.c
TARGET_DYN = $(BIN_DIR)/arianna_dynamic

# Enhanced delta test
SRCS_TEST_DELTA = $(SRC_DIR)/delta.c $(SRC_DIR)/delta_enhanced.c $(SRC_DIR)/body_sense.c tests/test_delta_enhanced.c
TARGET_TEST_DELTA = $(BIN_DIR)/test_delta_enhanced

# Full version with Go inner_world
SRCS_FULL = $(SRCS_DYN)
TARGET_FULL = $(BIN_DIR)/arianna_full

.PHONY: all clean run init dynamic full go-lib

all: $(TARGET)

dynamic: $(TARGET_DYN)

full: go-lib $(TARGET_FULL)

both: $(TARGET) $(TARGET_DYN)

# Build Go library
go-lib:
	@echo "Building Go inner_world library..."
	cd inner_world && go build -buildmode=c-shared -o libinner_world.dylib .
	@mkdir -p $(GO_LIB_DIR)
	cp inner_world/libinner_world.dylib $(GO_LIB_DIR)/
	@echo "Go library ready in $(GO_LIB_DIR)/"

$(TARGET): $(SRCS) $(SRC_DIR)/arianna.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $(SRCS) -o $(TARGET) $(LDFLAGS)
	@echo "Built $(TARGET)"

$(TARGET_DYN): $(SRCS_DYN) $(SRC_DIR)/arianna.h $(SRC_DIR)/delta.h $(SRC_DIR)/delta_enhanced.h $(SRC_DIR)/mood.h $(SRC_DIR)/guided.h $(SRC_DIR)/subjectivity.h $(SRC_DIR)/cooccur.h $(SRC_DIR)/body_sense.h $(SRC_DIR)/selfsense.h $(SRC_DIR)/mathbrain.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $(SRCS_DYN) -o $(TARGET_DYN) $(LDFLAGS)
	@echo "Built $(TARGET_DYN)"
	@echo ""
	@echo "Dynamic version usage:"
	@echo "  ./bin/arianna_dynamic weights.bin \"She finds that \" 100 0.8"
	@echo "  ./bin/arianna_dynamic weights.bin -shard exp.bin \"She \" 100 0.8"

# Full version with Go inner_world (C + Go hybrid)
$(TARGET_FULL): $(SRCS_FULL) $(SRC_DIR)/arianna.h $(SRC_DIR)/delta.h $(SRC_DIR)/delta_enhanced.h $(SRC_DIR)/mood.h $(SRC_DIR)/guided.h $(SRC_DIR)/subjectivity.h $(SRC_DIR)/cooccur.h $(SRC_DIR)/body_sense.h $(SRC_DIR)/selfsense.h $(SRC_DIR)/mathbrain.h $(SRC_DIR)/inner_world.h $(GO_LIB_DIR)/libinner_world.dylib
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -DUSE_GO_INNER_WORLD -I$(SRC_DIR) $(SRCS_FULL) -o $(TARGET_FULL) $(LDFLAGS) $(GO_LDFLAGS)
	@# Copy library next to binary and fix load path
	@cp $(GO_LIB_DIR)/libinner_world.dylib $(BIN_DIR)/
	@install_name_tool -change libinner_world.dylib @loader_path/libinner_world.dylib $(TARGET_FULL)
	@echo "Built $(TARGET_FULL) (C + Go hybrid)"
	@echo ""
	@echo "Full version with Go inner_world:"
	@echo "  ./bin/arianna_full weights.bin \"She finds that \" 100 0.8"
	@echo ""
	@echo "Features: trauma detection, emotional drift, prophecy debt, wormholes..."

clean:
	rm -rf $(BIN_DIR)/*

init: $(TARGET)
	./$(TARGET) --init data/arianna_random.bin

run: $(TARGET)
	./$(TARGET) data/arianna_c.bin "She finds that " 100 0.8

run-dynamic: $(TARGET_DYN)
	./$(TARGET_DYN) data/arianna_c.bin "She finds that " 100 0.8 -signals

# MathBrain test
test-math: $(SRC_DIR)/mathbrain.c $(SRC_DIR)/mathbrain.h tests/test_mathbrain.c
	$(CC) $(CFLAGS) -I$(SRC_DIR) tests/test_mathbrain.c $(SRC_DIR)/mathbrain.c -o $(BIN_DIR)/test_mathbrain $(LDFLAGS)
	@echo "Built $(BIN_DIR)/test_mathbrain"
	./$(BIN_DIR)/test_mathbrain

# Enhanced Delta test (5 revolutionary improvements)
test-delta: $(SRCS_TEST_DELTA) $(SRC_DIR)/delta.h $(SRC_DIR)/delta_enhanced.h $(SRC_DIR)/body_sense.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $(SRCS_TEST_DELTA) -o $(TARGET_TEST_DELTA) $(LDFLAGS)
	@echo "Built $(TARGET_TEST_DELTA)"
	./$(TARGET_TEST_DELTA)

# AMK Kernel test
test-amk: tests/test_amk.c $(SRC_DIR)/amk_kernel.c $(SRC_DIR)/amk_kernel.h $(SRC_DIR)/schumann.c
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) tests/test_amk.c $(SRC_DIR)/amk_kernel.c $(SRC_DIR)/schumann.c -o $(BIN_DIR)/test_amk $(LDFLAGS)
	@echo "Built $(BIN_DIR)/test_amk"
	./$(BIN_DIR)/test_amk

# External Brain test
test-brain: tests/test_brain.c $(SRC_DIR)/external_brain.c $(SRC_DIR)/external_brain.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) tests/test_brain.c $(SRC_DIR)/external_brain.c -o $(BIN_DIR)/test_brain $(LDFLAGS)
	@echo "Built $(BIN_DIR)/test_brain"
	@echo "Note: test_brain requires weights/gpt2_30m/ to run"

# Cloud test
test-cloud: tests/test_cloud.c $(SRC_DIR)/cloud.c $(SRC_DIR)/cloud.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) tests/test_cloud.c $(SRC_DIR)/cloud.c -o $(BIN_DIR)/test_cloud $(LDFLAGS)
	@echo "Built $(BIN_DIR)/test_cloud"
	./$(BIN_DIR)/test_cloud

# Inner Arianna test
test-inner: tests/test_inner.c $(SRC_DIR)/inner_arianna.c $(SRC_DIR)/inner_arianna.h $(SRC_DIR)/delta.c $(SRC_DIR)/delta.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) tests/test_inner.c $(SRC_DIR)/inner_arianna.c $(SRC_DIR)/delta.c -o $(BIN_DIR)/test_inner $(LDFLAGS)
	@echo "Built $(BIN_DIR)/test_inner"
	./$(BIN_DIR)/test_inner

# Run all tests
test: test-math test-delta test-amk test-cloud test-inner
	@echo ""
	@echo "All tests completed!"

# Debug build
debug: CFLAGS = -g -Wall -Wextra -fsanitize=address
debug: $(TARGET)
