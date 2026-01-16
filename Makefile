# arianna.c Makefile
# Personality Weights Transformer in Pure C

CC = gcc
CFLAGS = -O3 -Wall -Wextra -march=native
LDFLAGS = -lm

SRC_DIR = src
BIN_DIR = bin

# Basic version
SRCS = $(SRC_DIR)/model.c $(SRC_DIR)/main.c
TARGET = $(BIN_DIR)/arianna

# Dynamic version with full Stanley-style architecture
SRCS_DYN = $(SRC_DIR)/model.c $(SRC_DIR)/delta.c $(SRC_DIR)/delta_enhanced.c $(SRC_DIR)/mood.c $(SRC_DIR)/guided.c $(SRC_DIR)/subjectivity.c $(SRC_DIR)/cooccur.c $(SRC_DIR)/body_sense.c $(SRC_DIR)/selfsense.c $(SRC_DIR)/mathbrain.c $(SRC_DIR)/arianna_dynamic.c
TARGET_DYN = $(BIN_DIR)/arianna_dynamic

# Enhanced delta test
SRCS_TEST_DELTA = $(SRC_DIR)/delta.c $(SRC_DIR)/delta_enhanced.c $(SRC_DIR)/body_sense.c test_delta_enhanced.c
TARGET_TEST_DELTA = $(BIN_DIR)/test_delta_enhanced

.PHONY: all clean run init dynamic

all: $(TARGET)

dynamic: $(TARGET_DYN)

both: $(TARGET) $(TARGET_DYN)

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

clean:
	rm -rf $(BIN_DIR)/*

init: $(TARGET)
	./$(TARGET) --init data/arianna_random.bin

run: $(TARGET)
	./$(TARGET) data/arianna_c.bin "She finds that " 100 0.8

run-dynamic: $(TARGET_DYN)
	./$(TARGET_DYN) data/arianna_c.bin "She finds that " 100 0.8 -signals

# MathBrain test
test-math: $(SRC_DIR)/mathbrain.c $(SRC_DIR)/mathbrain.h test_mathbrain.c
	$(CC) $(CFLAGS) -I$(SRC_DIR) test_mathbrain.c $(SRC_DIR)/mathbrain.c -o $(BIN_DIR)/test_mathbrain $(LDFLAGS)
	@echo "Built $(BIN_DIR)/test_mathbrain"
	./$(BIN_DIR)/test_mathbrain

# Enhanced Delta test (5 revolutionary improvements)
test-delta: $(SRCS_TEST_DELTA) $(SRC_DIR)/delta.h $(SRC_DIR)/delta_enhanced.h $(SRC_DIR)/body_sense.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $(SRCS_TEST_DELTA) -o $(TARGET_TEST_DELTA) $(LDFLAGS)
	@echo "Built $(TARGET_TEST_DELTA)"
	./$(TARGET_TEST_DELTA)

# Run all tests
test: test-math test-delta
	@echo ""
	@echo "All tests completed!"

# Debug build
debug: CFLAGS = -g -Wall -Wextra -fsanitize=address
debug: $(TARGET)
