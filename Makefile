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

# Dynamic version with delta + mood + guided attention + subjectivity support
SRCS_DYN = $(SRC_DIR)/model.c $(SRC_DIR)/delta.c $(SRC_DIR)/mood.c $(SRC_DIR)/guided.c $(SRC_DIR)/subjectivity.c $(SRC_DIR)/arianna_dynamic.c
TARGET_DYN = $(BIN_DIR)/arianna_dynamic

.PHONY: all clean run init dynamic

all: $(TARGET)

dynamic: $(TARGET_DYN)

both: $(TARGET) $(TARGET_DYN)

$(TARGET): $(SRCS) $(SRC_DIR)/arianna.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $(SRCS) -o $(TARGET) $(LDFLAGS)
	@echo "Built $(TARGET)"

$(TARGET_DYN): $(SRCS_DYN) $(SRC_DIR)/arianna.h $(SRC_DIR)/delta.h $(SRC_DIR)/mood.h $(SRC_DIR)/guided.h $(SRC_DIR)/subjectivity.h
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

# Debug build
debug: CFLAGS = -g -Wall -Wextra -fsanitize=address
debug: $(TARGET)
