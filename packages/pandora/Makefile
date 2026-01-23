# Pandora Package Makefile
# Pure C vocabulary extraction from GPT2-30M

CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -ffast-math
LDFLAGS = -lm

SRC_DIR = src
BUILD_DIR = build
LIB_DIR = lib

SOURCES = $(SRC_DIR)/pandora.c $(SRC_DIR)/gpt2_30m.c
OBJECTS = $(BUILD_DIR)/pandora.o $(BUILD_DIR)/gpt2_30m.o
HEADERS = $(SRC_DIR)/pandora.h $(SRC_DIR)/gpt2_30m.h

# Default target
all: dirs $(LIB_DIR)/libpandora.a $(BUILD_DIR)/pandora_test

# Create directories
dirs:
	@mkdir -p $(BUILD_DIR) $(LIB_DIR)

# Static library
$(LIB_DIR)/libpandora.a: $(OBJECTS)
	ar rcs $@ $^

# Object files
$(BUILD_DIR)/pandora.o: $(SRC_DIR)/pandora.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/gpt2_30m.o: $(SRC_DIR)/gpt2_30m.c $(SRC_DIR)/gpt2_30m.h
	$(CC) $(CFLAGS) -c $< -o $@

# Test binary
$(BUILD_DIR)/pandora_test: $(SRC_DIR)/test_pandora.c $(LIB_DIR)/libpandora.a
	$(CC) $(CFLAGS) $< -L$(LIB_DIR) -lpandora $(LDFLAGS) -o $@

# Run test
test: $(BUILD_DIR)/pandora_test
	./$(BUILD_DIR)/pandora_test

# Clean
clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR)

# Install (copy headers and lib to system paths)
install: all
	@echo "Installing to /usr/local..."
	cp $(SRC_DIR)/pandora.h /usr/local/include/
	cp $(SRC_DIR)/gpt2_30m.h /usr/local/include/
	cp $(LIB_DIR)/libpandora.a /usr/local/lib/

# Symlink weights
weights:
	@if [ ! -L weights/gpt2_30m ]; then \
		mkdir -p weights; \
		ln -sf ../../../weights/gpt2_30m weights/gpt2_30m; \
		echo "Linked weights/gpt2_30m -> ../../../weights/gpt2_30m"; \
	fi

.PHONY: all dirs test clean install weights
