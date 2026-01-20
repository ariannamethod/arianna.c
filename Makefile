# arianna.c Makefile
# Llama 3.5 Arianna Edition - C + Go

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

# Dynamic version with full pipeline (includes Lua if available)
SRCS_DYN_CORE = $(SRC_DIR)/model.c $(SRC_DIR)/cloud.c $(SRC_DIR)/julia_bridge.c \
           $(SRC_DIR)/schumann.c $(SRC_DIR)/pandora.c $(SRC_DIR)/delta.c \
           $(SRC_DIR)/delta_enhanced.c $(SRC_DIR)/mood.c $(SRC_DIR)/guided.c \
           $(SRC_DIR)/subjectivity.c $(SRC_DIR)/cooccur.c $(SRC_DIR)/body_sense.c \
           $(SRC_DIR)/selfsense.c $(SRC_DIR)/mathbrain.c $(SRC_DIR)/inner_arianna.c \
           $(SRC_DIR)/amk_kernel.c $(SRC_DIR)/arianna_dsl.c \
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

# Lua support (optional)
LUA_CFLAGS = $(shell pkg-config --cflags lua5.4 2>/dev/null || pkg-config --cflags lua 2>/dev/null)
LUA_LDFLAGS = $(shell pkg-config --libs lua5.4 2>/dev/null || pkg-config --libs lua 2>/dev/null)
SRCS_LUA = $(SRC_DIR)/amk_lua.c
TARGET_LUA = $(BIN_DIR)/arianna_lua

.PHONY: all clean dynamic full go-lib both lua

all: $(TARGET)

dynamic: $(TARGET_DYN)

lua: $(TARGET_LUA)

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
	@echo "Usage: ./bin/arianna weights/arianna.bin weights/tokenizer.json \"prompt\""

$(TARGET_DYN): $(SRCS_DYN) $(SRC_DIR)/arianna.h $(SRC_DIR)/delta.h $(SRC_DIR)/mood.h \
               $(SRC_DIR)/guided.h $(SRC_DIR)/subjectivity.h $(SRC_DIR)/cooccur.h \
               $(SRC_DIR)/body_sense.h $(SRC_DIR)/selfsense.h $(SRC_DIR)/mathbrain.h \
               $(SRC_DIR)/julia_bridge.h $(SRC_DIR)/delta_enhanced.h $(SRC_DIR)/inner_arianna.h \
               $(SRC_DIR)/amk_kernel.h $(SRC_DIR)/arianna_dsl.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(DYN_CFLAGS) -I$(SRC_DIR) $(SRCS_DYN) -o $(TARGET_DYN) $(LDFLAGS) $(DYN_LDFLAGS)
ifeq ($(HAS_LUA),yes)
	@echo "Built $(TARGET_DYN) [WITH LUA]"
else
	@echo "Built $(TARGET_DYN) [no lua - install lua to enable]"
endif
	@echo ""
	@echo "Usage:"
	@echo "  ./bin/arianna_dynamic weights/arianna.bin weights/tokenizer.json \"prompt\" 100 0.9"
	@echo "  ./bin/arianna_dynamic weights/arianna.bin weights/tokenizer.json \"prompt\" 100 0.9 -lua scripts/amk_default.lua"

# Full version with Go inner_world (C + Go hybrid)
$(TARGET_FULL): $(SRCS_DYN) $(SRC_DIR)/arianna.h $(SRC_DIR)/delta.h $(SRC_DIR)/mood.h \
                $(SRC_DIR)/guided.h $(SRC_DIR)/subjectivity.h $(SRC_DIR)/cooccur.h \
                $(SRC_DIR)/body_sense.h $(SRC_DIR)/selfsense.h $(SRC_DIR)/mathbrain.h \
                $(SRC_DIR)/inner_world.h $(GO_LIB_DIR)/libinner_world.dylib
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -DUSE_GO_INNER_WORLD -I$(SRC_DIR) $(SRCS_DYN) -o $(TARGET_FULL) $(LDFLAGS) $(GO_LDFLAGS)
	@cp $(GO_LIB_DIR)/libinner_world.dylib $(BIN_DIR)/
	@install_name_tool -change libinner_world.dylib @loader_path/libinner_world.dylib $(TARGET_FULL)
	@echo "Built $(TARGET_FULL) (C + Go hybrid)"
	@echo ""
	@echo "Usage:"
	@echo "  ./bin/arianna_full weights/arianna.bin weights/tokenizer.json \"prompt\" 100 0.9 -async"

# Lua-enabled dynamic version
$(TARGET_LUA): $(SRCS_DYN) $(SRCS_LUA) $(SRC_DIR)/arianna.h $(SRC_DIR)/delta.h $(SRC_DIR)/mood.h \
               $(SRC_DIR)/guided.h $(SRC_DIR)/subjectivity.h $(SRC_DIR)/cooccur.h \
               $(SRC_DIR)/body_sense.h $(SRC_DIR)/selfsense.h $(SRC_DIR)/mathbrain.h \
               $(SRC_DIR)/julia_bridge.h $(SRC_DIR)/delta_enhanced.h $(SRC_DIR)/inner_arianna.h \
               $(SRC_DIR)/amk_kernel.h $(SRC_DIR)/arianna_dsl.h $(SRC_DIR)/amk_lua.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(LUA_CFLAGS) -DUSE_LUA -I$(SRC_DIR) $(SRCS_DYN) $(SRCS_LUA) -o $(TARGET_LUA) $(LDFLAGS) $(LUA_LDFLAGS)
	@echo "Built $(TARGET_LUA) (with Lua scripting)"
	@echo ""
	@echo "Usage:"
	@echo "  ./bin/arianna_lua weights/arianna.bin weights/tokenizer.json \"prompt\" 100 0.9 -lua scripts/amk_default.lua"

clean:
	rm -rf $(BIN_DIR)/*
