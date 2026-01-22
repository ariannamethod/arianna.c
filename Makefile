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

.PHONY: all clean dynamic full go-lib cloud-lib both lua

all: $(TARGET)

dynamic: $(TARGET_DYN)

lua: $(TARGET_LUA)

full: go-lib $(TARGET_FULL)

both: $(TARGET) $(TARGET_DYN)

# Build Go libraries (inner_world + cloud)
go-lib: cloud-lib
	cd inner_world && go build -buildmode=c-shared -o libinner_world.$(DYLIB_EXT) inner_world.go
	@mkdir -p $(GO_LIB_DIR)
	cp inner_world/libinner_world.$(DYLIB_EXT) $(GO_LIB_DIR)/

cloud-lib:
	cd inner_world && go build -buildmode=c-shared -o ../$(GO_LIB_DIR)/libcloud.$(DYLIB_EXT) cloud.go
	@mkdir -p $(GO_LIB_DIR)

$(TARGET): $(SRCS) $(SRC_DIR)/arianna.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $(SRCS) -o $(TARGET) $(LDFLAGS)

$(TARGET_DYN): $(SRCS_DYN) $(SRC_DIR)/arianna.h $(SRC_DIR)/delta.h $(SRC_DIR)/mood.h \
               $(SRC_DIR)/guided.h $(SRC_DIR)/subjectivity.h $(SRC_DIR)/cooccur.h \
               $(SRC_DIR)/body_sense.h $(SRC_DIR)/selfsense.h $(SRC_DIR)/mathbrain.h \
               $(SRC_DIR)/julia_bridge.h $(SRC_DIR)/delta_enhanced.h $(SRC_DIR)/inner_arianna.h \
               $(SRC_DIR)/amk_kernel.h $(SRC_DIR)/arianna_dsl.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(DYN_CFLAGS) -I$(SRC_DIR) $(SRCS_DYN) -o $(TARGET_DYN) $(LDFLAGS) $(DYN_LDFLAGS)

# Full version with Go inner_world (C + Go hybrid)
$(TARGET_FULL): $(SRCS_DYN) $(SRC_DIR)/arianna.h $(SRC_DIR)/delta.h $(SRC_DIR)/mood.h \
                $(SRC_DIR)/guided.h $(SRC_DIR)/subjectivity.h $(SRC_DIR)/cooccur.h \
                $(SRC_DIR)/body_sense.h $(SRC_DIR)/selfsense.h $(SRC_DIR)/mathbrain.h \
                $(SRC_DIR)/inner_world.h $(GO_LIB_DIR)/libinner_world.$(DYLIB_EXT)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -DUSE_GO_INNER_WORLD -I$(SRC_DIR) $(SRCS_DYN) -o $(TARGET_FULL) $(LDFLAGS) $(GO_LDFLAGS)
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
