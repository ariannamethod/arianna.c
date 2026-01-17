# inner_world Makefile — Build Go inner world library
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: all shared static test clean

# Default: build shared library
all: shared

# Build shared library (.so on Linux, .dylib on macOS)
shared:
	go build -buildmode=c-shared -o libinner_world.so .
	@echo "Built: libinner_world.so"
	@echo "Also generated: libinner_world.h (cgo header)"

# Build static library (.a)
static:
	go build -buildmode=c-archive -o libinner_world.a .
	@echo "Built: libinner_world.a"
	@echo "Also generated: libinner_world.h (cgo header)"

# Build test binary
test: shared
	gcc -o test_inner_world test_inner_world.c -L. -linner_world -Wl,-rpath,.
	./test_inner_world

# Run Go tests
gotest:
	go test -v ./...

# Clean build artifacts
clean:
	rm -f libinner_world.so libinner_world.a libinner_world.h
	rm -f test_inner_world

# Install to parent project
install: shared
	cp libinner_world.so ../lib/
	cp inner_world.h ../src/
	@echo "Installed to ../lib/ and ../src/"

# For macOS: build .dylib
macos:
	go build -buildmode=c-shared -o libinner_world.dylib .
	@echo "Built: libinner_world.dylib"
