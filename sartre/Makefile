# SARTRE Makefile
# "L'enfer, c'est les autres processus."

CC = clang
CFLAGS = -O2 -Wall -Wextra -std=c11

# Standalone REPL
sartre_repl: sartre.c sartre.h
	$(CC) $(CFLAGS) -DSARTRE_MAIN -o sartre_repl sartre.c -lm

# Static library for integration
libsartre.a: sartre.o
	ar rcs libsartre.a sartre.o

sartre.o: sartre.c sartre.h
	$(CC) $(CFLAGS) -c sartre.c -o sartre.o

# Shared library
libsartre.dylib: sartre.c sartre.h
	$(CC) $(CFLAGS) -shared -fPIC -o libsartre.dylib sartre.c -lm

# Clean
clean:
	rm -f sartre_repl sartre.o libsartre.a libsartre.dylib

# Test
test: sartre_repl
	./sartre_repl

.PHONY: clean test
