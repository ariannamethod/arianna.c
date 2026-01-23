# ═══════════════════════════════════════════════════════════════════════════════
# LOCUS COERULEUS — RESONANCE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════
# The "blue spot" fires. Norepinephrine floods. SARTRE speaks.
# ═══════════════════════════════════════════════════════════════════════════════

CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c11
LDFLAGS = -lm

# Targets
LIB = liblocus.a
TEST = locus_test

# Sources
SRC = locus.c
OBJ = $(SRC:.c=.o)
TEST_SRC = locus_test.c

# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: all clean test

all: $(LIB)

$(LIB): $(OBJ)
	ar rcs $@ $^

%.o: %.c locus.h
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST): $(TEST_SRC) $(LIB)
	$(CC) $(CFLAGS) $< -L. -llocus $(LDFLAGS) -o $@

test: $(TEST)
	./$(TEST)

clean:
	rm -f $(OBJ) $(LIB) $(TEST)

# ═══════════════════════════════════════════════════════════════════════════════
