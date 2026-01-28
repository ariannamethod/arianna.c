CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra
LDFLAGS = -lm

SRC = src/nano.c
BIN = nano

all: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

debug: $(SRC)
	$(CC) -g -O0 -Wall -Wextra -fsanitize=address -o nano_debug $< $(LDFLAGS)

clean:
	rm -f $(BIN) nano_debug

.PHONY: all debug clean
