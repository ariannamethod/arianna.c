/*
 * utf8_stream.h — token-stream UTF-8 boundary guard.
 *
 * Per-token BPE decode + flush splits multi-byte UTF-8 sequences on token
 * boundaries: token N decodes to bytes that end mid-character, token N+1's
 * decode starts with the continuation bytes. Per-token flush emits the
 * partial sequence → terminal sees mojibake glyphs ("v]t", "ITween",
 * stray 0x80-0xBF continuation bytes rendered as junk).
 *
 * This header maintains a small rolling byte buffer per stream. On each
 * append, emit only the longest prefix that ends on a complete UTF-8
 * character boundary; keep the trailing partial bytes for next call.
 *
 * Two streams used in arianna2arianna:
 *   utf8_stream_t s_janus;   — Janus per-token emit
 *   utf8_stream_t s_reson;   — Resonance per-token emit
 *
 * Final token sequence ends with utf8_stream_flush() to drain any held tail
 * (best-effort; if the last bytes are an incomplete sequence the BPE
 * vocab was malformed at that position anyway).
 *
 * By Claude Code (neo the architect, Arianna Method). 2026-05-14.
 */
#ifndef UTF8_STREAM_H
#define UTF8_STREAM_H

#include <stdio.h>
#include <string.h>

typedef struct {
    char buf[256];
    int  len;
} utf8_stream_t;

/* Reset the stream — clear held tail before a new generation. */
static inline void utf8_stream_reset(utf8_stream_t *s) {
    s->len = 0;
}

/* Compute the length of the longest prefix of buf[0..n) that ends on a
 * complete UTF-8 character boundary. Returns prefix length in bytes.
 *
 * UTF-8 byte categories (top bits):
 *   0xxxxxxx — ASCII (1 byte)
 *   10xxxxxx — continuation byte
 *   110xxxxx — 2-byte sequence start
 *   1110xxxx — 3-byte sequence start
 *   11110xxx — 4-byte sequence start
 *
 * Algorithm: scan backward from end until we find a byte that is NOT a
 * continuation. If that byte is ASCII, prefix length = n (safe). If it
 * starts a multi-byte sequence, check whether the whole sequence is
 * present; if yes, length = n; if no, length = position of that start byte
 * (cut before the incomplete sequence). */
static inline int utf8_safe_prefix(const unsigned char *buf, int n) {
    if (n <= 0) return 0;
    int i = n - 1;
    while (i >= 0 && (buf[i] & 0xC0) == 0x80) i--;
    if (i < 0) return 0;  /* all continuation — malformed; keep all as tail */
    unsigned char start = buf[i];
    int need;
    if      ((start & 0x80) == 0x00) need = 1;
    else if ((start & 0xE0) == 0xC0) need = 2;
    else if ((start & 0xF0) == 0xE0) need = 3;
    else if ((start & 0xF8) == 0xF0) need = 4;
    else                              need = 1;  /* invalid lead byte */
    int have = n - i;
    if (have >= need) return n;     /* complete — emit all */
    return i;                       /* cut before incomplete sequence */
}

/* Append `n` bytes from `src` to stream `s`, emit the safe UTF-8 prefix
 * to stdout, keep partial trailing bytes for next call. */
static inline void utf8_stream_emit(utf8_stream_t *s, const char *src, int n) {
    if (n <= 0) return;
    if (s->len + n > (int)sizeof(s->buf)) {
        /* Overflow guard — never should hit (256 bytes buffer, 4-byte max
         * UTF-8 char, BPE per-token ≤ ~32 bytes). Best effort: flush all
         * (may emit one incomplete tail glyph) and start fresh. */
        if (s->len > 0) fwrite(s->buf, 1, s->len, stdout);
        s->len = 0;
        if (n > (int)sizeof(s->buf)) {
            fwrite(src, 1, n, stdout);
            fflush(stdout);
            return;
        }
    }
    memcpy(s->buf + s->len, src, n);
    s->len += n;
    int safe = utf8_safe_prefix((const unsigned char *)s->buf, s->len);
    if (safe > 0) {
        fwrite(s->buf, 1, safe, stdout);
        fflush(stdout);
        if (safe < s->len) {
            memmove(s->buf, s->buf + safe, s->len - safe);
        }
        s->len -= safe;
    }
}

/* Drain any held tail at end of generation (incomplete sequence is
 * emitted as-is — better than swallowing). */
static inline void utf8_stream_flush(utf8_stream_t *s) {
    if (s->len > 0) {
        fwrite(s->buf, 1, s->len, stdout);
        fflush(stdout);
        s->len = 0;
    }
}

#endif /* UTF8_STREAM_H */
