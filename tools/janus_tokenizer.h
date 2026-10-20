/* Janus base byte-rank BPE. The header pairs reconstruct the vocabulary; they
 * are not a complete encoding algorithm. Any adjacent byte spans whose joined
 * bytes have a base rank can merge. PCRE2 preserves the base Unicode boundaries.
 *
 * Link with pcre2-8. Initialize once from janus_v4_bpe_merges, then use the same
 * encoder for the sequence prompt and the field injection.
 */
#ifndef ARIANNA_JANUS_TOKENIZER_H
#define ARIANNA_JANUS_TOKENIZER_H
#ifndef PCRE2_CODE_UNIT_WIDTH
#define PCRE2_CODE_UNIT_WIDTH 8
#endif
#include <pcre2.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

static const char janus_tokenizer_pattern[] =
    "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}{1,2}"
    "| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+";

typedef struct {
    unsigned char **tokens;
    size_t *lengths;
    int *table;                    /* token ID + 1; zero means an empty slot */
    size_t table_size, max_length;
    int vocab_size;
    pcre2_code *pattern;
} JanusTokenizer;

static uint64_t jt_hash(const unsigned char *s, size_t n) {
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < n; i++) {
        h ^= s[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static int jt_rank(const JanusTokenizer *t, const unsigned char *s, size_t n) {
    if (n > t->max_length) return -1;
    size_t slot = (size_t)jt_hash(s, n) & (t->table_size - 1);
    while (t->table[slot]) {
        int id = t->table[slot] - 1;
        if (t->lengths[id] == n && !memcmp(t->tokens[id], s, n)) return id;
        slot = (slot + 1) & (t->table_size - 1);
    }
    return -1;
}

static void janus_tokenizer_free(JanusTokenizer *t) {
    if (!t) return;
    if (t->tokens)
        for (int i = 0; i < t->vocab_size; i++) free(t->tokens[i]);
    free(t->tokens);
    free(t->lengths);
    free(t->table);
    if (t->pattern) pcre2_code_free(t->pattern);
    memset(t, 0, sizeof(*t));
}

/* Call on an uninitialized/zeroed object; free before initializing it again. */
static int janus_tokenizer_init(JanusTokenizer *t, const int merges[][2],
                                int n_merges) {
    if (!t || !merges || n_merges != 32503) return -1;
    memset(t, 0, sizeof(*t));
    t->vocab_size = 256 + n_merges;
    t->tokens = calloc((size_t)t->vocab_size, sizeof(*t->tokens));
    t->lengths = calloc((size_t)t->vocab_size, sizeof(*t->lengths));
    t->table_size = 1;
    while (t->table_size < 2u * (size_t)t->vocab_size) t->table_size *= 2;
    t->table = calloc(t->table_size, sizeof(*t->table));
    if (!t->tokens || !t->lengths || !t->table) goto fail;
    for (int i = 0; i < t->vocab_size; i++) {
        size_t len = 1;
        int a = 0, b = 0;
        if (i >= 256) {
            a = merges[i - 256][0];
            b = merges[i - 256][1];
            if (a < 0 || b < 0 || a >= i || b >= i) goto fail;
            if (t->lengths[a] > SIZE_MAX - t->lengths[b]) goto fail;
            len = t->lengths[a] + t->lengths[b];
        }
        t->tokens[i] = malloc(len);
        if (!t->tokens[i]) goto fail;
        t->lengths[i] = len;
        if (i < 256) t->tokens[i][0] = (unsigned char)i;
        else {
            memcpy(t->tokens[i], t->tokens[a], t->lengths[a]);
            memcpy(t->tokens[i] + t->lengths[a], t->tokens[b], t->lengths[b]);
        }
        if (len > t->max_length) t->max_length = len;
        size_t slot = (size_t)jt_hash(t->tokens[i], len) & (t->table_size - 1);
        while (t->table[slot]) {
            int old = t->table[slot] - 1;
            if (t->lengths[old] == len && !memcmp(t->tokens[old], t->tokens[i], len))
                goto fail;
            slot = (slot + 1) & (t->table_size - 1);
        }
        t->table[slot] = i + 1;
    }
    int error;
    PCRE2_SIZE error_offset;
    t->pattern = pcre2_compile((PCRE2_SPTR)janus_tokenizer_pattern,
                              PCRE2_ZERO_TERMINATED, PCRE2_UTF | PCRE2_UCP,
                              &error, &error_offset, NULL);
    if (!t->pattern) goto fail;
    return 0;
fail:
    janus_tokenizer_free(t);
    return -1;
}

typedef struct {
    size_t start, end;
    int prev, next;
    unsigned version;
} jt_piece;
typedef struct {
    int rank, left, right;
    unsigned lv, rv;
} jt_candidate;
typedef struct {
    jt_candidate *items;
    size_t count, capacity;
} jt_heap;

static int jt_before(jt_candidate a, jt_candidate b) {
    return a.rank < b.rank || (a.rank == b.rank && a.left < b.left);
}
static int jt_push(jt_heap *h, jt_candidate item) {
    if (h->count == h->capacity) {
        size_t cap = h->capacity ? h->capacity * 2 : 64;
        if (cap < h->capacity || cap > SIZE_MAX / sizeof(*h->items)) return -1;
        jt_candidate *p = realloc(h->items, cap * sizeof(*h->items));
        if (!p) return -1;
        h->items = p;
        h->capacity = cap;
    }
    size_t i = h->count++;
    while (i > 0) {
        size_t parent = (i - 1) / 2;
        if (!jt_before(item, h->items[parent])) break;
        h->items[i] = h->items[parent];
        i = parent;
    }
    h->items[i] = item;
    return 0;
}
static jt_candidate jt_pop(jt_heap *h) {
    jt_candidate result = h->items[0], last = h->items[--h->count];
    size_t i = 0;
    while (i * 2 + 1 < h->count) {
        size_t child = i * 2 + 1;
        if (child + 1 < h->count && jt_before(h->items[child + 1], h->items[child]))
            child++;
        if (!jt_before(h->items[child], last)) break;
        h->items[i] = h->items[child];
        i = child;
    }
    if (h->count) h->items[i] = last;
    return result;
}
static int jt_queue_pair(const JanusTokenizer *t, const unsigned char *s,
                         jt_piece *pieces, int left, jt_heap *heap) {
    if (left < 0) return 0;
    int right = pieces[left].next;
    if (right < 0) return 0;
    int rank = jt_rank(t, s + pieces[left].start,
                       pieces[right].end - pieces[left].start);
    if (rank < 0) return 0;
    jt_candidate candidate = {rank, left, right,
                              pieces[left].version, pieces[right].version};
    return jt_push(heap, candidate);
}

static int jt_encode_piece(const JanusTokenizer *t, const unsigned char *s,
                           size_t n, int *out, int cap) {
    int direct = jt_rank(t, s, n);
    if (direct >= 0) { out[0] = direct; return 1; }
    if (!n) return 0;
    if (n > INT_MAX || n > SIZE_MAX / sizeof(jt_piece)) return -1;
    jt_piece *pieces = calloc(n, sizeof(*pieces));
    jt_heap heap = {0};
    if (!pieces) return -1;
    for (size_t i = 0; i < n; i++) {
        pieces[i].start = i;
        pieces[i].end = i + 1;
        pieces[i].prev = i ? (int)i - 1 : -1;
        pieces[i].next = i + 1 < n ? (int)i + 1 : -1;
    }
    for (size_t i = 0; i + 1 < n; i++)
        if (jt_queue_pair(t, s, pieces, (int)i, &heap)) goto fail;
    while (heap.count) {
        jt_candidate c = jt_pop(&heap);
        jt_piece *left = &pieces[c.left], *right = &pieces[c.right];
        if (left->next != c.right || left->version != c.lv || right->version != c.rv)
            continue;
        left->end = right->end;
        left->next = right->next;
        left->version++;
        if (right->next >= 0) pieces[right->next].prev = c.left;
        right->next = -2;
        right->version++;
        if (jt_queue_pair(t, s, pieces, left->prev, &heap) ||
            jt_queue_pair(t, s, pieces, c.left, &heap)) goto fail;
    }
    int written = 0;
    for (int i = 0; i >= 0 && written < cap; i = pieces[i].next) {
        int id = jt_rank(t, s + pieces[i].start, pieces[i].end - pieces[i].start);
        if (id < 0) goto fail;
        out[written++] = id;
    }
    free(heap.items);
    free(pieces);
    return written;
fail:
    free(heap.items);
    free(pieces);
    return -1;
}

/* Returns token count, or -1 for invalid UTF-8/allocation/regex failure.
 * Capacity limits emitted tokens, after BPE, rather than truncating UTF-8 bytes.
 * Literal special-token spellings remain ordinary text.
 */
static int janus_tokenizer_encode(const JanusTokenizer *t, const char *text,
                                  int text_len, int *out, int max_tokens) {
    if (!t || !t->pattern || !text || !out || text_len < 0 || max_tokens < 0) return -1;
    if (!text_len || !max_tokens) return 0;
    pcre2_match_data *match = pcre2_match_data_create_from_pattern(t->pattern, NULL);
    if (!match) return -1;
    size_t offset = 0;
    int written = 0;
    while (offset < (size_t)text_len && written < max_tokens) {
        int rc = pcre2_match(t->pattern, (PCRE2_SPTR)text, (size_t)text_len,
                             offset, PCRE2_ANCHORED, match, NULL);
        if (rc < 0) { written = -1; break; }
        PCRE2_SIZE *v = pcre2_get_ovector_pointer(match);
        if (v[0] != offset || v[1] <= offset) { written = -1; break; }
        int n = jt_encode_piece(t, (const unsigned char *)text + offset,
                                 v[1] - offset, out + written, max_tokens - written);
        if (n < 0) { written = -1; break; }
        written += n;
        offset = v[1];
    }
    pcre2_match_data_free(match);
    return written;
}
#endif
