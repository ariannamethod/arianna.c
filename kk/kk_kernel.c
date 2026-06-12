/*
 * kk_kernel.c — Knowledge Kernel for the Dario Equation
 *
 * Refactored from kk.c as a library for the Dario equation project.
 * All original logic preserved: schema, FTS5, scoring, link inference.
 *
 * The third organ:
 *   dario.c         — brain  (formula, reaction, equation)
 *   sartre_kernel.c — body   (hardware, processes, environment)
 *   kk_kernel.c     — memory (knowledge, lineage, retrieval)
 *
 * Self-contained. SQLite only. Zero external dependencies.
 * Compiles alone: cc kk_kernel.c -O2 -lsqlite3 -lm -DKK_STANDALONE -o kk
 * Compiles with dario:
 *   cc dario.c sartre_kernel.c kk_kernel.c -DHAS_SARTRE -DHAS_KK -O2 -lm -lsqlite3 -o dario
 *
 * by Arianna Method
 */

#define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 700
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <sqlite3.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "kk_kernel.h"

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define KK_MAX_CHUNK_CHARS 900
#define KK_MIN_CHUNK_CHARS 220
#define KK_MAX_QUERY_TOKENS 16
#define KK_MAX_KEYWORDS 24
#define KK_KEYWORD_LEN 32
#define KK_LINK_KIND_STRUCTURAL 1
#define KK_LINK_KIND_RELATED 2
#define KK_DEFAULT_WATCH_INTERVAL 5
#define KK_DEFAULT_RETRIEVAL_MODE "compressed"
#define KK_DEFAULT_PACKET_MODE "deterministic-json-v1"
#define KK_DEFAULT_QUERY_PROFILE "balanced"
#define KK_PACKET_SCHEMA_VERSION "kk.packet.v2"
#define KK_ASK_SCHEMA_VERSION "kk.ask.v2"

/* ═══════════════════════════════════════════════════════════════════
 * INTERNAL TYPES
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    uint8_t data[64];
    uint32_t datalen;
    uint64_t bitlen;
    uint32_t state[8];
} SHA256_CTX;

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} StrBuf;

typedef struct {
    char *text;
    size_t start;
    size_t end;
    int level;
    int ordinal;
} SectionInfo;

typedef struct {
    char *text;
    size_t start;
    size_t end;
    int ordinal;
} ChunkInfo;

typedef struct {
    int id;
    int level;
    int ordinal;
    int first_chunk_id;
    int last_chunk_id;
    char *heading;
} SectionRuntime;

typedef struct {
    double lexical;
    double recency;
    double trust;
    double linkage;
    double scope;
    double namespace_match;
    double freshness;
} ScorePolicy;

typedef struct {
    int chunk_id;
    int doc_id;
    int version_id;
    int version_num;
    int ordinal;
    int seen_count;
    int structural_links;
    int related_links;
    int token_estimate;
    int char_delta;
    int token_delta;
    double lexical;
    double lexical_norm;
    double recency;
    double trust;
    double linkage;
    double scope_score;
    double namespace_score;
    double freshness;
    double change_ratio;
    double weighted_lexical;
    double weighted_recency;
    double weighted_trust;
    double weighted_linkage;
    double weighted_scope;
    double weighted_namespace;
    double weighted_freshness;
    double resonance;
    double hebbian_boost;
    const char *path;
    const char *filename;
    const char *namespace_name;
    const char *scope_name;
    const char *raw_text;
    const char *sha256;
    const char *ingest_ts;
    const char *first_seen_ts;
    const char *last_seen_ts;
    const char *section_title;
    const char *diff_summary;
} QueryResult;

typedef struct {
    const char *name;
    int result_cap;
    int excerpt_chars;
    int lineage_chars;
    int provenance_chars;
    int adjacent_chars;
    int lineage_verbosity;
    int score_breakdown_level;
    int provenance_level;
    int include_structural_links;
    int include_adjacent_context;
} QueryProfile;

typedef struct {
    char *text;
    int truncated;
} BudgetedText;

typedef struct {
    int applied;
    int requested_top_k;
    int profile_result_cap;
    int effective_result_cap;
    int retrieved_result_count;
    int returned_result_count;
    int result_truncated;
    int any_truncation;
} BudgetSummary;

typedef struct {
    char *model_name;
    char *scope_default;
    char *namespace_default;
    char *retrieval_mode_default;
    char *packet_mode_default;
    char *query_profile_default;
    char *notes;
    int is_active;
    int found;
} ModelAttachment;

typedef struct {
    char *namespace_name;
    char *scope;
    char *description;
    char *owner_model;
    char *created_ts;
    char *updated_ts;
    int found;
} NamespaceManifest;

typedef struct {
    const char *scope_name;
    int queried;
    int hit_count;
} ResolutionStage;

typedef struct {
    int id;
    int ordinal;
    int section_id;
    int token_estimate;
    char *text;
    char *heading;
    char keywords[KK_MAX_KEYWORDS][KK_KEYWORD_LEN];
    int keyword_count;
} LinkChunk;

/* ═══════════════════════════════════════════════════════════════════
 * kk_ctx — the opaque context
 * ═══════════════════════════════════════════════════════════════════ */

struct kk_ctx {
    sqlite3 *db;
    const kk_embedder *embedder;
    const kk_hebbian_bridge *bridge;
    int ready;
};

/* ═══════════════════════════════════════════════════════════════════
 * ERROR HANDLING — library vs CLI
 * ═══════════════════════════════════════════════════════════════════ */

#ifdef KK_STANDALONE
static volatile sig_atomic_t kk_stop = 0;
#endif

static void die(const char *msg) {
    fprintf(stderr, "fatal: %s\n", msg);
#ifdef KK_STANDALONE
    exit(1);
#endif
}

static void die_sqlite(sqlite3 *db, const char *msg) {
    fprintf(stderr, "sqlite error: %s: %s\n", msg, sqlite3_errmsg(db));
#ifdef KK_STANDALONE
    exit(1);
#endif
}

/* ═══════════════════════════════════════════════════════════════════
 * MEMORY UTILITIES
 * ═══════════════════════════════════════════════════════════════════ */

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xrealloc(void *ptr, size_t n) {
    void *p = realloc(ptr, n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static char *xstrdup(const char *s) {
    size_t n = strlen(s);
    char *r = xmalloc(n + 1);
    memcpy(r, s, n + 1);
    return r;
}

static char *xstrndup(const char *s, size_t n) {
    char *r = xmalloc(n + 1);
    memcpy(r, s, n);
    r[n] = '\0';
    return r;
}

/* ═══════════════════════════════════════════════════════════════════
 * STRING BUFFER
 * ═══════════════════════════════════════════════════════════════════ */

static void sb_init(StrBuf *sb) {
    sb->data = xmalloc(1);
    sb->data[0] = '\0';
    sb->len = 0;
    sb->cap = 1;
}

static void sb_reserve(StrBuf *sb, size_t extra) {
    size_t need = sb->len + extra + 1;
    if (need <= sb->cap) return;
    size_t cap = sb->cap;
    while (cap < need) cap *= 2;
    sb->data = xrealloc(sb->data, cap);
    sb->cap = cap;
}

static void sb_append_n(StrBuf *sb, const char *s, size_t n) {
    sb_reserve(sb, n);
    memcpy(sb->data + sb->len, s, n);
    sb->len += n;
    sb->data[sb->len] = '\0';
}

static void sb_append(StrBuf *sb, const char *s) {
    sb_append_n(sb, s, strlen(s));
}

static void sb_appendf(StrBuf *sb, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int need = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (need < 0) die("vsnprintf failed");
    sb_reserve(sb, (size_t)need);
    vsnprintf(sb->data + sb->len, sb->cap - sb->len, fmt, ap2);
    va_end(ap2);
    sb->len += (size_t)need;
}

/* ═══════════════════════════════════════════════════════════════════
 * SHA256
 * ═══════════════════════════════════════════════════════════════════ */

static uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

static const uint32_t k256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

static void sha256_transform(SHA256_CTX *ctx, const uint8_t data[]) {
    uint32_t m[64];
    for (int i = 0, j = 0; i < 16; i++, j += 4) {
        m[i] = ((uint32_t)data[j] << 24) | ((uint32_t)data[j + 1] << 16) |
               ((uint32_t)data[j + 2] << 8) | (uint32_t)data[j + 3];
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr(m[i - 15], 7) ^ rotr(m[i - 15], 18) ^ (m[i - 15] >> 3);
        uint32_t s1 = rotr(m[i - 2], 17) ^ rotr(m[i - 2], 19) ^ (m[i - 2] >> 10);
        m[i] = m[i - 16] + s0 + m[i - 7] + s1;
    }

    uint32_t a = ctx->state[0], b = ctx->state[1], c = ctx->state[2], d = ctx->state[3];
    uint32_t e = ctx->state[4], f = ctx->state[5], g = ctx->state[6], h = ctx->state[7];

    for (int i = 0; i < 64; i++) {
        uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t temp1 = h + s1 + ch + k256[i] + m[i];
        uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = s0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

static void sha256_init(SHA256_CTX *ctx) {
    ctx->datalen = 0;
    ctx->bitlen = 0;
    ctx->state[0] = 0x6a09e667;
    ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372;
    ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f;
    ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab;
    ctx->state[7] = 0x5be0cd19;
}

static void sha256_update(SHA256_CTX *ctx, const uint8_t *data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        ctx->data[ctx->datalen++] = data[i];
        if (ctx->datalen == 64) {
            sha256_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}

static void sha256_final(SHA256_CTX *ctx, uint8_t hash[]) {
    uint32_t i = ctx->datalen;
    if (ctx->datalen < 56) {
        ctx->data[i++] = 0x80;
        while (i < 56) ctx->data[i++] = 0x00;
    } else {
        ctx->data[i++] = 0x80;
        while (i < 64) ctx->data[i++] = 0x00;
        sha256_transform(ctx, ctx->data);
        memset(ctx->data, 0, 56);
    }
    ctx->bitlen += (uint64_t)ctx->datalen * 8;
    ctx->data[63] = (uint8_t)(ctx->bitlen);
    ctx->data[62] = (uint8_t)(ctx->bitlen >> 8);
    ctx->data[61] = (uint8_t)(ctx->bitlen >> 16);
    ctx->data[60] = (uint8_t)(ctx->bitlen >> 24);
    ctx->data[59] = (uint8_t)(ctx->bitlen >> 32);
    ctx->data[58] = (uint8_t)(ctx->bitlen >> 40);
    ctx->data[57] = (uint8_t)(ctx->bitlen >> 48);
    ctx->data[56] = (uint8_t)(ctx->bitlen >> 56);
    sha256_transform(ctx, ctx->data);
    for (i = 0; i < 4; i++) {
        hash[i]      = (ctx->state[0] >> (24 - i * 8)) & 0xff;
        hash[i + 4]  = (ctx->state[1] >> (24 - i * 8)) & 0xff;
        hash[i + 8]  = (ctx->state[2] >> (24 - i * 8)) & 0xff;
        hash[i + 12] = (ctx->state[3] >> (24 - i * 8)) & 0xff;
        hash[i + 16] = (ctx->state[4] >> (24 - i * 8)) & 0xff;
        hash[i + 20] = (ctx->state[5] >> (24 - i * 8)) & 0xff;
        hash[i + 24] = (ctx->state[6] >> (24 - i * 8)) & 0xff;
        hash[i + 28] = (ctx->state[7] >> (24 - i * 8)) & 0xff;
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * SHA256 from buffer (not file)
 * ═══════════════════════════════════════════════════════════════════ */

static void sha256_buffer(const char *data, size_t len, char sha_hex[65]) {
    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, (const uint8_t *)data, len);
    uint8_t hash[32];
    sha256_final(&ctx, hash);
    for (int i = 0; i < 32; i++) sprintf(sha_hex + i * 2, "%02x", hash[i]);
    sha_hex[64] = '\0';
}

/* ═══════════════════════════════════════════════════════════════════
 * FILE UTILITIES
 * ═══════════════════════════════════════════════════════════════════ */

static int is_supported_ext(const char *path) {
    const char *dot = strrchr(path, '.');
    if (!dot) return 0;
    return !strcasecmp(dot, ".md") || !strcasecmp(dot, ".txt") ||
           !strcasecmp(dot, ".json") || !strcasecmp(dot, ".html") ||
           !strcasecmp(dot, ".csv");
}

static const char *detect_source_type(const char *path) {
    const char *dot = strrchr(path, '.');
    return dot ? dot + 1 : "bin";
}

static int is_markdown_heading(const char *line, int *level_out) {
    int level = 0;
    while (*line == '#') {
        level++;
        line++;
    }
    if (level > 0 && level <= 6 && (*line == ' ' || *line == '\t')) {
        if (level_out) *level_out = level;
        return 1;
    }
    return 0;
}

static char *trim_inplace(char *s) {
    while (isspace((unsigned char)*s)) s++;
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) end--;
    *end = '\0';
    return s;
}

static char *read_file_all(const char *path, size_t *out_len, char sha_hex[65], off_t *out_size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "warn: cannot open %s: %s\n", path, strerror(errno));
        return NULL;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        fprintf(stderr, "warn: cannot stat %s: %s\n", path, strerror(errno));
        return NULL;
    }
    char *buf = xmalloc((size_t)st.st_size + 1);
    size_t off = 0;
    SHA256_CTX ctx;
    sha256_init(&ctx);
    while (off < (size_t)st.st_size) {
        ssize_t n = read(fd, buf + off, (size_t)st.st_size - off);
        if (n < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "warn: cannot read %s: %s\n", path, strerror(errno));
            free(buf);
            close(fd);
            return NULL;
        }
        if (n == 0) break;
        sha256_update(&ctx, (const uint8_t *)(buf + off), (size_t)n);
        off += (size_t)n;
    }
    close(fd);
    buf[off] = '\0';
    uint8_t hash[32];
    sha256_final(&ctx, hash);
    for (int i = 0; i < 32; i++) sprintf(sha_hex + i * 2, "%02x", hash[i]);
    sha_hex[64] = '\0';
    if (out_len) *out_len = off;
    if (out_size) *out_size = st.st_size;
    return buf;
}

static char *to_absolute_path(const char *path) {
    char *resolved = realpath(path, NULL);
    if (resolved) return resolved;
    return xstrdup(path);
}

/* ═══════════════════════════════════════════════════════════════════
 * SQL UTILITIES
 * ═══════════════════════════════════════════════════════════════════ */

static int exec_sql(sqlite3 *db, const char *sql) {
    char *err = NULL;
    int rc = sqlite3_exec(db, sql, NULL, NULL, &err);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "sqlite exec failed: %s\nSQL: %s\n", err ? err : "?", sql);
        sqlite3_free(err);
    }
    return rc;
}

static int begin_tx(sqlite3 *db) {
    if (exec_sql(db, "BEGIN IMMEDIATE;") != SQLITE_OK) {
        die_sqlite(db, "begin tx");
        return -1;
    }
    return 0;
}

static int commit_tx(sqlite3 *db) {
    if (exec_sql(db, "COMMIT;") != SQLITE_OK) {
        die_sqlite(db, "commit tx");
        return -1;
    }
    return 0;
}

static int table_has_column(sqlite3 *db, const char *table, const char *column) {
    sqlite3_stmt *stmt = NULL;
    char sql[256];
    snprintf(sql, sizeof(sql), "PRAGMA table_info(%s);", table);
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) {
        die_sqlite(db, "prepare table_info");
        return 0;
    }
    int found = 0;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const unsigned char *name = sqlite3_column_text(stmt, 1);
        if (name && !strcmp((const char *)name, column)) {
            found = 1;
            break;
        }
    }
    sqlite3_finalize(stmt);
    return found;
}

static void ensure_column(sqlite3 *db, const char *table, const char *column, const char *definition) {
    if (table_has_column(db, table, column)) return;
    char sql[512];
    snprintf(sql, sizeof(sql), "ALTER TABLE %s ADD COLUMN %s %s;", table, column, definition);
    if (exec_sql(db, sql) != SQLITE_OK) die_sqlite(db, "alter table add column");
}

/* ═══════════════════════════════════════════════════════════════════
 * SCHEMA
 * ═══════════════════════════════════════════════════════════════════ */

static int ensure_schema(sqlite3 *db) {
    const char *sql_core =
        "PRAGMA journal_mode=WAL;"
        "PRAGMA foreign_keys=ON;"
        "CREATE TABLE IF NOT EXISTS namespaces ("
        "  id INTEGER PRIMARY KEY,"
        "  name TEXT NOT NULL UNIQUE,"
        "  scope TEXT NOT NULL,"
        "  visibility TEXT NOT NULL,"
        "  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"
        ");"
        "CREATE TABLE IF NOT EXISTS documents ("
        "  id INTEGER PRIMARY KEY,"
        "  namespace_id INTEGER NOT NULL REFERENCES namespaces(id),"
        "  path TEXT NOT NULL,"
        "  filename TEXT NOT NULL,"
        "  source_type TEXT NOT NULL,"
        "  size_bytes INTEGER NOT NULL DEFAULT 0,"
        "  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "  latest_version_id INTEGER,"
        "  UNIQUE(namespace_id, path)"
        ");"
        "CREATE TABLE IF NOT EXISTS document_versions ("
        "  id INTEGER PRIMARY KEY,"
        "  document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,"
        "  version_num INTEGER NOT NULL,"
        "  sha256 TEXT NOT NULL,"
        "  content TEXT NOT NULL,"
        "  ingest_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "  previous_version_id INTEGER REFERENCES document_versions(id),"
        "  is_latest INTEGER NOT NULL DEFAULT 1,"
        "  trust REAL NOT NULL DEFAULT 0.60,"
        "  UNIQUE(document_id, version_num)"
        ");"
        "CREATE TABLE IF NOT EXISTS sections ("
        "  id INTEGER PRIMARY KEY,"
        "  version_id INTEGER NOT NULL REFERENCES document_versions(id) ON DELETE CASCADE,"
        "  parent_document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,"
        "  ordinal INTEGER NOT NULL,"
        "  heading TEXT NOT NULL,"
        "  heading_level INTEGER NOT NULL DEFAULT 0,"
        "  char_start INTEGER NOT NULL,"
        "  char_end INTEGER NOT NULL,"
        "  body_text TEXT NOT NULL"
        ");"
        "CREATE TABLE IF NOT EXISTS chunks ("
        "  id INTEGER PRIMARY KEY,"
        "  version_id INTEGER NOT NULL REFERENCES document_versions(id) ON DELETE CASCADE,"
        "  parent_document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,"
        "  parent_section_id INTEGER NOT NULL REFERENCES sections(id) ON DELETE CASCADE,"
        "  ordinal INTEGER NOT NULL,"
        "  char_start INTEGER NOT NULL,"
        "  char_end INTEGER NOT NULL,"
        "  token_estimate INTEGER NOT NULL,"
        "  raw_text TEXT NOT NULL"
        ");"
        "CREATE TABLE IF NOT EXISTS links ("
        "  id INTEGER PRIMARY KEY,"
        "  from_chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,"
        "  to_chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,"
        "  kind INTEGER NOT NULL,"
        "  weight REAL NOT NULL DEFAULT 1.0,"
        "  reason TEXT NOT NULL"
        ");"
        "CREATE TABLE IF NOT EXISTS retrieval_log ("
        "  id INTEGER PRIMARY KEY,"
        "  query_text TEXT NOT NULL,"
        "  access_scope TEXT NOT NULL,"
        "  namespace_filter TEXT,"
        "  mode TEXT NOT NULL,"
        "  top_k INTEGER NOT NULL,"
        "  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"
        ");"
        "CREATE TABLE IF NOT EXISTS model_registry ("
        "  id INTEGER PRIMARY KEY,"
        "  model_name TEXT NOT NULL UNIQUE,"
        "  scope_default TEXT NOT NULL,"
        "  namespace_default TEXT NOT NULL,"
        "  retrieval_mode_default TEXT NOT NULL DEFAULT 'compressed',"
        "  packet_mode_default TEXT NOT NULL DEFAULT 'deterministic-json-v1',"
        "  query_profile_default TEXT NOT NULL DEFAULT 'balanced',"
        "  created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "  updated_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "  notes TEXT NOT NULL DEFAULT '',"
        "  is_active INTEGER NOT NULL DEFAULT 1,"
        "  detached_ts TEXT"
        ");"
        "CREATE TABLE IF NOT EXISTS namespace_manifest ("
        "  id INTEGER PRIMARY KEY,"
        "  namespace TEXT NOT NULL,"
        "  scope TEXT NOT NULL,"
        "  description TEXT NOT NULL,"
        "  owner_model TEXT,"
        "  created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "  updated_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "  UNIQUE(namespace, scope)"
        ");"
        "CREATE TABLE IF NOT EXISTS model_attachment_events ("
        "  id INTEGER PRIMARY KEY,"
        "  model_name TEXT NOT NULL,"
        "  event_type TEXT NOT NULL,"
        "  old_scope TEXT,"
        "  new_scope TEXT,"
        "  old_namespace TEXT,"
        "  new_namespace TEXT,"
        "  old_retrieval_mode TEXT,"
        "  new_retrieval_mode TEXT,"
        "  old_packet_mode TEXT,"
        "  new_packet_mode TEXT,"
        "  old_query_profile TEXT,"
        "  new_query_profile TEXT,"
        "  note TEXT NOT NULL DEFAULT '',"
        "  created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"
        ");"
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5("
        "  chunk_id UNINDEXED, raw_text, namespace, scope, path, filename, section_title"
        ");";
    const char *sql_indexes =
        "CREATE INDEX IF NOT EXISTS idx_documents_namespace_path ON documents(namespace_id, path);"
        "CREATE INDEX IF NOT EXISTS idx_versions_doc_latest ON document_versions(document_id, is_latest, version_num DESC);"
        "CREATE INDEX IF NOT EXISTS idx_versions_sha ON document_versions(document_id, sha256);"
        "CREATE INDEX IF NOT EXISTS idx_sections_version_ord ON sections(version_id, ordinal);"
        "CREATE INDEX IF NOT EXISTS idx_chunks_version_ord ON chunks(version_id, ordinal);"
        "CREATE INDEX IF NOT EXISTS idx_links_from_kind ON links(from_chunk_id, kind);"
        "CREATE INDEX IF NOT EXISTS idx_links_to_kind ON links(to_chunk_id, kind);"
        "CREATE INDEX IF NOT EXISTS idx_model_registry_active ON model_registry(is_active, model_name);"
        "CREATE INDEX IF NOT EXISTS idx_namespace_manifest_scope ON namespace_manifest(scope, namespace);"
        "CREATE INDEX IF NOT EXISTS idx_model_attachment_events_name_ts ON model_attachment_events(model_name, created_ts, id);";
    if (exec_sql(db, sql_core) != SQLITE_OK) { die_sqlite(db, "schema init core"); return -1; }
    if (exec_sql(db, sql_indexes) != SQLITE_OK) { die_sqlite(db, "schema init indexes"); return -1; }

    /* chunk_meta — PostGPT-style statistical fingerprints per chunk */
    exec_sql(db,
        "CREATE TABLE IF NOT EXISTS chunk_meta ("
        "  chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,"
        "  affinity BLOB,"
        "  bigram_data BLOB,"
        "  hebbian_data BLOB,"
        "  n_tokens INTEGER NOT NULL DEFAULT 0"
        ");"
        "CREATE INDEX IF NOT EXISTS idx_chunk_meta_id ON chunk_meta(chunk_id);");

    ensure_column(db, "document_versions", "first_seen_ts", "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP");
    ensure_column(db, "document_versions", "last_seen_ts", "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP");
    ensure_column(db, "document_versions", "seen_count", "INTEGER NOT NULL DEFAULT 1");
    ensure_column(db, "document_versions", "char_delta", "INTEGER NOT NULL DEFAULT 0");
    ensure_column(db, "document_versions", "token_delta", "INTEGER NOT NULL DEFAULT 0");
    ensure_column(db, "document_versions", "change_ratio", "REAL NOT NULL DEFAULT 0.0");
    ensure_column(db, "document_versions", "diff_summary", "TEXT NOT NULL DEFAULT ''");
    ensure_column(db, "model_registry", "retrieval_mode_default", "TEXT NOT NULL DEFAULT 'compressed'");
    ensure_column(db, "model_registry", "packet_mode_default", "TEXT NOT NULL DEFAULT 'deterministic-json-v1'");
    ensure_column(db, "model_registry", "query_profile_default", "TEXT NOT NULL DEFAULT 'balanced'");
    ensure_column(db, "model_registry", "created_ts", "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP");
    ensure_column(db, "model_registry", "updated_ts", "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP");
    ensure_column(db, "model_registry", "notes", "TEXT NOT NULL DEFAULT ''");
    ensure_column(db, "model_registry", "is_active", "INTEGER NOT NULL DEFAULT 1");
    ensure_column(db, "model_registry", "detached_ts", "TEXT");
    ensure_column(db, "namespace_manifest", "owner_model", "TEXT");
    ensure_column(db, "namespace_manifest", "created_ts", "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP");
    ensure_column(db, "namespace_manifest", "updated_ts", "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP");
    ensure_column(db, "model_attachment_events", "note", "TEXT NOT NULL DEFAULT ''");
    ensure_column(db, "model_attachment_events", "old_query_profile", "TEXT");
    ensure_column(db, "model_attachment_events", "new_query_profile", "TEXT");

    exec_sql(db, "UPDATE document_versions SET first_seen_ts=COALESCE(first_seen_ts, ingest_ts), last_seen_ts=COALESCE(last_seen_ts, ingest_ts), seen_count=COALESCE(seen_count, 1), char_delta=COALESCE(char_delta, 0), token_delta=COALESCE(token_delta, 0), change_ratio=COALESCE(change_ratio, 0.0), diff_summary=COALESCE(diff_summary, '');");
    exec_sql(db, "UPDATE model_registry SET retrieval_mode_default=COALESCE(retrieval_mode_default, 'compressed'), packet_mode_default=COALESCE(packet_mode_default, 'deterministic-json-v1'), query_profile_default=COALESCE(query_profile_default, 'balanced'), created_ts=COALESCE(created_ts, CURRENT_TIMESTAMP), updated_ts=COALESCE(updated_ts, CURRENT_TIMESTAMP), notes=COALESCE(notes, ''), is_active=COALESCE(is_active, 1);");
    return 0;
}

static sqlite3 *open_db(const char *db_path) {
    sqlite3 *db = NULL;
    if (sqlite3_open(db_path, &db) != SQLITE_OK) {
        die_sqlite(db, "open db");
        return NULL;
    }
    if (ensure_schema(db) != 0) {
        sqlite3_close(db);
        return NULL;
    }
    return db;
}

/* ═══════════════════════════════════════════════════════════════════
 * VALIDATION
 * ═══════════════════════════════════════════════════════════════════ */

static int validate_scope(const char *scope) {
    if (!scope || !*scope) return 0;
    if (!strcmp(scope, "public")) return 1;
    if (!strncmp(scope, "private:", 8) && scope[8] != '\0') return 1;
    if (!strncmp(scope, "shared:", 7) && scope[7] != '\0') return 1;
    return 0;
}

static int require_scope_lib(const char *scope) {
    if (!validate_scope(scope)) {
        fprintf(stderr, "invalid scope '%s' (expected public | shared:<name> | private:<model>)\n", scope ? scope : "<null>");
        return -1;
    }
    return 0;
}

#ifdef KK_STANDALONE
static void require_scope(const char *scope) {
    if (!validate_scope(scope)) {
        fprintf(stderr, "invalid scope '%s' (expected public | shared:<name> | private:<model>)\n", scope ? scope : "<null>");
        exit(1);
    }
}

static void require_mode(const char *mode) {
    if (!strcmp(mode, "raw") || !strcmp(mode, "citation") || !strcmp(mode, "compressed")) return;
    fprintf(stderr, "invalid mode '%s' (expected raw | citation | compressed)\n", mode);
    exit(1);
}

static void require_nonempty(const char *label, const char *value) {
    if (value && *value) return;
    fprintf(stderr, "invalid %s: value must be non-empty\n", label);
    exit(1);
}

static void require_packet_mode(const char *mode) {
    if (!strcmp(mode, KK_DEFAULT_PACKET_MODE)) return;
    fprintf(stderr, "invalid packet mode '%s' (expected %s)\n", mode, KK_DEFAULT_PACKET_MODE);
    exit(1);
}
#endif /* KK_STANDALONE */

static int validate_mode(const char *mode) {
    if (!mode) return 0;
    return !strcmp(mode, "raw") || !strcmp(mode, "citation") || !strcmp(mode, "compressed");
}

static int validate_nonempty(const char *value) {
    return value && *value;
}

static int validate_packet_mode(const char *mode) {
    return mode && !strcmp(mode, KK_DEFAULT_PACKET_MODE);
}

/* ═══════════════════════════════════════════════════════════════════
 * QUERY PROFILES
 * ═══════════════════════════════════════════════════════════════════ */

static const QueryProfile *query_profiles(void) {
    static const QueryProfile profiles[] = {
        {"tiny", 2, 160, 96, 96, 96, 0, 0, 0, 0, 0},
        {"balanced", 4, 240, 160, 160, 140, 1, 1, 1, 1, 0},
        {"deep", 6, 420, 256, 256, 220, 2, 2, 2, 1, 1},
    };
    return profiles;
}

static int query_profile_count(void) {
    return 3;
}

static const QueryProfile *find_query_profile(const char *name) {
    const QueryProfile *profiles = query_profiles();
    for (int i = 0; i < query_profile_count(); i++) {
        if (!strcmp(profiles[i].name, name)) return &profiles[i];
    }
    return NULL;
}

static const QueryProfile *profile_from_enum(kk_profile p) {
    const QueryProfile *profiles = query_profiles();
    switch (p) {
        case KK_PROFILE_TINY:     return &profiles[0];
        case KK_PROFILE_BALANCED: return &profiles[1];
        case KK_PROFILE_DEEP:     return &profiles[2];
    }
    return &profiles[1];
}

#ifdef KK_STANDALONE
static void require_query_profile(const char *name) {
    if (find_query_profile(name)) return;
    fprintf(stderr, "invalid query profile '%s' (expected tiny | balanced | deep)\n", name ? name : "<null>");
    exit(1);
}
#endif

/* ═══════════════════════════════════════════════════════════════════
 * BUDGET & SCOPE HELPERS
 * ═══════════════════════════════════════════════════════════════════ */

static BudgetedText budget_text(const char *text, size_t limit) {
    BudgetedText out;
    memset(&out, 0, sizeof(out));
    if (!text) {
        out.text = xstrdup("");
        return out;
    }
    size_t len = strlen(text);
    if (len <= limit) {
        out.text = xstrdup(text);
        return out;
    }
    if (limit <= 3) {
        out.text = xstrndup("...", limit);
    } else {
        out.text = xstrndup(text, limit - 3);
        memcpy(out.text + (limit - 3), "...", 4);
    }
    out.truncated = 1;
    return out;
}

static const char *scope_visibility(const char *scope) {
    if (!strcmp(scope, "public")) return "public";
    if (!strncmp(scope, "shared:", 7)) return "shared";
    return "private";
}

/* ═══════════════════════════════════════════════════════════════════
 * JSON HELPERS
 * ═══════════════════════════════════════════════════════════════════ */

static void json_append_string(StrBuf *sb, const char *value) {
    sb_append_n(sb, "\"", 1);
    for (const unsigned char *p = (const unsigned char *)(value ? value : ""); *p; p++) {
        switch (*p) {
            case '\\': sb_append(sb, "\\\\"); break;
            case '"': sb_append(sb, "\\\""); break;
            case '\n': sb_append(sb, "\\n"); break;
            case '\r': sb_append(sb, "\\r"); break;
            case '\t': sb_append(sb, "\\t"); break;
            default:
                if (*p < 0x20) sb_appendf(sb, "\\u%04x", *p);
                else sb_append_n(sb, (const char *)p, 1);
                break;
        }
    }
    sb_append_n(sb, "\"", 1);
}

/* ═══════════════════════════════════════════════════════════════════
 * EXCERPT / LOCATOR / LINEAGE
 * ═══════════════════════════════════════════════════════════════════ */

static char *make_excerpt(const char *text, size_t limit) {
    StrBuf sb;
    sb_init(&sb);
    int prev_space = 0;
    size_t emitted = 0;
    for (const unsigned char *p = (const unsigned char *)(text ? text : ""); *p; p++) {
        unsigned char c = *p;
        if (c == '\n' || c == '\r' || c == '\t') c = ' ';
        if (isspace(c)) {
            if (!prev_space && emitted < limit) {
                sb_append_n(&sb, " ", 1);
                emitted++;
            }
            prev_space = 1;
            continue;
        }
        if (emitted >= limit) break;
        sb_append_n(&sb, (const char *)&c, 1);
        emitted++;
        prev_space = 0;
    }
    while (sb.len > 0 && sb.data[sb.len - 1] == ' ') sb.data[--sb.len] = '\0';
    if (text && strlen(text) > emitted) sb_append(&sb, "...");
    return sb.data;
}

static BudgetedText make_excerpt_budgeted(const char *text, size_t limit) {
    BudgetedText out;
    memset(&out, 0, sizeof(out));
    StrBuf sb;
    sb_init(&sb);
    int prev_space = 0;
    size_t emitted = 0;
    for (const unsigned char *p = (const unsigned char *)(text ? text : ""); *p; p++) {
        unsigned char c = *p;
        if (isspace(c)) {
            if (!prev_space && emitted > 0) {
                if (emitted >= limit) break;
                sb_append_n(&sb, " ", 1);
                emitted++;
            }
            prev_space = 1;
            continue;
        }
        if (emitted >= limit) break;
        sb_append_n(&sb, (const char *)&c, 1);
        emitted++;
        prev_space = 0;
    }
    while (sb.len > 0 && sb.data[sb.len - 1] == ' ') sb.data[--sb.len] = '\0';
    if (text && strlen(text) > emitted) {
        sb_append(&sb, "...");
        out.truncated = 1;
    }
    out.text = sb.data;
    return out;
}

static char *make_locator(const QueryResult *r) {
    StrBuf sb;
    sb_init(&sb);
    sb_append(&sb, r->path ? r->path : "");
    sb_appendf(&sb, "#version=%d;chunk=%d", r->version_num, r->chunk_id);
    if (r->section_title && *r->section_title) {
        sb_append(&sb, ";section=");
        for (const unsigned char *p = (const unsigned char *)r->section_title; *p; p++) {
            unsigned char c = *p;
            if (isalnum(c)) sb_append_n(&sb, (const char *)&c, 1);
            else if (c == ' ' || c == '-' || c == '_') sb_append_n(&sb, "-", 1);
        }
    }
    return sb.data;
}

static char *make_lineage_summary(const QueryResult *r) {
    StrBuf sb;
    sb_init(&sb);
    sb_appendf(&sb, "%s; seen=%d; char_delta=%+d; token_delta=%+d; change_ratio=%.2f",
               r->diff_summary ? r->diff_summary : "", r->seen_count, r->char_delta, r->token_delta, r->change_ratio);
    return sb.data;
}

/* ═══════════════════════════════════════════════════════════════════
 * MODEL / NAMESPACE FETCH
 * ═══════════════════════════════════════════════════════════════════ */

static void free_model_attachment(ModelAttachment *model) {
    if (!model) return;
    free(model->model_name);
    free(model->scope_default);
    free(model->namespace_default);
    free(model->retrieval_mode_default);
    free(model->packet_mode_default);
    free(model->query_profile_default);
    free(model->notes);
    memset(model, 0, sizeof(*model));
}

static void free_namespace_manifest(NamespaceManifest *manifest) {
    if (!manifest) return;
    free(manifest->namespace_name);
    free(manifest->scope);
    free(manifest->description);
    free(manifest->owner_model);
    free(manifest->created_ts);
    free(manifest->updated_ts);
    memset(manifest, 0, sizeof(*manifest));
}

static ModelAttachment fetch_model_attachment_with_state(sqlite3 *db, const char *model_name, int active_only) {
    ModelAttachment model;
    memset(&model, 0, sizeof(model));
    sqlite3_stmt *stmt = NULL;
    const char *sql_active = "SELECT model_name, scope_default, namespace_default, retrieval_mode_default, packet_mode_default, query_profile_default, notes, is_active FROM model_registry WHERE model_name=? AND is_active=1;";
    const char *sql_any = "SELECT model_name, scope_default, namespace_default, retrieval_mode_default, packet_mode_default, query_profile_default, notes, is_active FROM model_registry WHERE model_name=?;";
    const char *sql = active_only ? sql_active : sql_any;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(db, "prepare model select"); return model; }
    sqlite3_bind_text(stmt, 1, model_name, -1, SQLITE_STATIC);
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        model.model_name = xstrdup((const char *)sqlite3_column_text(stmt, 0));
        model.scope_default = xstrdup((const char *)sqlite3_column_text(stmt, 1));
        model.namespace_default = xstrdup((const char *)sqlite3_column_text(stmt, 2));
        model.retrieval_mode_default = xstrdup((const char *)sqlite3_column_text(stmt, 3));
        model.packet_mode_default = xstrdup((const char *)sqlite3_column_text(stmt, 4));
        model.query_profile_default = xstrdup((const char *)sqlite3_column_text(stmt, 5));
        model.notes = xstrdup((const char *)sqlite3_column_text(stmt, 6));
        model.is_active = sqlite3_column_int(stmt, 7);
        model.found = 1;
    }
    sqlite3_finalize(stmt);
    return model;
}

static ModelAttachment fetch_model_attachment(sqlite3 *db, const char *model_name) {
    return fetch_model_attachment_with_state(db, model_name, 1);
}

static NamespaceManifest fetch_namespace_manifest(sqlite3 *db, const char *namespace_name, const char *scope) {
    NamespaceManifest manifest;
    memset(&manifest, 0, sizeof(manifest));
    sqlite3_stmt *stmt = NULL;
    const char *sql =
        "SELECT namespace, scope, description, COALESCE(owner_model, ''), created_ts, updated_ts "
        "FROM namespace_manifest WHERE namespace=? AND scope=?;";
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(db, "prepare namespace manifest select"); return manifest; }
    sqlite3_bind_text(stmt, 1, namespace_name, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, scope, -1, SQLITE_STATIC);
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        manifest.namespace_name = xstrdup((const char *)sqlite3_column_text(stmt, 0));
        manifest.scope = xstrdup((const char *)sqlite3_column_text(stmt, 1));
        manifest.description = xstrdup((const char *)sqlite3_column_text(stmt, 2));
        manifest.owner_model = xstrdup((const char *)sqlite3_column_text(stmt, 3));
        manifest.created_ts = xstrdup((const char *)sqlite3_column_text(stmt, 4));
        manifest.updated_ts = xstrdup((const char *)sqlite3_column_text(stmt, 5));
        manifest.found = 1;
    }
    sqlite3_finalize(stmt);
    return manifest;
}

static int namespace_manifest_count(sqlite3 *db, const char *namespace_name) {
    sqlite3_stmt *stmt = NULL;
    const char *sql = "SELECT COUNT(*) FROM namespace_manifest WHERE namespace=?;";
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(db, "prepare namespace manifest count"); return 0; }
    sqlite3_bind_text(stmt, 1, namespace_name, -1, SQLITE_STATIC);
    int count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) count = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
    return count;
}

static char *fetch_namespace_manifest_scope(sqlite3 *db, const char *namespace_name) {
    sqlite3_stmt *stmt = NULL;
    const char *sql = "SELECT scope FROM namespace_manifest WHERE namespace=? ORDER BY scope ASC LIMIT 1;";
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(db, "prepare namespace manifest scope"); return NULL; }
    sqlite3_bind_text(stmt, 1, namespace_name, -1, SQLITE_STATIC);
    char *scope = NULL;
    if (sqlite3_step(stmt) == SQLITE_ROW) scope = xstrdup((const char *)sqlite3_column_text(stmt, 0));
    sqlite3_finalize(stmt);
    return scope;
}

static int build_resolution_order(const ModelAttachment *model, const char **scopes, int max_scopes) {
    int count = 0;
    (void)max_scopes;
    scopes[count++] = model->scope_default;
    return count;
}

static int validate_model_binding_lib(const char *model_name, const char *scope_default, const char *namespace_default) {
    if (!validate_nonempty(model_name)) return -1;
    if (!validate_nonempty(namespace_default)) return -1;
    if (!validate_scope(scope_default)) return -1;
    if (!strncmp(scope_default, "private:", 8) && strcmp(scope_default + 8, model_name)) return -1;
    return 0;
}

#ifdef KK_STANDALONE
static void validate_model_binding(const char *model_name, const char *scope_default, const char *namespace_default) {
    require_nonempty("model_name", model_name);
    require_nonempty("namespace_default", namespace_default);
    require_scope(scope_default);
    if (!strncmp(scope_default, "private:", 8) && strcmp(scope_default + 8, model_name)) {
        fprintf(stderr, "private scope must exactly match model_name (expected private:%s)\n", model_name);
        exit(1);
    }
}
#endif

static int require_manifest_bound_namespace_lib(sqlite3 *db, const char *namespace_default, const char *scope_default) {
    NamespaceManifest manifest = fetch_namespace_manifest(db, namespace_default, scope_default);
    if (manifest.found) {
        free_namespace_manifest(&manifest);
        return 0;
    }
    free_namespace_manifest(&manifest);
    char *manifest_scope = fetch_namespace_manifest_scope(db, namespace_default);
    if (!manifest_scope) {
        fprintf(stderr, "namespace manifest missing: namespace=%s scope=%s\n", namespace_default, scope_default);
        return -1;
    }
    fprintf(stderr, "namespace scope mismatch: namespace=%s manifest_scope=%s requested=%s\n",
            namespace_default, manifest_scope, scope_default);
    free(manifest_scope);
    return -1;
}

#ifdef KK_STANDALONE
static void require_manifest_bound_namespace(sqlite3 *db, const char *namespace_default, const char *scope_default) {
    if (require_manifest_bound_namespace_lib(db, namespace_default, scope_default) != 0) exit(1);
}
#endif

static const char *model_event_type_from_prior(const ModelAttachment *prior) {
    if (!prior || !prior->found || !prior->model_name) return "attach";
    if (!prior->is_active) return "attach";
    return "update";
}

static void append_model_attachment_event(sqlite3 *db, const char *model_name, const char *event_type,
                                          const char *old_scope, const char *new_scope,
                                          const char *old_namespace, const char *new_namespace,
                                          const char *old_retrieval_mode, const char *new_retrieval_mode,
                                          const char *old_packet_mode, const char *new_packet_mode,
                                          const char *old_query_profile, const char *new_query_profile,
                                          const char *note) {
    sqlite3_stmt *stmt = NULL;
    const char *sql =
        "INSERT INTO model_attachment_events("
        "model_name, event_type, old_scope, new_scope, old_namespace, new_namespace, "
        "old_retrieval_mode, new_retrieval_mode, old_packet_mode, new_packet_mode, old_query_profile, new_query_profile, note, created_ts"
        ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP);";
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(db, "prepare model attachment event"); return; }
    sqlite3_bind_text(stmt, 1, model_name, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, event_type, -1, SQLITE_STATIC);
    if (old_scope) sqlite3_bind_text(stmt, 3, old_scope, -1, SQLITE_STATIC); else sqlite3_bind_null(stmt, 3);
    if (new_scope) sqlite3_bind_text(stmt, 4, new_scope, -1, SQLITE_STATIC); else sqlite3_bind_null(stmt, 4);
    if (old_namespace) sqlite3_bind_text(stmt, 5, old_namespace, -1, SQLITE_STATIC); else sqlite3_bind_null(stmt, 5);
    if (new_namespace) sqlite3_bind_text(stmt, 6, new_namespace, -1, SQLITE_STATIC); else sqlite3_bind_null(stmt, 6);
    if (old_retrieval_mode) sqlite3_bind_text(stmt, 7, old_retrieval_mode, -1, SQLITE_STATIC); else sqlite3_bind_null(stmt, 7);
    if (new_retrieval_mode) sqlite3_bind_text(stmt, 8, new_retrieval_mode, -1, SQLITE_STATIC); else sqlite3_bind_null(stmt, 8);
    if (old_packet_mode) sqlite3_bind_text(stmt, 9, old_packet_mode, -1, SQLITE_STATIC); else sqlite3_bind_null(stmt, 9);
    if (new_packet_mode) sqlite3_bind_text(stmt, 10, new_packet_mode, -1, SQLITE_STATIC); else sqlite3_bind_null(stmt, 10);
    if (old_query_profile) sqlite3_bind_text(stmt, 11, old_query_profile, -1, SQLITE_STATIC); else sqlite3_bind_null(stmt, 11);
    if (new_query_profile) sqlite3_bind_text(stmt, 12, new_query_profile, -1, SQLITE_STATIC); else sqlite3_bind_null(stmt, 12);
    sqlite3_bind_text(stmt, 13, note ? note : "", -1, SQLITE_STATIC);
    if (sqlite3_step(stmt) != SQLITE_DONE) die_sqlite(db, "insert model attachment event");
    sqlite3_finalize(stmt);
}

/* ═══════════════════════════════════════════════════════════════════
 * NAMESPACE / DOCUMENT HELPERS
 * ═══════════════════════════════════════════════════════════════════ */

static int get_namespace_id(sqlite3 *db, const char *name, const char *scope) {
    const char *visibility = scope_visibility(scope);
    sqlite3_stmt *stmt = NULL;
    const char *sel = "SELECT id, scope FROM namespaces WHERE name=?;";
    if (sqlite3_prepare_v2(db, sel, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(db, "prepare namespace select"); return -1; }
    sqlite3_bind_text(stmt, 1, name, -1, SQLITE_STATIC);
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        int id = sqlite3_column_int(stmt, 0);
        const char *existing_scope = (const char *)sqlite3_column_text(stmt, 1);
        if (existing_scope && strcmp(existing_scope, scope)) {
            fprintf(stderr, "namespace scope mismatch: namespace=%s existing=%s requested=%s\n", name, existing_scope, scope);
            sqlite3_finalize(stmt);
            return -1;
        }
        sqlite3_finalize(stmt);
        return id;
    }
    sqlite3_finalize(stmt);

    const char *ins = "INSERT INTO namespaces(name, scope, visibility) VALUES(?, ?, ?) RETURNING id;";
    if (sqlite3_prepare_v2(db, ins, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(db, "prepare namespace insert"); return -1; }
    sqlite3_bind_text(stmt, 1, name, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, scope, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, visibility, -1, SQLITE_STATIC);
    if (sqlite3_step(stmt) != SQLITE_ROW) { die_sqlite(db, "insert namespace"); sqlite3_finalize(stmt); return -1; }
    int id = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
    return id;
}

static int estimate_tokens(const char *text) {
    int count = 0, in_token = 0;
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
        if (isalnum(*p)) {
            if (!in_token) count++;
            in_token = 1;
        } else {
            in_token = 0;
        }
    }
    return count > 0 ? count : 1;
}

static int contains_word_ci(const char *haystack, const char *needle) {
    size_t n = strlen(needle);
    if (n == 0) return 0;
    for (const char *p = haystack; *p; p++) {
        if (!strncasecmp(p, needle, n)) return 1;
    }
    return 0;
}

static double clamp01(double v) {
    if (v < 0.0) return 0.0;
    if (v > 1.0) return 1.0;
    return v;
}

static int access_scope_allows(const char *doc_scope, const char *access_scope) {
    if (!strcmp(doc_scope, "public")) return 1;
    if (!strcmp(doc_scope, access_scope)) return 1;
    if (!strncmp(doc_scope, "shared:", 7) && !strcmp(doc_scope, access_scope)) return 1;
    return 0;
}

static double scope_compatibility(const char *doc_scope, const char *access_scope) {
    if (!strcmp(doc_scope, access_scope)) return 1.0;
    if (!strcmp(doc_scope, "public")) return 0.75;
    if (!strncmp(doc_scope, "shared:", 7) && !strcmp(doc_scope, access_scope)) return 0.95;
    return 0.0;
}

/* ═══════════════════════════════════════════════════════════════════
 * TEXT NORMALIZATION
 * ═══════════════════════════════════════════════════════════════════ */

static char *normalize_text(const char *input) {
    StrBuf out;
    sb_init(&out);
    int prev_space = 0;
    for (const unsigned char *p = (const unsigned char *)input; *p; p++) {
        unsigned char c = *p;
        if (c == '\r') continue;
        if (c == '\t') c = ' ';
        if (iscntrl(c) && c != '\n') c = ' ';
        if (c == '\n') {
            sb_append_n(&out, "\n", 1);
            prev_space = 0;
            continue;
        }
        if (isspace(c)) {
            if (!prev_space) sb_append_n(&out, " ", 1);
            prev_space = 1;
        } else {
            sb_append_n(&out, (char *)&c, 1);
            prev_space = 0;
        }
    }
    return out.data;
}

/* ═══════════════════════════════════════════════════════════════════
 * KEYWORD EXTRACTION
 * ═══════════════════════════════════════════════════════════════════ */

static int is_stopword(const char *tok) {
    static const char *words[] = {
        "the", "and", "that", "with", "from", "this", "there", "their", "have", "will",
        "would", "could", "should", "into", "about", "after", "before", "under", "over",
        "your", "ours", "they", "them", "were", "been", "than", "then", "what", "when",
        "where", "which", "while", "just", "also", "does", "done", "each", "more", "most",
        "such", "only", "very", "some", "much", "many", "into", "onto", "across", "because",
        "kernel", "knowledge"
    };
    for (size_t i = 0; i < sizeof(words) / sizeof(words[0]); i++) if (!strcmp(tok, words[i])) return 1;
    return 0;
}

static int keyword_exists(char terms[KK_MAX_KEYWORDS][KK_KEYWORD_LEN], int count, const char *tok) {
    for (int i = 0; i < count; i++) if (!strcmp(terms[i], tok)) return 1;
    return 0;
}

static int collect_keywords(const char *text, char terms[KK_MAX_KEYWORDS][KK_KEYWORD_LEN]) {
    int count = 0;
    char *copy = xstrdup(text ? text : "");
    char *save = NULL;
    for (char *tok = strtok_r(copy, " \t\n,.;:!?()[]{}<>\"'/-", &save);
         tok && count < KK_MAX_KEYWORDS;
         tok = strtok_r(NULL, " \t\n,.;:!?()[]{}<>\"'/-", &save)) {
        char lower[KK_KEYWORD_LEN];
        size_t n = strlen(tok);
        if (n < 4) continue;
        if (n >= sizeof(lower)) n = sizeof(lower) - 1;
        for (size_t i = 0; i < n; i++) lower[i] = (char)tolower((unsigned char)tok[i]);
        lower[n] = '\0';
        if (is_stopword(lower) || keyword_exists(terms, count, lower)) continue;
        strncpy(terms[count], lower, sizeof(terms[count]));
        terms[count][sizeof(terms[count]) - 1] = '\0';
        count++;
    }
    free(copy);
    return count;
}

static int shared_keyword_count(char a[KK_MAX_KEYWORDS][KK_KEYWORD_LEN], int ac, char b[KK_MAX_KEYWORDS][KK_KEYWORD_LEN], int bc) {
    int shared = 0;
    for (int i = 0; i < ac; i++) {
        for (int j = 0; j < bc; j++) {
            if (!strcmp(a[i], b[j])) {
                shared++;
                break;
            }
        }
    }
    return shared;
}

static int union_keyword_count(int ac, int bc, int shared) {
    int u = ac + bc - shared;
    return u > 0 ? u : 1;
}

/* ═══════════════════════════════════════════════════════════════════
 * SECTION / CHUNK SPLITTING
 * ═══════════════════════════════════════════════════════════════════ */

static void add_section(SectionInfo **arr, int *count, int *cap, const char *text, size_t start, size_t end, int level, int ordinal) {
    if (end <= start) return;
    if (*count == *cap) {
        *cap = *cap ? (*cap * 2) : 8;
        *arr = xrealloc(*arr, (size_t)*cap * sizeof(**arr));
    }
    (*arr)[*count].text = xstrndup(text + start, end - start);
    (*arr)[*count].start = start;
    (*arr)[*count].end = end;
    (*arr)[*count].level = level;
    (*arr)[*count].ordinal = ordinal;
    (*count)++;
}

static SectionInfo *split_sections(const char *text, int *out_count) {
    SectionInfo *sections = NULL;
    int count = 0, cap = 0;
    size_t len = strlen(text);
    size_t pos = 0, section_start = 0;
    int ordinal = 0;
    int current_level = 0;
    while (pos < len) {
        size_t line_start = pos;
        while (pos < len && text[pos] != '\n') pos++;
        size_t line_end = pos;
        if (pos < len && text[pos] == '\n') pos++;
        size_t line_len = line_end - line_start;
        char *line = xstrndup(text + line_start, line_len);
        int level = 0;
        if (is_markdown_heading(line, &level) && line_start != section_start) {
            add_section(&sections, &count, &cap, text, section_start, line_start, current_level, ordinal++);
            section_start = line_start;
            current_level = level;
        } else if (is_markdown_heading(line, &level) && line_start == section_start) {
            current_level = level;
        }
        free(line);
    }
    add_section(&sections, &count, &cap, text, section_start, len, current_level, ordinal++);
    if (count == 0) add_section(&sections, &count, &cap, text, 0, len, 0, 0);
    *out_count = count;
    return sections;
}

static char *section_heading_from_text(const char *text) {
    const char *line_end = strchr(text, '\n');
    size_t len = line_end ? (size_t)(line_end - text) : strlen(text);
    char *line = xstrndup(text, len);
    char *trimmed = trim_inplace(line);
    if (trimmed[0] == '#') {
        while (*trimmed == '#') trimmed++;
        while (*trimmed == ' ' || *trimmed == '\t') trimmed++;
    }
    char *res = xstrdup(*trimmed ? trimmed : "root");
    free(line);
    return res;
}

static void add_chunk(ChunkInfo **arr, int *count, int *cap, const char *text, size_t start, size_t end, int ordinal) {
    if (end <= start) return;
    while (start < end && isspace((unsigned char)text[start])) start++;
    while (end > start && isspace((unsigned char)text[end - 1])) end--;
    if (end <= start) return;
    if (*count == *cap) {
        *cap = *cap ? (*cap * 2) : 16;
        *arr = xrealloc(*arr, (size_t)*cap * sizeof(**arr));
    }
    (*arr)[*count].text = xstrndup(text + start, end - start);
    (*arr)[*count].start = start;
    (*arr)[*count].end = end;
    (*arr)[*count].ordinal = ordinal;
    (*count)++;
}

static ChunkInfo *split_chunks(const char *section_text, size_t section_base, int *out_count) {
    ChunkInfo *chunks = NULL;
    int count = 0, cap = 0, ordinal = 0;
    size_t len = strlen(section_text);
    size_t para_start = 0;
    for (size_t pos = 0; pos <= len; pos++) {
        int boundary = 0;
        if (pos == len) boundary = 1;
        else if (section_text[pos] == '\n' && pos + 1 < len && section_text[pos + 1] == '\n') boundary = 1;
        if (!boundary) continue;
        size_t para_end = pos;
        while (pos < len && section_text[pos] == '\n') pos++;
        size_t start = para_start;
        while (start < para_end && isspace((unsigned char)section_text[start])) start++;
        while (para_end > start && isspace((unsigned char)section_text[para_end - 1])) para_end--;
        if (para_end > start) {
            size_t seg = start;
            while (seg < para_end) {
                size_t take = para_end - seg;
                if (take > KK_MAX_CHUNK_CHARS) {
                    size_t cut = seg + KK_MAX_CHUNK_CHARS;
                    size_t best = cut;
                    for (size_t i = cut; i > seg + KK_MIN_CHUNK_CHARS; i--) {
                        if (isspace((unsigned char)section_text[i])) { best = i; break; }
                    }
                    take = best - seg;
                }
                add_chunk(&chunks, &count, &cap, section_text, seg, seg + take, ordinal++);
                seg += take;
                while (seg < para_end && isspace((unsigned char)section_text[seg])) seg++;
            }
        }
        para_start = pos;
    }
    if (count == 0) add_chunk(&chunks, &count, &cap, section_text, 0, len, 0);
    for (int i = 0; i < count; i++) {
        chunks[i].start += section_base;
        chunks[i].end += section_base;
    }
    *out_count = count;
    return chunks;
}

/* ═══════════════════════════════════════════════════════════════════
 * DB INSERT HELPERS
 * ═══════════════════════════════════════════════════════════════════ */

static int insert_document(sqlite3 *db, int namespace_id, const char *path, const char *filename, const char *source_type, sqlite3_int64 size_bytes) {
    sqlite3_stmt *stmt = NULL;
    const char *sql = "INSERT INTO documents(namespace_id, path, filename, source_type, size_bytes) VALUES(?, ?, ?, ?, ?) "
                      "ON CONFLICT(namespace_id, path) DO UPDATE SET filename=excluded.filename, source_type=excluded.source_type, size_bytes=excluded.size_bytes "
                      "RETURNING id;";
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(db, "prepare insert document"); return -1; }
    sqlite3_bind_int(stmt, 1, namespace_id);
    sqlite3_bind_text(stmt, 2, path, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, filename, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 4, source_type, -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 5, size_bytes);
    if (sqlite3_step(stmt) != SQLITE_ROW) { die_sqlite(db, "insert document"); sqlite3_finalize(stmt); return -1; }
    int id = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
    return id;
}

static int get_latest_version(sqlite3 *db, int document_id, int *version_num, char sha_out[65], int *version_id) {
    sqlite3_stmt *stmt = NULL;
    const char *sql = "SELECT id, version_num, sha256 FROM document_versions WHERE document_id=? AND is_latest=1 ORDER BY version_num DESC LIMIT 1;";
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(db, "prepare latest version"); return 0; }
    sqlite3_bind_int(stmt, 1, document_id);
    int found = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        if (version_id) *version_id = sqlite3_column_int(stmt, 0);
        if (version_num) *version_num = sqlite3_column_int(stmt, 1);
        const unsigned char *sha = sqlite3_column_text(stmt, 2);
        if (sha_out) {
            if (sha) memcpy(sha_out, (const char *)sha, 64);
            sha_out[64] = '\0';
        }
        found = 1;
    }
    sqlite3_finalize(stmt);
    return found;
}

static int get_version_by_sha(sqlite3 *db, int document_id, const char *sha256, int *version_id, int *version_num, int *is_latest) {
    sqlite3_stmt *stmt = NULL;
    const char *sql = "SELECT id, version_num, is_latest FROM document_versions WHERE document_id=? AND sha256=? ORDER BY version_num DESC LIMIT 1;";
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(db, "prepare version by sha"); return 0; }
    sqlite3_bind_int(stmt, 1, document_id);
    sqlite3_bind_text(stmt, 2, sha256, -1, SQLITE_STATIC);
    int found = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        if (version_id) *version_id = sqlite3_column_int(stmt, 0);
        if (version_num) *version_num = sqlite3_column_int(stmt, 1);
        if (is_latest) *is_latest = sqlite3_column_int(stmt, 2);
        found = 1;
    }
    sqlite3_finalize(stmt);
    return found;
}

static char *get_version_content(sqlite3 *db, int version_id) {
    sqlite3_stmt *stmt = NULL;
    const char *sql = "SELECT content FROM document_versions WHERE id=?;";
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(db, "prepare version content"); return NULL; }
    sqlite3_bind_int(stmt, 1, version_id);
    char *content = NULL;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const unsigned char *text = sqlite3_column_text(stmt, 0);
        content = xstrdup((const char *)(text ? text : (const unsigned char *)""));
    }
    sqlite3_finalize(stmt);
    return content;
}

static void set_latest_version(sqlite3 *db, int doc_id, int version_id) {
    sqlite3_stmt *stmt = NULL;
    const char *sql = "UPDATE document_versions SET is_latest = CASE WHEN id=? THEN 1 ELSE 0 END WHERE document_id=?;";
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(db, "prepare set latest version"); return; }
    sqlite3_bind_int(stmt, 1, version_id);
    sqlite3_bind_int(stmt, 2, doc_id);
    if (sqlite3_step(stmt) != SQLITE_DONE) die_sqlite(db, "set latest version");
    sqlite3_finalize(stmt);

    const char *upd_doc = "UPDATE documents SET latest_version_id=? WHERE id=?;";
    if (sqlite3_prepare_v2(db, upd_doc, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(db, "prepare update document latest"); return; }
    sqlite3_bind_int(stmt, 1, version_id);
    sqlite3_bind_int(stmt, 2, doc_id);
    if (sqlite3_step(stmt) != SQLITE_DONE) die_sqlite(db, "update document latest");
    sqlite3_finalize(stmt);
}

static void touch_version_seen(sqlite3 *db, int version_id) {
    sqlite3_stmt *stmt = NULL;
    const char *sql = "UPDATE document_versions SET last_seen_ts=CURRENT_TIMESTAMP, seen_count=COALESCE(seen_count,0)+1 WHERE id=?;";
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(db, "prepare touch version"); return; }
    sqlite3_bind_int(stmt, 1, version_id);
    if (sqlite3_step(stmt) != SQLITE_DONE) die_sqlite(db, "touch version");
    sqlite3_finalize(stmt);
}

/* ═══════════════════════════════════════════════════════════════════
 * DIFF METRICS
 * ═══════════════════════════════════════════════════════════════════ */

static int count_nonempty_lines(const char *text) {
    int count = 0;
    const char *p = text;
    while (*p) {
        const char *start = p;
        while (*p && *p != '\n') p++;
        const char *end = p;
        while (start < end && isspace((unsigned char)*start)) start++;
        while (end > start && isspace((unsigned char)end[-1])) end--;
        if (end > start) count++;
        if (*p == '\n') p++;
    }
    return count;
}

static int count_shared_lines(const char *a, const char *b) {
    int shared = 0;
    char *copy_a = xstrdup(a ? a : "");
    char *copy_b = xstrdup(b ? b : "");
    int bcap = 16, bcount = 0;
    char **lines_b = xmalloc((size_t)bcap * sizeof(*lines_b));

    char *save_b = NULL;
    for (char *line = strtok_r(copy_b, "\n", &save_b); line; line = strtok_r(NULL, "\n", &save_b)) {
        char *trimmed = trim_inplace(line);
        if (!*trimmed) continue;
        if (bcount == bcap) {
            bcap *= 2;
            lines_b = xrealloc(lines_b, (size_t)bcap * sizeof(*lines_b));
        }
        lines_b[bcount++] = trimmed;
    }

    char *save_a = NULL;
    for (char *line = strtok_r(copy_a, "\n", &save_a); line; line = strtok_r(NULL, "\n", &save_a)) {
        char *trimmed = trim_inplace(line);
        if (!*trimmed) continue;
        for (int i = 0; i < bcount; i++) {
            if (!strcmp(trimmed, lines_b[i])) {
                shared++;
                break;
            }
        }
    }

    free(lines_b);
    free(copy_a);
    free(copy_b);
    return shared;
}

static void compute_diff_metrics(const char *prev, const char *curr, int *char_delta, int *token_delta, double *change_ratio, char *summary, size_t summary_sz) {
    int prev_chars = (int)strlen(prev ? prev : "");
    int curr_chars = (int)strlen(curr ? curr : "");
    int prev_tokens = estimate_tokens(prev ? prev : "");
    int curr_tokens = estimate_tokens(curr ? curr : "");
    int prev_lines = count_nonempty_lines(prev ? prev : "");
    int curr_lines = count_nonempty_lines(curr ? curr : "");
    int shared_lines = count_shared_lines(prev ? prev : "", curr ? curr : "");
    double denom = (double)(prev_lines + curr_lines);
    double stability = denom > 0.0 ? (2.0 * (double)shared_lines / denom) : 1.0;

    if (char_delta) *char_delta = curr_chars - prev_chars;
    if (token_delta) *token_delta = curr_tokens - prev_tokens;
    if (change_ratio) *change_ratio = clamp01(1.0 - stability);
    snprintf(summary, summary_sz,
             "lines_shared=%d/%d->%d chars=%+d tokens=%+d delta_ratio=%.2f",
             shared_lines, prev_lines, curr_lines, curr_chars - prev_chars, curr_tokens - prev_tokens,
             clamp01(1.0 - stability));
}

/* ═══════════════════════════════════════════════════════════════════
 * VERSION INSERT
 * ═══════════════════════════════════════════════════════════════════ */

static int insert_version(sqlite3 *db, int doc_id, int version_num, const char *sha256, const char *content,
                          int prev_version_id, int char_delta, int token_delta, double change_ratio, const char *diff_summary) {
    sqlite3_stmt *stmt = NULL;
    const char *ins =
        "INSERT INTO document_versions(document_id, version_num, sha256, content, previous_version_id, char_delta, token_delta, change_ratio, diff_summary) "
        "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id;";
    if (sqlite3_prepare_v2(db, ins, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(db, "prepare insert version"); return -1; }
    sqlite3_bind_int(stmt, 1, doc_id);
    sqlite3_bind_int(stmt, 2, version_num);
    sqlite3_bind_text(stmt, 3, sha256, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 4, content, -1, SQLITE_STATIC);
    if (prev_version_id > 0) sqlite3_bind_int(stmt, 5, prev_version_id); else sqlite3_bind_null(stmt, 5);
    sqlite3_bind_int(stmt, 6, char_delta);
    sqlite3_bind_int(stmt, 7, token_delta);
    sqlite3_bind_double(stmt, 8, change_ratio);
    sqlite3_bind_text(stmt, 9, diff_summary, -1, SQLITE_STATIC);
    if (sqlite3_step(stmt) != SQLITE_ROW) { die_sqlite(db, "insert version"); sqlite3_finalize(stmt); return -1; }
    int vid = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
    set_latest_version(db, doc_id, vid);
    return vid;
}

/* ═══════════════════════════════════════════════════════════════════
 * SECTION / CHUNK / LINK / FTS INSERT
 * ═══════════════════════════════════════════════════════════════════ */

static int insert_section_row(sqlite3 *db, int version_id, int document_id, int ordinal, const char *heading, int level, int start, int end, const char *body_text) {
    sqlite3_stmt *stmt = NULL;
    const char *sql = "INSERT INTO sections(version_id, parent_document_id, ordinal, heading, heading_level, char_start, char_end, body_text) VALUES(?, ?, ?, ?, ?, ?, ?, ?) RETURNING id;";
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(db, "prepare section insert"); return -1; }
    sqlite3_bind_int(stmt, 1, version_id);
    sqlite3_bind_int(stmt, 2, document_id);
    sqlite3_bind_int(stmt, 3, ordinal);
    sqlite3_bind_text(stmt, 4, heading, -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 5, level);
    sqlite3_bind_int(stmt, 6, start);
    sqlite3_bind_int(stmt, 7, end);
    sqlite3_bind_text(stmt, 8, body_text, -1, SQLITE_STATIC);
    if (sqlite3_step(stmt) != SQLITE_ROW) { die_sqlite(db, "insert section"); sqlite3_finalize(stmt); return -1; }
    int id = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
    return id;
}

/* forward declarations for metaweights (defined after public query API) */
static int build_meta_from_text(const char *text, int text_len, kk_chunk_meta *meta);
static int store_chunk_meta(sqlite3 *db, int chunk_id, const kk_chunk_meta *meta);
static int load_chunk_meta(sqlite3 *db, int chunk_id, kk_chunk_meta *meta);

static int insert_chunk_row(sqlite3 *db, int version_id, int document_id, int section_id, int ordinal, int start, int end, int token_estimate, const char *raw_text) {
    sqlite3_stmt *stmt = NULL;
    const char *sql = "INSERT INTO chunks(version_id, parent_document_id, parent_section_id, ordinal, char_start, char_end, token_estimate, raw_text) VALUES(?, ?, ?, ?, ?, ?, ?, ?) RETURNING id;";
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(db, "prepare chunk insert"); return -1; }
    sqlite3_bind_int(stmt, 1, version_id);
    sqlite3_bind_int(stmt, 2, document_id);
    sqlite3_bind_int(stmt, 3, section_id);
    sqlite3_bind_int(stmt, 4, ordinal);
    sqlite3_bind_int(stmt, 5, start);
    sqlite3_bind_int(stmt, 6, end);
    sqlite3_bind_int(stmt, 7, token_estimate);
    sqlite3_bind_text(stmt, 8, raw_text, -1, SQLITE_STATIC);
    if (sqlite3_step(stmt) != SQLITE_ROW) { die_sqlite(db, "insert chunk"); sqlite3_finalize(stmt); return -1; }
    int id = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
    return id;
}

static void insert_link(sqlite3 *db, int from_chunk_id, int to_chunk_id, int kind, double weight, const char *reason) {
    sqlite3_stmt *stmt = NULL;
    const char *sql = "INSERT INTO links(from_chunk_id, to_chunk_id, kind, weight, reason) VALUES(?, ?, ?, ?, ?);";
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(db, "prepare insert link"); return; }
    sqlite3_bind_int(stmt, 1, from_chunk_id);
    sqlite3_bind_int(stmt, 2, to_chunk_id);
    sqlite3_bind_int(stmt, 3, kind);
    sqlite3_bind_double(stmt, 4, weight);
    sqlite3_bind_text(stmt, 5, reason, -1, SQLITE_STATIC);
    if (sqlite3_step(stmt) != SQLITE_DONE) die_sqlite(db, "insert link");
    sqlite3_finalize(stmt);
}

static void insert_fts_row(sqlite3 *db, int chunk_id, const char *raw_text, const char *namespace_name, const char *scope, const char *path, const char *filename, const char *section_title) {
    sqlite3_stmt *stmt = NULL;
    const char *sql = "INSERT INTO chunk_fts(chunk_id, raw_text, namespace, scope, path, filename, section_title) VALUES(?, ?, ?, ?, ?, ?, ?);";
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(db, "prepare insert fts"); return; }
    sqlite3_bind_int(stmt, 1, chunk_id);
    sqlite3_bind_text(stmt, 2, raw_text, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, namespace_name, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 4, scope, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 5, path, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 6, filename, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 7, section_title, -1, SQLITE_STATIC);
    if (sqlite3_step(stmt) != SQLITE_DONE) die_sqlite(db, "insert fts row");
    sqlite3_finalize(stmt);
}

/* ═══════════════════════════════════════════════════════════════════
 * LINK INFERENCE
 * ═══════════════════════════════════════════════════════════════════ */

static void infer_section_topology_links(sqlite3 *db, SectionRuntime *sections, int count) {
    int parent_stack[8];
    for (int i = 0; i < 8; i++) parent_stack[i] = -1;

    for (int i = 0; i < count; i++) {
        if (sections[i].first_chunk_id <= 0) continue;
        int level = sections[i].level;
        if (level < 0) level = 0;
        if (level >= 8) level = 7;

        if (i > 0 && sections[i - 1].first_chunk_id > 0) {
            const char *reason = (sections[i - 1].level == sections[i].level) ? "section sibling" : "section transition";
            double weight = (sections[i - 1].level == sections[i].level) ? 0.82 : 0.74;
            insert_link(db, sections[i - 1].first_chunk_id, sections[i].first_chunk_id, KK_LINK_KIND_STRUCTURAL, weight, reason);
            insert_link(db, sections[i].first_chunk_id, sections[i - 1].first_chunk_id, KK_LINK_KIND_STRUCTURAL, weight, reason);
        }

        for (int l = level; l < 8; l++) parent_stack[l] = -1;
        if (level > 0 && parent_stack[level - 1] >= 0) {
            int parent_idx = parent_stack[level - 1];
            if (sections[parent_idx].first_chunk_id > 0) {
                insert_link(db, sections[parent_idx].first_chunk_id, sections[i].first_chunk_id, KK_LINK_KIND_STRUCTURAL, 0.91, "section hierarchy");
                insert_link(db, sections[i].first_chunk_id, sections[parent_idx].first_chunk_id, KK_LINK_KIND_STRUCTURAL, 0.86, "section hierarchy");
            }
        }
        parent_stack[level] = i;
    }
}

static void free_link_chunks(LinkChunk *chunks, int count) {
    if (!chunks) return;
    for (int i = 0; i < count; i++) {
        free(chunks[i].text);
        free(chunks[i].heading);
    }
    free(chunks);
}

static void infer_related_links(sqlite3 *db, int version_id) {
    sqlite3_stmt *stmt = NULL;
    const char *sql =
        "SELECT c.id, c.ordinal, c.parent_section_id, c.token_estimate, c.raw_text, s.heading "
        "FROM chunks c JOIN sections s ON s.id = c.parent_section_id "
        "WHERE c.version_id=? ORDER BY c.ordinal;";
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(db, "prepare related links"); return; }
    sqlite3_bind_int(stmt, 1, version_id);

    int count = 0, cap = 16;
    LinkChunk *chunks = xmalloc((size_t)cap * sizeof(*chunks));
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        if (count == cap) {
            cap *= 2;
            chunks = xrealloc(chunks, (size_t)cap * sizeof(*chunks));
        }
        chunks[count].id = sqlite3_column_int(stmt, 0);
        chunks[count].ordinal = sqlite3_column_int(stmt, 1);
        chunks[count].section_id = sqlite3_column_int(stmt, 2);
        chunks[count].token_estimate = sqlite3_column_int(stmt, 3);
        chunks[count].text = xstrdup((const char *)sqlite3_column_text(stmt, 4));
        chunks[count].heading = xstrdup((const char *)sqlite3_column_text(stmt, 5));
        chunks[count].keyword_count = collect_keywords(chunks[count].text, chunks[count].keywords);
        count++;
    }
    sqlite3_finalize(stmt);

    for (int i = 0; i < count; i++) {
        for (int j = i + 1; j < count; j++) {
            int shared = shared_keyword_count(chunks[i].keywords, chunks[i].keyword_count,
                                              chunks[j].keywords, chunks[j].keyword_count);
            if (shared < 2) continue;
            int uni = union_keyword_count(chunks[i].keyword_count, chunks[j].keyword_count, shared);
            double jaccard = (double)shared / (double)uni;
            int heading_shared = 0;
            char hk1[KK_MAX_KEYWORDS][KK_KEYWORD_LEN] = {{0}};
            char hk2[KK_MAX_KEYWORDS][KK_KEYWORD_LEN] = {{0}};
            int hc1 = collect_keywords(chunks[i].heading, hk1);
            int hc2 = collect_keywords(chunks[j].heading, hk2);
            if (chunks[i].section_id == chunks[j].section_id) heading_shared = 1;
            else if (shared_keyword_count(hk1, hc1, hk2, hc2) > 0) heading_shared = 1;
            int ordinal_gap = abs(chunks[i].ordinal - chunks[j].ordinal);
            double proximity = 1.0 / (1.0 + (double)ordinal_gap);
            int min_tokens = chunks[i].token_estimate < chunks[j].token_estimate ? chunks[i].token_estimate : chunks[j].token_estimate;
            int max_tokens = chunks[i].token_estimate > chunks[j].token_estimate ? chunks[i].token_estimate : chunks[j].token_estimate;
            double balance = max_tokens > 0 ? (double)min_tokens / (double)max_tokens : 1.0;
            double score = 0.55 * jaccard + 0.20 * (heading_shared ? 1.0 : 0.0) +
                           0.15 * proximity + 0.10 * balance;
            if (score < 0.34) continue;
            double weight = clamp01(0.20 + score);
            char reason[256];
            snprintf(reason, sizeof(reason), "shared_keywords=%d jaccard=%.2f heading=%s gap=%d",
                     shared, jaccard, heading_shared ? "yes" : "no", ordinal_gap);
            insert_link(db, chunks[i].id, chunks[j].id, KK_LINK_KIND_RELATED, weight, reason);
            insert_link(db, chunks[j].id, chunks[i].id, KK_LINK_KIND_RELATED, weight, reason);
        }
    }

    free_link_chunks(chunks, count);
}

/* ═══════════════════════════════════════════════════════════════════
 * INGEST (file)
 * ═══════════════════════════════════════════════════════════════════ */

static int ingest_file(sqlite3 *db, const char *path, const char *namespace_name, const char *scope) {
    char sha[65];
    size_t len = 0;
    off_t size = 0;
    char *raw = read_file_all(path, &len, sha, &size);
    if (!raw) return 0;
    (void)len;
    char *content = normalize_text(raw);
    free(raw);

    char *abs_path = to_absolute_path(path);
    const char *filename = strrchr(abs_path, '/');
    filename = filename ? filename + 1 : abs_path;
    const char *source_type = detect_source_type(abs_path);

    begin_tx(db);
    int namespace_id = get_namespace_id(db, namespace_name, scope);
    if (namespace_id < 0) { commit_tx(db); free(abs_path); free(content); return -1; }
    int doc_id = insert_document(db, namespace_id, abs_path, filename, source_type, size);
    if (doc_id < 0) { commit_tx(db); free(abs_path); free(content); return -1; }

    int latest_num = 0, latest_version_id = 0;
    char latest_sha[65] = {0};
    int has_latest = get_latest_version(db, doc_id, &latest_num, latest_sha, &latest_version_id);
    if (has_latest && !strcmp(latest_sha, sha)) {
        touch_version_seen(db, latest_version_id);
        commit_tx(db);
        printf("skip unchanged: %s [%s]\n", abs_path, sha);
        free(abs_path);
        free(content);
        return 0;
    }

    int existing_version_id = 0, existing_version_num = 0, existing_is_latest = 0;
    if (get_version_by_sha(db, doc_id, sha, &existing_version_id, &existing_version_num, &existing_is_latest)) {
        touch_version_seen(db, existing_version_id);
        if (!existing_is_latest) set_latest_version(db, doc_id, existing_version_id);
        commit_tx(db);
        printf("reactivated lineage: %s => doc=%d version=%d sha=%s\n", abs_path, doc_id, existing_version_num, sha);
        free(abs_path);
        free(content);
        return 1;
    }

    int char_delta = 0, token_delta = 0;
    double change_ratio = 0.0;
    char diff_summary[256] = "initial ingest";
    char *prev_content = NULL;
    if (has_latest && latest_version_id > 0) {
        prev_content = get_version_content(db, latest_version_id);
        compute_diff_metrics(prev_content ? prev_content : "", content, &char_delta, &token_delta, &change_ratio, diff_summary, sizeof(diff_summary));
    }

    int version_num = latest_num + 1;
    int version_id = insert_version(db, doc_id, version_num, sha, content, has_latest ? latest_version_id : 0,
                                    char_delta, token_delta, change_ratio, diff_summary);
    free(prev_content);
    if (version_id < 0) { commit_tx(db); free(abs_path); free(content); return -1; }

    int section_count = 0;
    SectionInfo *sections = split_sections(content, &section_count);
    SectionRuntime *runtime = xmalloc((size_t)section_count * sizeof(*runtime));
    memset(runtime, 0, (size_t)section_count * sizeof(*runtime));

    int chunk_ordinal = 0;
    int prev_chunk_id = 0;
    for (int i = 0; i < section_count; i++) {
        char *heading = section_heading_from_text(sections[i].text);
        runtime[i].level = sections[i].level;
        runtime[i].ordinal = i;
        runtime[i].heading = xstrdup(heading);
        int section_id = insert_section_row(db, version_id, doc_id, i, heading, sections[i].level,
                                           (int)sections[i].start, (int)sections[i].end, sections[i].text);
        runtime[i].id = section_id;
        int chunk_count = 0;
        ChunkInfo *chunks = split_chunks(sections[i].text, sections[i].start, &chunk_count);
        for (int j = 0; j < chunk_count; j++) {
            int chunk_id = insert_chunk_row(db, version_id, doc_id, section_id, chunk_ordinal++,
                                            (int)chunks[j].start, (int)chunks[j].end,
                                            estimate_tokens(chunks[j].text), chunks[j].text);
            insert_fts_row(db, chunk_id, chunks[j].text, namespace_name, scope, abs_path, filename, heading);
            /* build PostGPT-style metaweights for RRPRAM resonance */
            { kk_chunk_meta cmeta; build_meta_from_text(chunks[j].text, (int)strlen(chunks[j].text), &cmeta); store_chunk_meta(db, chunk_id, &cmeta); }
            if (runtime[i].first_chunk_id == 0) runtime[i].first_chunk_id = chunk_id;
            runtime[i].last_chunk_id = chunk_id;
            if (prev_chunk_id > 0) {
                insert_link(db, prev_chunk_id, chunk_id, KK_LINK_KIND_STRUCTURAL, 1.0, "adjacent chunk");
                insert_link(db, chunk_id, prev_chunk_id, KK_LINK_KIND_STRUCTURAL, 1.0, "adjacent chunk");
            }
            prev_chunk_id = chunk_id;
            free(chunks[j].text);
        }
        free(chunks);
        free(heading);
        free(sections[i].text);
    }
    free(sections);

    infer_section_topology_links(db, runtime, section_count);
    for (int i = 0; i < section_count; i++) free(runtime[i].heading);
    free(runtime);
    infer_related_links(db, version_id);
    commit_tx(db);
    printf("ingested: %s => doc=%d version=%d sha=%s diff={%s}\n", abs_path, doc_id, version_num, sha, diff_summary);
    free(abs_path);
    free(content);
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════
 * INGEST (buffer — not from file)
 * ═══════════════════════════════════════════════════════════════════ */

static int ingest_buffer_internal(sqlite3 *db, const char *path, const char *text, size_t text_len,
                                  const char *namespace_name, const char *scope) {
    char sha[65];
    sha256_buffer(text, text_len, sha);

    char *content = normalize_text(text);
    const char *filename = strrchr(path, '/');
    filename = filename ? filename + 1 : path;
    const char *source_type = detect_source_type(path);

    begin_tx(db);
    int namespace_id = get_namespace_id(db, namespace_name, scope);
    if (namespace_id < 0) { commit_tx(db); free(content); return -1; }
    int doc_id = insert_document(db, namespace_id, path, filename, source_type, (sqlite3_int64)text_len);
    if (doc_id < 0) { commit_tx(db); free(content); return -1; }

    int latest_num = 0, latest_version_id = 0;
    char latest_sha[65] = {0};
    int has_latest = get_latest_version(db, doc_id, &latest_num, latest_sha, &latest_version_id);
    if (has_latest && !strcmp(latest_sha, sha)) {
        touch_version_seen(db, latest_version_id);
        commit_tx(db);
        free(content);
        return 0;
    }

    int existing_version_id = 0, existing_version_num = 0, existing_is_latest = 0;
    if (get_version_by_sha(db, doc_id, sha, &existing_version_id, &existing_version_num, &existing_is_latest)) {
        touch_version_seen(db, existing_version_id);
        if (!existing_is_latest) set_latest_version(db, doc_id, existing_version_id);
        commit_tx(db);
        free(content);
        return 1;
    }

    int char_delta = 0, token_delta = 0;
    double change_ratio = 0.0;
    char diff_summary[256] = "initial ingest";
    char *prev_content = NULL;
    if (has_latest && latest_version_id > 0) {
        prev_content = get_version_content(db, latest_version_id);
        compute_diff_metrics(prev_content ? prev_content : "", content, &char_delta, &token_delta, &change_ratio, diff_summary, sizeof(diff_summary));
    }

    int version_num = latest_num + 1;
    int version_id = insert_version(db, doc_id, version_num, sha, content, has_latest ? latest_version_id : 0,
                                    char_delta, token_delta, change_ratio, diff_summary);
    free(prev_content);
    if (version_id < 0) { commit_tx(db); free(content); return -1; }

    int section_count = 0;
    SectionInfo *sections = split_sections(content, &section_count);
    SectionRuntime *runtime = xmalloc((size_t)section_count * sizeof(*runtime));
    memset(runtime, 0, (size_t)section_count * sizeof(*runtime));

    int chunk_ordinal = 0;
    int prev_chunk_id = 0;
    for (int i = 0; i < section_count; i++) {
        char *heading = section_heading_from_text(sections[i].text);
        runtime[i].level = sections[i].level;
        runtime[i].ordinal = i;
        runtime[i].heading = xstrdup(heading);
        int section_id = insert_section_row(db, version_id, doc_id, i, heading, sections[i].level,
                                           (int)sections[i].start, (int)sections[i].end, sections[i].text);
        runtime[i].id = section_id;
        int chunk_count = 0;
        ChunkInfo *chunks = split_chunks(sections[i].text, sections[i].start, &chunk_count);
        for (int j = 0; j < chunk_count; j++) {
            int chunk_id = insert_chunk_row(db, version_id, doc_id, section_id, chunk_ordinal++,
                                            (int)chunks[j].start, (int)chunks[j].end,
                                            estimate_tokens(chunks[j].text), chunks[j].text);
            insert_fts_row(db, chunk_id, chunks[j].text, namespace_name, scope, path, filename, heading);
            { kk_chunk_meta cmeta; build_meta_from_text(chunks[j].text, (int)strlen(chunks[j].text), &cmeta); store_chunk_meta(db, chunk_id, &cmeta); }
            if (runtime[i].first_chunk_id == 0) runtime[i].first_chunk_id = chunk_id;
            runtime[i].last_chunk_id = chunk_id;
            if (prev_chunk_id > 0) {
                insert_link(db, prev_chunk_id, chunk_id, KK_LINK_KIND_STRUCTURAL, 1.0, "adjacent chunk");
                insert_link(db, chunk_id, prev_chunk_id, KK_LINK_KIND_STRUCTURAL, 1.0, "adjacent chunk");
            }
            prev_chunk_id = chunk_id;
            free(chunks[j].text);
        }
        free(chunks);
        free(heading);
        free(sections[i].text);
    }
    free(sections);

    infer_section_topology_links(db, runtime, section_count);
    for (int i = 0; i < section_count; i++) free(runtime[i].heading);
    free(runtime);
    infer_related_links(db, version_id);
    commit_tx(db);
    free(content);
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════
 * SCAN DIR
 * ═══════════════════════════════════════════════════════════════════ */

static void scan_dir(sqlite3 *db, const char *root, const char *namespace_name, const char *scope, int *count) {
    DIR *dir = opendir(root);
    if (!dir) {
        fprintf(stderr, "warn: cannot open directory %s: %s\n", root, strerror(errno));
        return;
    }
    struct dirent *ent;
    while ((ent = readdir(dir))) {
        if (!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, "..")) continue;
        char path[PATH_MAX];
        snprintf(path, sizeof(path), "%s/%s", root, ent->d_name);
        struct stat st;
        if (lstat(path, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            scan_dir(db, path, namespace_name, scope, count);
        } else if (S_ISREG(st.st_mode) && is_supported_ext(path)) {
            int rc = ingest_file(db, path, namespace_name, scope);
            if (rc > 0) *count += rc;
        }
    }
    closedir(dir);
}

/* ═══════════════════════════════════════════════════════════════════
 * QUERY TERM / FTS MATCH
 * ═══════════════════════════════════════════════════════════════════ */

static int build_query_terms(const char *query_text, char terms[KK_MAX_QUERY_TOKENS][64]) {
    int count = 0;
    char *copy = xstrdup(query_text);
    char *save = NULL;
    for (char *tok = strtok_r(copy, " \t\n,.;:!?()[]{}<>\"'", &save);
         tok && count < KK_MAX_QUERY_TOKENS;
         tok = strtok_r(NULL, " \t\n,.;:!?()[]{}<>\"'", &save)) {
        char lower[64];
        size_t n = strlen(tok);
        if (n >= sizeof(lower)) n = sizeof(lower) - 1;
        for (size_t i = 0; i < n; i++) lower[i] = (char)tolower((unsigned char)tok[i]);
        lower[n] = '\0';
        if (n >= 2) {
            strncpy(terms[count], lower, sizeof(terms[count]));
            terms[count][sizeof(terms[count]) - 1] = '\0';
            count++;
        }
    }
    free(copy);
    return count;
}

static void log_retrieval(sqlite3 *db, const char *query_text, const char *access_scope, const char *namespace_filter, const char *mode, int top_k) {
    sqlite3_stmt *stmt = NULL;
    const char *sql = "INSERT INTO retrieval_log(query_text, access_scope, namespace_filter, mode, top_k) VALUES(?, ?, ?, ?, ?);";
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(db, "prepare retrieval log"); return; }
    sqlite3_bind_text(stmt, 1, query_text, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, access_scope, -1, SQLITE_STATIC);
    if (namespace_filter) sqlite3_bind_text(stmt, 3, namespace_filter, -1, SQLITE_STATIC); else sqlite3_bind_null(stmt, 3);
    sqlite3_bind_text(stmt, 4, mode, -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 5, top_k);
    if (sqlite3_step(stmt) != SQLITE_DONE) die_sqlite(db, "insert retrieval log");
    sqlite3_finalize(stmt);
}

static char *build_fts_match(const char *query_text) {
    char terms[KK_MAX_QUERY_TOKENS][64];
    int count = build_query_terms(query_text, terms);
    if (count == 0) return xstrdup("kernel");
    StrBuf sb;
    sb_init(&sb);
    for (int i = 0; i < count; i++) {
        if (i) sb_append_n(&sb, " OR ", 4);
        sb_append_n(&sb, terms[i], strlen(terms[i]));
    }
    return sb.data;
}

/* ═══════════════════════════════════════════════════════════════════
 * SCORING
 * ═══════════════════════════════════════════════════════════════════ */

static ScorePolicy default_score_policy(void) {
    ScorePolicy p = {0.36, 0.12, 0.10, 0.16, 0.10, 0.08, 0.08};
    return p;
}

static void normalize_score_policy(ScorePolicy *p) {
    double total = p->lexical + p->recency + p->trust + p->linkage + p->scope + p->namespace_match + p->freshness;
    if (total <= 0.0) {
        *p = default_score_policy();
        total = p->lexical + p->recency + p->trust + p->linkage + p->scope + p->namespace_match + p->freshness;
    }
    p->lexical /= total;
    p->recency /= total;
    p->trust /= total;
    p->linkage /= total;
    p->scope /= total;
    p->namespace_match /= total;
    p->freshness /= total;
}

static void maybe_assign_weight(const char *key, double value, ScorePolicy *p) {
    if (!strcmp(key, "lexical")) p->lexical = value;
    else if (!strcmp(key, "recency")) p->recency = value;
    else if (!strcmp(key, "trust")) p->trust = value;
    else if (!strcmp(key, "linkage")) p->linkage = value;
    else if (!strcmp(key, "scope")) p->scope = value;
    else if (!strcmp(key, "namespace") || !strcmp(key, "namespace_match")) p->namespace_match = value;
    else if (!strcmp(key, "freshness")) p->freshness = value;
}

static ScorePolicy load_score_policy(void) {
    ScorePolicy p = default_score_policy();
    const char *env = getenv("KK_SCORE_POLICY");
    if (!env || !*env) return p;
    char *copy = xstrdup(env);
    char *save = NULL;
    for (char *part = strtok_r(copy, ",;", &save); part; part = strtok_r(NULL, ",;", &save)) {
        char *eq = strchr(part, '=');
        if (!eq) continue;
        *eq = '\0';
        char *key = trim_inplace(part);
        char *val = trim_inplace(eq + 1);
        double num = atof(val);
        maybe_assign_weight(key, num, &p);
    }
    free(copy);
    normalize_score_policy(&p);
    return p;
}

static void format_score_policy(const ScorePolicy *p, char *buf, size_t buf_sz) {
    snprintf(buf, buf_sz,
             "lexical=%.2f recency=%.2f trust=%.2f linkage=%.2f scope=%.2f namespace=%.2f freshness=%.2f",
             p->lexical, p->recency, p->trust, p->linkage, p->scope, p->namespace_match, p->freshness);
}

static void explain_score(const QueryResult *r, const char *query_text, char *buf, size_t buf_sz) {
    char terms[KK_MAX_QUERY_TOKENS][64];
    int term_count = build_query_terms(query_text, terms);
    int overlap = 0;
    for (int i = 0; i < term_count; i++) if (contains_word_ci(r->raw_text, terms[i])) overlap++;
    snprintf(buf, buf_sz,
             "overlap=%d/%d lexical=%.2f*%.2f=%.3f recency=%.2f->%.3f trust=%.2f->%.3f linkage=%.2f->%.3f scope=%.2f->%.3f namespace=%.2f->%.3f freshness=%.2f->%.3f change=%.2f",
             overlap, term_count,
             r->lexical_norm, r->weighted_lexical / (r->lexical_norm > 0.0 ? r->lexical_norm : 1.0), r->weighted_lexical,
             r->recency, r->weighted_recency,
             r->trust, r->weighted_trust,
             r->linkage, r->weighted_linkage,
             r->scope_score, r->weighted_scope,
             r->namespace_score, r->weighted_namespace,
             r->freshness, r->weighted_freshness,
             r->change_ratio);
}

/* ═══════════════════════════════════════════════════════════════════
 * LOAD QUERY RESULTS (with Hebbian boost)
 * ═══════════════════════════════════════════════════════════════════ */

static int load_query_results(sqlite3_stmt *stmt, const char *access_scope, const char *namespace_filter, int top_k, QueryResult **out_results,
                              const ScorePolicy *policy, const kk_hebbian_bridge *bridge) {
    QueryResult *results = xmalloc((size_t)(top_k * 6 + 4) * sizeof(*results));
    int count = 0;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        QueryResult r;
        memset(&r, 0, sizeof(r));
        r.chunk_id = sqlite3_column_int(stmt, 0);
        r.doc_id = sqlite3_column_int(stmt, 1);
        r.version_id = sqlite3_column_int(stmt, 2);
        r.version_num = sqlite3_column_int(stmt, 3);
        r.ordinal = sqlite3_column_int(stmt, 4);
        r.structural_links = sqlite3_column_int(stmt, 5);
        r.related_links = sqlite3_column_int(stmt, 6);
        r.token_estimate = sqlite3_column_int(stmt, 7);
        r.lexical = sqlite3_column_double(stmt, 8);
        r.path = xstrdup((const char *)sqlite3_column_text(stmt, 9));
        r.filename = xstrdup((const char *)sqlite3_column_text(stmt, 10));
        r.namespace_name = xstrdup((const char *)sqlite3_column_text(stmt, 11));
        r.scope_name = xstrdup((const char *)sqlite3_column_text(stmt, 12));
        r.raw_text = xstrdup((const char *)sqlite3_column_text(stmt, 13));
        r.sha256 = xstrdup((const char *)sqlite3_column_text(stmt, 14));
        r.ingest_ts = xstrdup((const char *)sqlite3_column_text(stmt, 15));
        r.first_seen_ts = xstrdup((const char *)sqlite3_column_text(stmt, 16));
        r.last_seen_ts = xstrdup((const char *)sqlite3_column_text(stmt, 17));
        r.section_title = xstrdup((const char *)sqlite3_column_text(stmt, 18));
        r.trust = sqlite3_column_double(stmt, 19);
        r.freshness = sqlite3_column_double(stmt, 20);
        double age_days = sqlite3_column_double(stmt, 21);
        r.char_delta = sqlite3_column_int(stmt, 22);
        r.token_delta = sqlite3_column_int(stmt, 23);
        r.change_ratio = sqlite3_column_double(stmt, 24);
        r.diff_summary = xstrdup((const char *)sqlite3_column_text(stmt, 25));
        r.seen_count = sqlite3_column_int(stmt, 26);

        if (!access_scope_allows(r.scope_name, access_scope)) {
            free((char *)r.path); free((char *)r.filename); free((char *)r.namespace_name); free((char *)r.scope_name);
            free((char *)r.raw_text); free((char *)r.sha256); free((char *)r.ingest_ts); free((char *)r.first_seen_ts); free((char *)r.last_seen_ts);
            free((char *)r.section_title); free((char *)r.diff_summary);
            continue;
        }
        r.scope_score = scope_compatibility(r.scope_name, access_scope);
        r.namespace_score = namespace_filter ? (!strcmp(r.namespace_name, namespace_filter) ? 1.0 : 0.0) : 0.88;
        r.recency = 1.0 / (1.0 + fmax(0.0, age_days) / 10.0);
        double structure_density = (double)r.structural_links / (double)(r.token_estimate + 3);
        double related_density = (double)r.related_links / (double)(r.token_estimate + 3);
        r.linkage = clamp01(structure_density * 9.0 + related_density * 14.0);
        r.lexical_norm = clamp01(r.lexical / 8.0);
        r.weighted_lexical = policy->lexical * r.lexical_norm;
        r.weighted_recency = policy->recency * r.recency;
        r.weighted_trust = policy->trust * r.trust;
        r.weighted_linkage = policy->linkage * r.linkage;
        r.weighted_scope = policy->scope * r.scope_score;
        r.weighted_namespace = policy->namespace_match * r.namespace_score;
        r.weighted_freshness = policy->freshness * r.freshness;
        r.resonance = clamp01(r.weighted_lexical + r.weighted_recency + r.weighted_trust + r.weighted_linkage +
                              r.weighted_scope + r.weighted_namespace + r.weighted_freshness);

        /* Hebbian boost from dario's co-occurrence field */
        r.hebbian_boost = 0.0;
        if (bridge && bridge->word_resonance) {
            char kw[KK_MAX_KEYWORDS][KK_KEYWORD_LEN];
            int kw_count = collect_keywords(r.raw_text, kw);
            if (kw_count > 0) {
                double total_resonance = 0.0;
                for (int ki = 0; ki < kw_count; ki++) {
                    total_resonance += (double)bridge->word_resonance(kw[ki], bridge->user_data);
                }
                r.hebbian_boost = (total_resonance / (double)kw_count) * 0.15;
                r.resonance += r.hebbian_boost;
            }
        }
        r.resonance = clamp01(r.resonance);

        results[count++] = r;
        if (count >= top_k * 6 + 4) break;
    }

    for (int i = 0; i < count; i++) {
        for (int j = i + 1; j < count; j++) {
            if (results[j].resonance > results[i].resonance) {
                QueryResult tmp = results[i];
                results[i] = results[j];
                results[j] = tmp;
            }
        }
    }
    if (count > top_k) count = top_k;
    *out_results = results;
    return count;
}

/* ═══════════════════════════════════════════════════════════════════
 * FETCH RESULTS
 * ═══════════════════════════════════════════════════════════════════ */

static int fetch_results(kk_ctx *ctx, const char *query_text, const char *access_scope, const char *namespace_filter, int top_k, QueryResult **out_results, ScorePolicy *policy_out) {
    const char *sql =
        "SELECT c.id, c.parent_document_id, c.version_id, dv.version_num, c.ordinal, "
        "COALESCE((SELECT COUNT(*) FROM links l WHERE l.from_chunk_id=c.id AND l.kind=1),0) AS structural_links,"
        "COALESCE((SELECT COUNT(*) FROM links l WHERE l.from_chunk_id=c.id AND l.kind=2),0) AS related_links,"
        "c.token_estimate, -bm25(chunk_fts) AS lexical, "
        "d.path, d.filename, ns.name, ns.scope, c.raw_text, dv.sha256, dv.ingest_ts, dv.first_seen_ts, dv.last_seen_ts, s.heading, dv.trust,"
        "CASE WHEN dv.is_latest=1 THEN 1.0 ELSE 0.35 END AS freshness,"
        "julianday('now') - julianday(COALESCE(dv.last_seen_ts, dv.ingest_ts)) AS age_days,"
        "dv.char_delta, dv.token_delta, dv.change_ratio, dv.diff_summary, dv.seen_count "
        "FROM chunk_fts "
        "JOIN chunks c ON c.id = chunk_fts.chunk_id "
        "JOIN document_versions dv ON dv.id = c.version_id "
        "JOIN documents d ON d.id = c.parent_document_id "
        "JOIN namespaces ns ON ns.id = d.namespace_id "
        "JOIN sections s ON s.id = c.parent_section_id "
        "WHERE chunk_fts MATCH ? "
        "AND (ns.scope='public' OR ns.scope=?) "
        "AND (? IS NULL OR ns.name = ?) "
        "ORDER BY lexical DESC LIMIT ?;";
    sqlite3_stmt *stmt = NULL;
    char *fts_query = build_fts_match(query_text);
    ScorePolicy policy = load_score_policy();
    if (policy_out) *policy_out = policy;
    if (sqlite3_prepare_v2(ctx->db, sql, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(ctx->db, "prepare query"); free(fts_query); return 0; }
    sqlite3_bind_text(stmt, 1, fts_query, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, access_scope, -1, SQLITE_STATIC);
    if (namespace_filter) {
        sqlite3_bind_text(stmt, 3, namespace_filter, -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 4, namespace_filter, -1, SQLITE_STATIC);
    } else {
        sqlite3_bind_null(stmt, 3);
        sqlite3_bind_null(stmt, 4);
    }
    sqlite3_bind_int(stmt, 5, top_k * 6);
    int count = load_query_results(stmt, access_scope, namespace_filter, top_k, out_results, &policy, ctx->bridge);
    sqlite3_finalize(stmt);
    free(fts_query);
    return count;
}

static int fetch_results_exact_scope(kk_ctx *ctx, const char *query_text, const char *scope_name, const char *namespace_filter, int top_k, QueryResult **out_results, ScorePolicy *policy_out) {
    const char *sql =
        "SELECT c.id, c.parent_document_id, c.version_id, dv.version_num, c.ordinal, "
        "COALESCE((SELECT COUNT(*) FROM links l WHERE l.from_chunk_id=c.id AND l.kind=1),0) AS structural_links,"
        "COALESCE((SELECT COUNT(*) FROM links l WHERE l.from_chunk_id=c.id AND l.kind=2),0) AS related_links,"
        "c.token_estimate, -bm25(chunk_fts) AS lexical, "
        "d.path, d.filename, ns.name, ns.scope, c.raw_text, dv.sha256, dv.ingest_ts, dv.first_seen_ts, dv.last_seen_ts, s.heading, dv.trust,"
        "CASE WHEN dv.is_latest=1 THEN 1.0 ELSE 0.35 END AS freshness,"
        "julianday('now') - julianday(COALESCE(dv.last_seen_ts, dv.ingest_ts)) AS age_days,"
        "dv.char_delta, dv.token_delta, dv.change_ratio, dv.diff_summary, dv.seen_count "
        "FROM chunk_fts "
        "JOIN chunks c ON c.id = chunk_fts.chunk_id "
        "JOIN document_versions dv ON dv.id = c.version_id "
        "JOIN documents d ON d.id = c.parent_document_id "
        "JOIN namespaces ns ON ns.id = d.namespace_id "
        "JOIN sections s ON s.id = c.parent_section_id "
        "WHERE chunk_fts MATCH ? "
        "AND ns.scope=? "
        "AND (? IS NULL OR ns.name = ?) "
        "ORDER BY lexical DESC LIMIT ?;";
    sqlite3_stmt *stmt = NULL;
    char *fts_query = build_fts_match(query_text);
    ScorePolicy policy = load_score_policy();
    if (policy_out) *policy_out = policy;
    if (sqlite3_prepare_v2(ctx->db, sql, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(ctx->db, "prepare scoped query"); free(fts_query); return 0; }
    sqlite3_bind_text(stmt, 1, fts_query, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, scope_name, -1, SQLITE_STATIC);
    if (namespace_filter) {
        sqlite3_bind_text(stmt, 3, namespace_filter, -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 4, namespace_filter, -1, SQLITE_STATIC);
    } else {
        sqlite3_bind_null(stmt, 3);
        sqlite3_bind_null(stmt, 4);
    }
    sqlite3_bind_int(stmt, 5, top_k * 6);
    int count = load_query_results(stmt, scope_name, namespace_filter, top_k, out_results, &policy, ctx->bridge);
    sqlite3_finalize(stmt);
    free(fts_query);
    return count;
}

static void free_query_results(QueryResult *results, int count) {
    if (!results) return;
    for (int i = 0; i < count; i++) {
        free((char *)results[i].path);
        free((char *)results[i].filename);
        free((char *)results[i].namespace_name);
        free((char *)results[i].scope_name);
        free((char *)results[i].raw_text);
        free((char *)results[i].sha256);
        free((char *)results[i].ingest_ts);
        free((char *)results[i].first_seen_ts);
        free((char *)results[i].last_seen_ts);
        free((char *)results[i].section_title);
        free((char *)results[i].diff_summary);
    }
    free(results);
}

/* ═══════════════════════════════════════════════════════════════════
 * RENDERING (query term overlap, score policy JSON, result JSON)
 * ═══════════════════════════════════════════════════════════════════ */

static int query_term_overlap(const QueryResult *r, const char *query_text, int *term_count_out) {
    char terms[KK_MAX_QUERY_TOKENS][64];
    int term_count = build_query_terms(query_text, terms);
    int overlap = 0;
    for (int i = 0; i < term_count; i++) if (contains_word_ci(r->raw_text, terms[i])) overlap++;
    if (term_count_out) *term_count_out = term_count;
    return overlap;
}

static void append_score_policy_json(StrBuf *sb, const ScorePolicy *policy) {
    sb_append(sb, "\"score_policy\":{");
    sb_appendf(sb, "\"lexical\":%.6f,", policy->lexical);
    sb_appendf(sb, "\"recency\":%.6f,", policy->recency);
    sb_appendf(sb, "\"trust\":%.6f,", policy->trust);
    sb_appendf(sb, "\"linkage\":%.6f,", policy->linkage);
    sb_appendf(sb, "\"scope\":%.6f,", policy->scope);
    sb_appendf(sb, "\"namespace\":%.6f,", policy->namespace_match);
    sb_appendf(sb, "\"freshness\":%.6f", policy->freshness);
    sb_append(sb, "}");
}

static void append_result_json(StrBuf *sb, const QueryResult *r, const char *query_text) {
    int term_count = 0;
    int overlap = query_term_overlap(r, query_text, &term_count);
    char *excerpt = make_excerpt(r->raw_text, 220);
    char *locator = make_locator(r);
    char *lineage = make_lineage_summary(r);
    sb_append(sb, "{");
    sb_append(sb, "\"document_path\":"); json_append_string(sb, r->path); sb_append(sb, ",");
    sb_append(sb, "\"title\":"); json_append_string(sb, (r->section_title && *r->section_title) ? r->section_title : r->filename); sb_append(sb, ",");
    sb_append(sb, "\"version\":{");
    sb_appendf(sb, "\"version_num\":%d,", r->version_num);
    sb_appendf(sb, "\"version_id\":%d,", r->version_id);
    sb_append(sb, "\"sha256\":"); json_append_string(sb, r->sha256); sb_append(sb, ",");
    sb_appendf(sb, "\"is_latest\":%s", r->freshness >= 0.999 ? "true" : "false");
    sb_append(sb, "},");
    sb_appendf(sb, "\"chunk_id\":%d,", r->chunk_id);
    sb_append(sb, "\"anchor\":"); json_append_string(sb, r->section_title ? r->section_title : "root"); sb_append(sb, ",");
    sb_append(sb, "\"text\":"); json_append_string(sb, excerpt); sb_append(sb, ",");
    sb_append(sb, "\"locator\":"); json_append_string(sb, locator); sb_append(sb, ",");
    sb_appendf(sb, "\"score\":%.6f,", r->resonance);
    sb_appendf(sb, "\"hebbian_boost\":%.6f,", r->hebbian_boost);
    sb_append(sb, "\"score_breakdown\":{");
    sb_appendf(sb, "\"query_term_overlap\":%d,", overlap);
    sb_appendf(sb, "\"query_term_count\":%d,", term_count);
    sb_appendf(sb, "\"lexical_raw\":%.6f,", r->lexical);
    sb_appendf(sb, "\"lexical_norm\":%.6f,", r->lexical_norm);
    sb_appendf(sb, "\"recency\":%.6f,", r->recency);
    sb_appendf(sb, "\"trust\":%.6f,", r->trust);
    sb_appendf(sb, "\"linkage\":%.6f,", r->linkage);
    sb_appendf(sb, "\"scope\":%.6f,", r->scope_score);
    sb_appendf(sb, "\"namespace\":%.6f,", r->namespace_score);
    sb_appendf(sb, "\"freshness\":%.6f,", r->freshness);
    sb_appendf(sb, "\"weighted_lexical\":%.6f,", r->weighted_lexical);
    sb_appendf(sb, "\"weighted_recency\":%.6f,", r->weighted_recency);
    sb_appendf(sb, "\"weighted_trust\":%.6f,", r->weighted_trust);
    sb_appendf(sb, "\"weighted_linkage\":%.6f,", r->weighted_linkage);
    sb_appendf(sb, "\"weighted_scope\":%.6f,", r->weighted_scope);
    sb_appendf(sb, "\"weighted_namespace\":%.6f,", r->weighted_namespace);
    sb_appendf(sb, "\"weighted_freshness\":%.6f", r->weighted_freshness);
    sb_append(sb, "},");
    sb_append(sb, "\"lineage_summary\":"); json_append_string(sb, lineage); sb_append(sb, ",");
    sb_append(sb, "\"trust_provenance\":{");
    sb_appendf(sb, "\"trust\":%.6f,", r->trust);
    sb_append(sb, "\"namespace\":"); json_append_string(sb, r->namespace_name); sb_append(sb, ",");
    sb_append(sb, "\"scope\":"); json_append_string(sb, r->scope_name); sb_append(sb, ",");
    sb_append(sb, "\"ingest_ts\":"); json_append_string(sb, r->ingest_ts); sb_append(sb, ",");
    sb_append(sb, "\"first_seen_ts\":"); json_append_string(sb, r->first_seen_ts); sb_append(sb, ",");
    sb_append(sb, "\"last_seen_ts\":"); json_append_string(sb, r->last_seen_ts); sb_append(sb, ",");
    sb_appendf(sb, "\"seen_count\":%d", r->seen_count);
    sb_append(sb, "}");
    sb_append(sb, "}");
    free(excerpt);
    free(locator);
    free(lineage);
}

/* ═══════════════════════════════════════════════════════════════════
 * UPSERT MODEL REGISTRY
 * ═══════════════════════════════════════════════════════════════════ */

static void upsert_model_registry(sqlite3 *db, const char *model_name, const char *scope_default, const char *namespace_default,
                                  const char *retrieval_mode_default, const char *packet_mode_default,
                                  const char *query_profile_default, int active) {
    sqlite3_stmt *stmt = NULL;
    const char *sql =
        "INSERT INTO model_registry(model_name, scope_default, namespace_default, retrieval_mode_default, packet_mode_default, query_profile_default, notes, is_active, detached_ts, created_ts, updated_ts) "
        "VALUES(?, ?, ?, ?, ?, ?, '', ?, NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
        "ON CONFLICT(model_name) DO UPDATE SET "
        "scope_default=excluded.scope_default, "
        "namespace_default=excluded.namespace_default, "
        "retrieval_mode_default=excluded.retrieval_mode_default, "
        "packet_mode_default=excluded.packet_mode_default, "
        "query_profile_default=excluded.query_profile_default, "
        "is_active=excluded.is_active, "
        "detached_ts=CASE WHEN excluded.is_active=1 THEN NULL ELSE CURRENT_TIMESTAMP END, "
        "updated_ts=CURRENT_TIMESTAMP;";
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(db, "prepare model upsert"); return; }
    sqlite3_bind_text(stmt, 1, model_name, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, scope_default, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, namespace_default, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 4, retrieval_mode_default, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 5, packet_mode_default, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 6, query_profile_default, -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 7, active);
    if (sqlite3_step(stmt) != SQLITE_DONE) die_sqlite(db, "model upsert");
    sqlite3_finalize(stmt);
}

/* ═══════════════════════════════════════════════════════════════════
 * INTEGRITY CHECK
 * ═══════════════════════════════════════════════════════════════════ */

static int integrity_count_query(sqlite3 *db, const char *sql) {
    sqlite3_stmt *stmt = NULL;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(db, "prepare integrity count query"); return 0; }
    int count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) count = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
    return count;
}

/* ═══════════════════════════════════════════════════════════════════
 * ADJACENT CHUNK FETCH
 * ═══════════════════════════════════════════════════════════════════ */

static char *fetch_adjacent_chunk_text(sqlite3 *db, int version_id, int ordinal) {
    sqlite3_stmt *stmt = NULL;
    const char *sql = "SELECT raw_text FROM chunks WHERE version_id=? AND ordinal=? LIMIT 1;";
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(db, "prepare adjacent chunk"); return NULL; }
    sqlite3_bind_int(stmt, 1, version_id);
    sqlite3_bind_int(stmt, 2, ordinal);
    char *text = NULL;
    if (sqlite3_step(stmt) == SQLITE_ROW) text = xstrdup((const char *)sqlite3_column_text(stmt, 0));
    sqlite3_finalize(stmt);
    return text;
}

/* ═══════════════════════════════════════════════════════════════════
 * DOCUMENT PATH LOOKUP
 * ═══════════════════════════════════════════════════════════════════ */

static int find_document_id_by_path(sqlite3 *db, const char *document_path, char **resolved_path_out) {
    char *resolved = to_absolute_path(document_path);
    sqlite3_stmt *stmt = NULL;
    const char *sql = "SELECT id, path FROM documents WHERE path=? ORDER BY id ASC LIMIT 1;";
    int doc_id = 0;

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(db, "prepare document path lookup"); free(resolved); return 0; }
    sqlite3_bind_text(stmt, 1, resolved, -1, SQLITE_STATIC);
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        doc_id = sqlite3_column_int(stmt, 0);
        if (resolved_path_out) *resolved_path_out = xstrdup((const char *)sqlite3_column_text(stmt, 1));
    }
    sqlite3_finalize(stmt);

    if (!doc_id && strcmp(resolved, document_path) != 0) {
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(db, "prepare document path fallback lookup"); free(resolved); return 0; }
        sqlite3_bind_text(stmt, 1, document_path, -1, SQLITE_STATIC);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            doc_id = sqlite3_column_int(stmt, 0);
            if (resolved_path_out) *resolved_path_out = xstrdup((const char *)sqlite3_column_text(stmt, 1));
        }
        sqlite3_finalize(stmt);
    }

    free(resolved);
    return doc_id;
}

/* ═══════════════════════════════════════════════════════════════════
 * PUBLIC API — LIFECYCLE
 * ═══════════════════════════════════════════════════════════════════ */

kk_ctx *kk_open(const char *db_path) {
    if (!db_path) return NULL;
    sqlite3 *db = open_db(db_path);
    if (!db) return NULL;
    kk_ctx *k = xmalloc(sizeof(*k));
    memset(k, 0, sizeof(*k));
    k->db = db;
    k->ready = 1;
    return k;
}

void kk_close(kk_ctx *k) {
    if (!k) return;
    if (k->db) sqlite3_close(k->db);
    free(k);
}

int kk_is_ready(kk_ctx *k) {
    return k && k->ready && k->db;
}

/* ═══════════════════════════════════════════════════════════════════
 * PUBLIC API — EMBEDDING & HEBBIAN SLOTS
 * ═══════════════════════════════════════════════════════════════════ */

void kk_set_embedder(kk_ctx *k, const kk_embedder *embedder) {
    if (!k) return;
    k->embedder = embedder;
}

void kk_set_hebbian_bridge(kk_ctx *k, const kk_hebbian_bridge *bridge) {
    if (!k) return;
    k->bridge = bridge;
}

/* ═══════════════════════════════════════════════════════════════════
 * PUBLIC API — INGEST
 * ═══════════════════════════════════════════════════════════════════ */

int kk_ingest_file(kk_ctx *k, const char *path,
                   const char *namespace_name, const char *scope) {
    if (!kk_is_ready(k) || !path || !namespace_name || !scope) return -1;
    if (require_scope_lib(scope) != 0) return -1;
    return ingest_file(k->db, path, namespace_name, scope);
}

int kk_ingest_dir(kk_ctx *k, const char *dir_path,
                  const char *namespace_name, const char *scope) {
    if (!kk_is_ready(k) || !dir_path || !namespace_name || !scope) return -1;
    if (require_scope_lib(scope) != 0) return -1;
    int count = 0;
    scan_dir(k->db, dir_path, namespace_name, scope, &count);
    return count;
}

int kk_store(kk_ctx *k, const char *path, const char *text,
                     size_t text_len, const char *namespace_name,
                     const char *scope) {
    if (!kk_is_ready(k) || !path || !text || !namespace_name || !scope) return -1;
    if (require_scope_lib(scope) != 0) return -1;
    return ingest_buffer_internal(k->db, path, text, text_len, namespace_name, scope);
}

/* ═══════════════════════════════════════════════════════════════════
 * RRPRAM METAWEIGHTS — PostGPT-style statistical fingerprints
 *
 * At ingest, each chunk gets a positional affinity vector (the Wr column),
 * bigram patterns, and Hebbian co-occurrence pairs.
 * At query time, the organism's current embedding resonates against these.
 *
 * The model doesn't search. Knowledge resonates.
 * ═══════════════════════════════════════════════════════════════════ */

/* Simple byte-level tokenization for metaweight computation.
 * Returns token IDs in out[], returns count. Max max_tokens. */
static int meta_tokenize(const char *text, int text_len, int *out, int max_tokens) {
    int n = 0;
    for (int i = 0; i < text_len && n < max_tokens; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c > 32) { /* skip control chars and spaces */
            out[n++] = (int)c;
        }
    }
    return n;
}

/* Build positional affinity, bigrams, and Hebbian co-occurrence for a chunk */
static int build_meta_from_text(const char *text, int text_len, kk_chunk_meta *meta) {
    memset(meta, 0, sizeof(*meta));

    int tokens[4096];
    int n = meta_tokenize(text, text_len, tokens, 4096);
    meta->n_tokens = n;
    if (n < 2) return 0;

    /* 1. Positional affinity — P(position | token) aggregated into dense vector.
     * Each position in the chunk gets a weighted contribution from its token.
     * Result: affinity[pos % DIM] += 1/n_tokens. Normalized to [0,1]. */
    int dim = KK_META_AFFINITY_DIM;
    for (int i = 0; i < n; i++) {
        int pos = i % dim;
        float weight = 1.0f + 0.5f * (float)(tokens[i] % 64) / 64.0f;
        meta->affinity[pos] += weight;
    }
    /* normalize */
    float amax = 0;
    for (int i = 0; i < dim; i++) if (meta->affinity[i] > amax) amax = meta->affinity[i];
    if (amax > 1e-6f) for (int i = 0; i < dim; i++) meta->affinity[i] /= amax;

    /* 2. Bigrams — top transitions within chunk */
    typedef struct { int src, dst; float count; } BigramEntry;
    BigramEntry bg[1024];
    int bg_n = 0;
    for (int i = 0; i < n - 1 && bg_n < 1024; i++) {
        int found = 0;
        for (int j = 0; j < bg_n; j++) {
            if (bg[j].src == tokens[i] && bg[j].dst == tokens[i+1]) {
                bg[j].count += 1.0f;
                found = 1;
                break;
            }
        }
        if (!found && bg_n < 1024) {
            bg[bg_n++] = (BigramEntry){tokens[i], tokens[i+1], 1.0f};
        }
    }
    /* sort by count descending, take top KK_META_BIGRAM_MAX */
    for (int i = 0; i < bg_n; i++)
        for (int j = i+1; j < bg_n; j++)
            if (bg[j].count > bg[i].count) { BigramEntry t = bg[i]; bg[i] = bg[j]; bg[j] = t; }
    meta->bigram_n = bg_n < KK_META_BIGRAM_MAX ? bg_n : KK_META_BIGRAM_MAX;
    float bg_total = 0;
    for (int i = 0; i < meta->bigram_n; i++) bg_total += bg[i].count;
    for (int i = 0; i < meta->bigram_n; i++) {
        meta->bigram_src[i] = bg[i].src;
        meta->bigram_dst[i] = bg[i].dst;
        meta->bigram_prob[i] = bg_total > 0 ? bg[i].count / bg_total : 0;
    }

    /* 3. Hebbian co-occurrence — distance-weighted pairs within window=8 */
    typedef struct { int a, b; float strength; } HebbEntry;
    HebbEntry hb[2048];
    int hb_n = 0;
    int window = 8;
    int hebb_limit = n < 2000 ? n : 2000;
    for (int i = 0; i < hebb_limit; i++) {
        for (int j = (i - window > 0 ? i - window : 0); j < i && j < hebb_limit; j++) {
            int a = tokens[i] < tokens[j] ? tokens[i] : tokens[j];
            int b = tokens[i] < tokens[j] ? tokens[j] : tokens[i];
            float decay = 1.0f / (1.0f + (float)(i - j));
            int found = 0;
            for (int k = 0; k < hb_n; k++) {
                if (hb[k].a == a && hb[k].b == b) { hb[k].strength += decay; found = 1; break; }
            }
            if (!found && hb_n < 2048) {
                hb[hb_n++] = (HebbEntry){a, b, decay};
            }
        }
    }
    /* sort, take top */
    for (int i = 0; i < hb_n; i++)
        for (int j = i+1; j < hb_n; j++)
            if (hb[j].strength > hb[i].strength) { HebbEntry t = hb[i]; hb[i] = hb[j]; hb[j] = t; }
    meta->hebb_n = hb_n < KK_META_HEBBIAN_MAX ? hb_n : KK_META_HEBBIAN_MAX;
    float hmax = hb_n > 0 ? hb[0].strength : 1.0f;
    for (int i = 0; i < meta->hebb_n; i++) {
        meta->hebb_a[i] = hb[i].a;
        meta->hebb_b[i] = hb[i].b;
        meta->hebb_strength[i] = hb[i].strength / hmax;
    }

    return 0;
}

/* Store chunk_meta to SQLite */
static int store_chunk_meta(sqlite3 *db, int chunk_id, const kk_chunk_meta *meta) {
    const char *sql = "INSERT OR REPLACE INTO chunk_meta(chunk_id, affinity, bigram_data, hebbian_data, n_tokens) VALUES(?,?,?,?,?)";
    sqlite3_stmt *stmt;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) return -1;
    sqlite3_bind_int(stmt, 1, chunk_id);
    sqlite3_bind_blob(stmt, 2, meta->affinity, sizeof(meta->affinity), SQLITE_TRANSIENT);
    /* pack bigrams: [src0,dst0,prob0, src1,dst1,prob1, ...] as floats */
    float bg_buf[KK_META_BIGRAM_MAX * 3];
    for (int i = 0; i < meta->bigram_n; i++) {
        bg_buf[i*3+0] = (float)meta->bigram_src[i];
        bg_buf[i*3+1] = (float)meta->bigram_dst[i];
        bg_buf[i*3+2] = meta->bigram_prob[i];
    }
    sqlite3_bind_blob(stmt, 3, bg_buf, meta->bigram_n * 3 * sizeof(float), SQLITE_TRANSIENT);
    /* pack hebbian: [a0,b0,str0, ...] */
    float hb_buf[KK_META_HEBBIAN_MAX * 3];
    for (int i = 0; i < meta->hebb_n; i++) {
        hb_buf[i*3+0] = (float)meta->hebb_a[i];
        hb_buf[i*3+1] = (float)meta->hebb_b[i];
        hb_buf[i*3+2] = meta->hebb_strength[i];
    }
    sqlite3_bind_blob(stmt, 4, hb_buf, meta->hebb_n * 3 * sizeof(float), SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 5, meta->n_tokens);
    int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return rc == SQLITE_DONE ? 0 : -1;
}

/* Load chunk_meta from SQLite */
static int load_chunk_meta(sqlite3 *db, int chunk_id, kk_chunk_meta *meta) {
    memset(meta, 0, sizeof(*meta));
    const char *sql = "SELECT affinity, bigram_data, hebbian_data, n_tokens FROM chunk_meta WHERE chunk_id=?";
    sqlite3_stmt *stmt;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) return -1;
    sqlite3_bind_int(stmt, 1, chunk_id);
    if (sqlite3_step(stmt) != SQLITE_ROW) { sqlite3_finalize(stmt); return -1; }

    const void *aff = sqlite3_column_blob(stmt, 0);
    int aff_sz = sqlite3_column_bytes(stmt, 0);
    if (aff && aff_sz == (int)sizeof(meta->affinity))
        memcpy(meta->affinity, aff, sizeof(meta->affinity));

    const float *bg = (const float *)sqlite3_column_blob(stmt, 1);
    int bg_sz = sqlite3_column_bytes(stmt, 1);
    meta->bigram_n = bg_sz / (3 * sizeof(float));
    if (meta->bigram_n > KK_META_BIGRAM_MAX) meta->bigram_n = KK_META_BIGRAM_MAX;
    for (int i = 0; i < meta->bigram_n; i++) {
        meta->bigram_src[i] = (int)bg[i*3+0];
        meta->bigram_dst[i] = (int)bg[i*3+1];
        meta->bigram_prob[i] = bg[i*3+2];
    }

    const float *hb = (const float *)sqlite3_column_blob(stmt, 2);
    int hb_sz = sqlite3_column_bytes(stmt, 2);
    meta->hebb_n = hb_sz / (3 * sizeof(float));
    if (meta->hebb_n > KK_META_HEBBIAN_MAX) meta->hebb_n = KK_META_HEBBIAN_MAX;
    for (int i = 0; i < meta->hebb_n; i++) {
        meta->hebb_a[i] = (int)hb[i*3+0];
        meta->hebb_b[i] = (int)hb[i*3+1];
        meta->hebb_strength[i] = hb[i*3+2];
    }
    meta->n_tokens = sqlite3_column_int(stmt, 3);
    sqlite3_finalize(stmt);
    return 0;
}

/* ── Public API: metaweights ── */

int kk_build_chunk_meta(kk_ctx *k, int chunk_id) {
    if (!kk_is_ready(k)) return -1;
    sqlite3 *db = k->db;
    /* fetch chunk text */
    const char *sql = "SELECT raw_text FROM chunks WHERE id=?";
    sqlite3_stmt *stmt;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) return -1;
    sqlite3_bind_int(stmt, 1, chunk_id);
    if (sqlite3_step(stmt) != SQLITE_ROW) { sqlite3_finalize(stmt); return -1; }
    const char *text = (const char *)sqlite3_column_text(stmt, 0);
    int text_len = sqlite3_column_bytes(stmt, 0);

    kk_chunk_meta meta;
    build_meta_from_text(text, text_len, &meta);
    sqlite3_finalize(stmt);

    return store_chunk_meta(db, chunk_id, &meta);
}

int kk_rebuild_all_meta(kk_ctx *k) {
    if (!kk_is_ready(k)) return -1;
    sqlite3 *db = k->db;
    const char *sql = "SELECT id FROM chunks";
    sqlite3_stmt *stmt;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) return -1;
    int count = 0;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int chunk_id = sqlite3_column_int(stmt, 0);
        kk_build_chunk_meta(k, chunk_id);
        count++;
    }
    sqlite3_finalize(stmt);
    return count;
}

float kk_chunk_resonance(const kk_chunk_meta *meta,
                         const float *current_embedding, int dim) {
    if (!meta || !current_embedding || dim <= 0) return 0.0f;
    /* dot product between embedding and positional affinity (the X·Wr operation) */
    int eff_dim = dim < KK_META_AFFINITY_DIM ? dim : KK_META_AFFINITY_DIM;
    float dot = 0, norm_a = 0, norm_e = 0;
    for (int i = 0; i < eff_dim; i++) {
        dot += meta->affinity[i] * current_embedding[i];
        norm_a += meta->affinity[i] * meta->affinity[i];
        norm_e += current_embedding[i] * current_embedding[i];
    }
    float denom = sqrtf(norm_a) * sqrtf(norm_e);
    if (denom < 1e-8f) return 0.0f;
    float cosine = dot / denom;
    return cosine > 0.0f ? cosine : 0.0f; /* clamp negative */
}

int kk_retrieve_resonant(kk_ctx *k, const char *query_text,
                      const float *current_embedding, int embed_dim,
                      const char *access_scope, const char *namespace_filter,
                      int top_k, kk_profile profile,
                      kk_result **results_out) {
    /* first, run standard query */
    int count = kk_retrieve(k, query_text, access_scope, namespace_filter,
                         top_k * 2, profile, results_out);
    if (count <= 0 || !current_embedding || embed_dim <= 0) return count;

    /* augment with RRPRAM resonance */
    kk_result *results = *results_out;
    for (int i = 0; i < count; i++) {
        kk_chunk_meta meta;
        if (load_chunk_meta(k->db, results[i].chunk_id, &meta) == 0) {
            results[i].rrpram_resonance = (double)kk_chunk_resonance(&meta, current_embedding, embed_dim);
            results[i].resonance += results[i].rrpram_resonance * 0.20; /* 20% RRPRAM weight */
            if (results[i].resonance > 1.0) results[i].resonance = 1.0;
        }
    }

    /* re-sort by updated resonance */
    for (int i = 0; i < count; i++)
        for (int j = i+1; j < count; j++)
            if (results[j].resonance > results[i].resonance) {
                kk_result tmp = results[i]; results[i] = results[j]; results[j] = tmp;
            }

    /* trim to requested top_k */
    if (count > top_k) count = top_k;
    return count;
}

/* ═══════════════════════════════════════════════════════════════════
 * PUBLIC API — QUERY
 * ═══════════════════════════════════════════════════════════════════ */

int kk_retrieve(kk_ctx *k, const char *query_text,
             const char *access_scope, const char *namespace_filter,
             int top_k, kk_profile profile,
             kk_result **results_out) {
    if (!kk_is_ready(k) || !query_text || !access_scope || !results_out) return -1;
    if (require_scope_lib(access_scope) != 0) return -1;
    if (top_k <= 0) top_k = 5;

    const QueryProfile *prof = profile_from_enum(profile);
    int effective_top_k = top_k < prof->result_cap ? top_k : prof->result_cap;

    log_retrieval(k->db, query_text, access_scope, namespace_filter, "compressed", top_k);

    QueryResult *internal_results = NULL;
    ScorePolicy policy;
    int count = fetch_results(k, query_text, access_scope, namespace_filter, effective_top_k, &internal_results, &policy);

    /* Convert internal QueryResult to public kk_result */
    kk_result *out = xmalloc((size_t)(count > 0 ? count : 1) * sizeof(*out));
    for (int i = 0; i < count; i++) {
        memset(&out[i], 0, sizeof(out[i]));
        out[i].chunk_id = internal_results[i].chunk_id;
        out[i].doc_id = internal_results[i].doc_id;
        out[i].version_num = internal_results[i].version_num;
        out[i].ordinal = internal_results[i].ordinal;
        out[i].seen_count = internal_results[i].seen_count;
        out[i].path = xstrdup(internal_results[i].path ? internal_results[i].path : "");
        out[i].namespace_name = xstrdup(internal_results[i].namespace_name ? internal_results[i].namespace_name : "");
        out[i].scope_name = xstrdup(internal_results[i].scope_name ? internal_results[i].scope_name : "");
        out[i].section_title = xstrdup(internal_results[i].section_title ? internal_results[i].section_title : "");

        /* Budget text based on profile */
        BudgetedText bt = make_excerpt_budgeted(internal_results[i].raw_text, (size_t)prof->excerpt_chars);
        out[i].text = bt.text;
        out[i].text_len = (int)strlen(bt.text);

        out[i].resonance = internal_results[i].resonance;
        out[i].lexical = internal_results[i].lexical_norm;
        out[i].recency = internal_results[i].recency;
        out[i].trust = internal_results[i].trust;
        out[i].linkage = internal_results[i].linkage;
        out[i].freshness = internal_results[i].freshness;
        out[i].hebbian_boost = internal_results[i].hebbian_boost;
        out[i].sha256 = xstrdup(internal_results[i].sha256 ? internal_results[i].sha256 : "");
        out[i].ingest_ts = xstrdup(internal_results[i].ingest_ts ? internal_results[i].ingest_ts : "");
        out[i].char_delta = internal_results[i].char_delta;
        out[i].change_ratio = internal_results[i].change_ratio;
        out[i].diff_summary = xstrdup(internal_results[i].diff_summary ? internal_results[i].diff_summary : "");
    }

    free_query_results(internal_results, count);
    *results_out = out;
    return count;
}

int kk_ask(kk_ctx *k, const char *model_name, const char *query_text,
           int top_k, kk_result **results_out) {
    if (!kk_is_ready(k) || !model_name || !query_text || !results_out) return -1;
    if (top_k <= 0) top_k = 5;

    ModelAttachment model = fetch_model_attachment(k->db, model_name);
    if (!model.found) {
        free_model_attachment(&model);
        return -1;
    }

    if (!validate_scope(model.scope_default) || !validate_mode(model.retrieval_mode_default) ||
        !validate_packet_mode(model.packet_mode_default)) {
        free_model_attachment(&model);
        return -1;
    }

    const QueryProfile *prof = find_query_profile(model.query_profile_default);
    if (!prof) prof = profile_from_enum(KK_PROFILE_BALANCED);

    int effective_top_k = top_k < prof->result_cap ? top_k : prof->result_cap;

    NamespaceManifest dm = fetch_namespace_manifest(k->db, model.namespace_default, model.scope_default);
    if (!dm.found) {
        free_namespace_manifest(&dm);
        free_model_attachment(&model);
        return -1;
    }
    free_namespace_manifest(&dm);

    const char *resolution[4] = {0};
    int resolution_count = build_resolution_order(&model, resolution, 4);

    QueryResult *internal_results = NULL;
    int count = 0;
    ScorePolicy policy;
    int hit = 0;
    for (int i = 0; i < resolution_count; i++) {
        count = fetch_results_exact_scope(k, query_text, resolution[i], model.namespace_default,
                                          effective_top_k + 1, &internal_results, &policy);
        if (count > 0) { hit = 1; break; }
        free_query_results(internal_results, count);
        internal_results = NULL;
    }

    log_retrieval(k->db, query_text, model.scope_default, model.namespace_default, model.retrieval_mode_default, top_k);

    if (!hit) {
        free_model_attachment(&model);
        *results_out = NULL;
        return 0;
    }

    if (count > effective_top_k) count = effective_top_k;

    kk_result *out = xmalloc((size_t)(count > 0 ? count : 1) * sizeof(*out));
    for (int i = 0; i < count; i++) {
        memset(&out[i], 0, sizeof(out[i]));
        out[i].chunk_id = internal_results[i].chunk_id;
        out[i].doc_id = internal_results[i].doc_id;
        out[i].version_num = internal_results[i].version_num;
        out[i].ordinal = internal_results[i].ordinal;
        out[i].seen_count = internal_results[i].seen_count;
        out[i].path = xstrdup(internal_results[i].path ? internal_results[i].path : "");
        out[i].namespace_name = xstrdup(internal_results[i].namespace_name ? internal_results[i].namespace_name : "");
        out[i].scope_name = xstrdup(internal_results[i].scope_name ? internal_results[i].scope_name : "");
        out[i].section_title = xstrdup(internal_results[i].section_title ? internal_results[i].section_title : "");
        BudgetedText bt = make_excerpt_budgeted(internal_results[i].raw_text, (size_t)prof->excerpt_chars);
        out[i].text = bt.text;
        out[i].text_len = (int)strlen(bt.text);
        out[i].resonance = internal_results[i].resonance;
        out[i].lexical = internal_results[i].lexical_norm;
        out[i].recency = internal_results[i].recency;
        out[i].trust = internal_results[i].trust;
        out[i].linkage = internal_results[i].linkage;
        out[i].freshness = internal_results[i].freshness;
        out[i].hebbian_boost = internal_results[i].hebbian_boost;
        out[i].sha256 = xstrdup(internal_results[i].sha256 ? internal_results[i].sha256 : "");
        out[i].ingest_ts = xstrdup(internal_results[i].ingest_ts ? internal_results[i].ingest_ts : "");
        out[i].char_delta = internal_results[i].char_delta;
        out[i].change_ratio = internal_results[i].change_ratio;
        out[i].diff_summary = xstrdup(internal_results[i].diff_summary ? internal_results[i].diff_summary : "");
    }

    free_query_results(internal_results, count);
    free_model_attachment(&model);
    *results_out = out;
    return count;
}

/* ═══════════════════════════════════════════════════════════════════
 * PUBLIC API — MODEL MANAGEMENT
 * ═══════════════════════════════════════════════════════════════════ */

int kk_attach_model(kk_ctx *k, const char *model_name,
                    const char *scope_default, const char *namespace_default) {
    if (!kk_is_ready(k) || !model_name || !scope_default || !namespace_default) return -1;
    if (validate_model_binding_lib(model_name, scope_default, namespace_default) != 0) return -1;
    if (require_manifest_bound_namespace_lib(k->db, namespace_default, scope_default) != 0) return -1;

    ModelAttachment prior = fetch_model_attachment_with_state(k->db, model_name, 0);
    begin_tx(k->db);
    upsert_model_registry(k->db, model_name, scope_default, namespace_default,
                          KK_DEFAULT_RETRIEVAL_MODE, KK_DEFAULT_PACKET_MODE, KK_DEFAULT_QUERY_PROFILE, 1);
    append_model_attachment_event(k->db, model_name, model_event_type_from_prior(&prior),
                                  prior.scope_default, scope_default,
                                  prior.namespace_default, namespace_default,
                                  prior.retrieval_mode_default, KK_DEFAULT_RETRIEVAL_MODE,
                                  prior.packet_mode_default, KK_DEFAULT_PACKET_MODE,
                                  prior.query_profile_default, KK_DEFAULT_QUERY_PROFILE,
                                  "attach-model");
    commit_tx(k->db);
    free_model_attachment(&prior);
    return 0;
}

int kk_update_model(kk_ctx *k, const char *model_name,
                    const char *scope_default, const char *namespace_default) {
    if (!kk_is_ready(k) || !model_name || !scope_default || !namespace_default) return -1;
    if (validate_model_binding_lib(model_name, scope_default, namespace_default) != 0) return -1;

    ModelAttachment prior = fetch_model_attachment(k->db, model_name);
    if (!prior.found) { free_model_attachment(&prior); return -1; }
    if (require_manifest_bound_namespace_lib(k->db, namespace_default, scope_default) != 0) {
        free_model_attachment(&prior);
        return -1;
    }
    if (!validate_mode(prior.retrieval_mode_default) || !validate_packet_mode(prior.packet_mode_default) ||
        !find_query_profile(prior.query_profile_default)) {
        free_model_attachment(&prior);
        return -1;
    }

    begin_tx(k->db);
    upsert_model_registry(k->db, model_name, scope_default, namespace_default,
                          prior.retrieval_mode_default, prior.packet_mode_default, prior.query_profile_default, 1);
    append_model_attachment_event(k->db, model_name, "update",
                                  prior.scope_default, scope_default,
                                  prior.namespace_default, namespace_default,
                                  prior.retrieval_mode_default, prior.retrieval_mode_default,
                                  prior.packet_mode_default, prior.packet_mode_default,
                                  prior.query_profile_default, prior.query_profile_default,
                                  "update-model");
    commit_tx(k->db);
    free_model_attachment(&prior);
    return 0;
}

int kk_set_model_profile(kk_ctx *k, const char *model_name, kk_profile profile) {
    if (!kk_is_ready(k) || !model_name) return -1;
    const QueryProfile *prof = profile_from_enum(profile);
    if (!prof) return -1;

    ModelAttachment prior = fetch_model_attachment(k->db, model_name);
    if (!prior.found) { free_model_attachment(&prior); return -1; }

    begin_tx(k->db);
    upsert_model_registry(k->db, model_name, prior.scope_default, prior.namespace_default,
                          prior.retrieval_mode_default, prior.packet_mode_default, prof->name, 1);
    append_model_attachment_event(k->db, model_name, "profile",
                                  prior.scope_default, prior.scope_default,
                                  prior.namespace_default, prior.namespace_default,
                                  prior.retrieval_mode_default, prior.retrieval_mode_default,
                                  prior.packet_mode_default, prior.packet_mode_default,
                                  prior.query_profile_default, prof->name,
                                  "model-set-profile");
    commit_tx(k->db);
    free_model_attachment(&prior);
    return 0;
}

int kk_detach_model(kk_ctx *k, const char *model_name) {
    if (!kk_is_ready(k) || !model_name) return -1;

    ModelAttachment prior = fetch_model_attachment(k->db, model_name);
    if (!prior.found) { free_model_attachment(&prior); return -1; }

    begin_tx(k->db);
    sqlite3_stmt *stmt = NULL;
    const char *sql = "UPDATE model_registry SET is_active=0, detached_ts=CURRENT_TIMESTAMP, updated_ts=CURRENT_TIMESTAMP WHERE model_name=? AND is_active=1;";
    if (sqlite3_prepare_v2(k->db, sql, -1, &stmt, NULL) != SQLITE_OK) {
        die_sqlite(k->db, "prepare detach-model");
        commit_tx(k->db);
        free_model_attachment(&prior);
        return -1;
    }
    sqlite3_bind_text(stmt, 1, model_name, -1, SQLITE_STATIC);
    if (sqlite3_step(stmt) != SQLITE_DONE) die_sqlite(k->db, "detach model");
    sqlite3_finalize(stmt);
    append_model_attachment_event(k->db, model_name, "detach",
                                  prior.scope_default, NULL,
                                  prior.namespace_default, NULL,
                                  prior.retrieval_mode_default, NULL,
                                  prior.packet_mode_default, NULL,
                                  prior.query_profile_default, NULL,
                                  "detach-model");
    commit_tx(k->db);
    free_model_attachment(&prior);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * PUBLIC API — NAMESPACE MANAGEMENT
 * ═══════════════════════════════════════════════════════════════════ */

int kk_set_namespace(kk_ctx *k, const char *namespace_name,
                     const char *scope, const char *description) {
    if (!kk_is_ready(k) || !namespace_name || !*namespace_name || !scope || !description || !*description) return -1;
    if (require_scope_lib(scope) != 0) return -1;

    get_namespace_id(k->db, namespace_name, scope);

    sqlite3_stmt *stmt = NULL;
    const char *sql =
        "INSERT INTO namespace_manifest(namespace, scope, description, owner_model, created_ts, updated_ts) VALUES(?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
        "ON CONFLICT(namespace, scope) DO UPDATE SET description=excluded.description, updated_ts=CURRENT_TIMESTAMP;";
    if (sqlite3_prepare_v2(k->db, sql, -1, &stmt, NULL) != SQLITE_OK) { die_sqlite(k->db, "prepare namespace-set"); return -1; }
    sqlite3_bind_text(stmt, 1, namespace_name, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, scope, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, description, -1, SQLITE_STATIC);
    if (!strncmp(scope, "private:", 8)) sqlite3_bind_text(stmt, 4, scope + 8, -1, SQLITE_STATIC); else sqlite3_bind_null(stmt, 4);
    if (sqlite3_step(stmt) != SQLITE_DONE) { die_sqlite(k->db, "namespace set"); sqlite3_finalize(stmt); return -1; }
    sqlite3_finalize(stmt);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * PUBLIC API — INTEGRITY & MAINTENANCE
 * ═══════════════════════════════════════════════════════════════════ */

int kk_check_integrity(kk_ctx *k) {
    if (!kk_is_ready(k)) return -1;
    int issue_count = 0;

    sqlite3_stmt *stmt = NULL;
    if (sqlite3_prepare_v2(k->db, "PRAGMA integrity_check;", -1, &stmt, NULL) != SQLITE_OK) return -1;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *msg = (const char *)sqlite3_column_text(stmt, 0);
        if (!msg || !strcmp(msg, "ok")) continue;
        issue_count++;
    }
    sqlite3_finalize(stmt);

    issue_count += integrity_count_query(k->db,
        "SELECT COUNT(*) FROM document_versions dv LEFT JOIN documents d ON d.id=dv.document_id WHERE d.id IS NULL;");
    issue_count += integrity_count_query(k->db,
        "SELECT COUNT(*) FROM chunks c LEFT JOIN document_versions dv ON dv.id=c.version_id WHERE dv.id IS NULL;");
    issue_count += integrity_count_query(k->db,
        "SELECT COUNT(*) FROM chunks c JOIN document_versions dv ON dv.id=c.version_id AND dv.is_latest=1 "
        "LEFT JOIN chunk_fts f ON f.chunk_id=c.id WHERE f.chunk_id IS NULL;");
    issue_count += integrity_count_query(k->db,
        "SELECT COUNT(*) FROM namespace_manifest nm JOIN namespaces ns ON ns.name=nm.namespace WHERE ns.scope<>nm.scope;");
    issue_count += integrity_count_query(k->db,
        "SELECT COUNT(*) FROM namespace_manifest nm "
        "WHERE (nm.scope LIKE 'private:%' AND COALESCE(nm.owner_model, '') <> substr(nm.scope, 9)) "
        "   OR (nm.scope NOT LIKE 'private:%' AND COALESCE(nm.owner_model, '') <> '');");
    issue_count += integrity_count_query(k->db,
        "SELECT COUNT(*) FROM model_registry mr "
        "LEFT JOIN namespace_manifest nm ON nm.namespace=mr.namespace_default AND nm.scope=mr.scope_default "
        "WHERE mr.is_active=1 AND nm.id IS NULL;");
    issue_count += integrity_count_query(k->db,
        "SELECT COUNT(*) FROM model_registry mr "
        "JOIN namespace_manifest nm ON nm.namespace=mr.namespace_default AND nm.scope=mr.scope_default "
        "WHERE mr.is_active=1 AND COALESCE(nm.owner_model, '') <> '' AND nm.owner_model <> mr.model_name;");
    issue_count += integrity_count_query(k->db,
        "SELECT "
        "  (SELECT COUNT(*) FROM namespaces WHERE scope<>'public' AND scope NOT LIKE 'shared:%' AND scope NOT LIKE 'private:%') + "
        "  (SELECT COUNT(*) FROM namespace_manifest WHERE scope<>'public' AND scope NOT LIKE 'shared:%' AND scope NOT LIKE 'private:%') + "
        "  (SELECT COUNT(*) FROM model_registry WHERE is_active=1 AND scope_default<>'public' AND scope_default NOT LIKE 'shared:%' AND scope_default NOT LIKE 'private:%');");

    return issue_count;
}

int kk_rebuild_fts(kk_ctx *k) {
    if (!kk_is_ready(k)) return -1;
    begin_tx(k->db);
    if (exec_sql(k->db, "DELETE FROM chunk_fts;") != SQLITE_OK) { commit_tx(k->db); return -1; }
    if (exec_sql(k->db,
            "INSERT INTO chunk_fts(chunk_id, raw_text, namespace, scope, path, filename, section_title) "
            "SELECT c.id, c.raw_text, ns.name, ns.scope, d.path, d.filename, s.heading "
            "FROM chunks c "
            "JOIN documents d ON d.id=c.parent_document_id "
            "JOIN namespaces ns ON ns.id=d.namespace_id "
            "JOIN sections s ON s.id=c.parent_section_id "
            "ORDER BY c.id ASC;") != SQLITE_OK) {
        commit_tx(k->db);
        return -1;
    }
    commit_tx(k->db);
    return 0;
}

int kk_get_stats(kk_ctx *k, kk_stats *out) {
    if (!kk_is_ready(k) || !out) return -1;
    memset(out, 0, sizeof(*out));

    const char *sql =
        "SELECT (SELECT COUNT(*) FROM documents),"
        "       (SELECT COUNT(*) FROM document_versions),"
        "       (SELECT COUNT(*) FROM chunks),"
        "       (SELECT COUNT(*) FROM namespaces),"
        "       (SELECT COUNT(*) FROM model_registry WHERE is_active=1),"
        "       (SELECT COUNT(*) FROM links);";
    sqlite3_stmt *stmt = NULL;
    if (sqlite3_prepare_v2(k->db, sql, -1, &stmt, NULL) != SQLITE_OK) return -1;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        out->documents = sqlite3_column_int(stmt, 0);
        out->versions = sqlite3_column_int(stmt, 1);
        out->chunks = sqlite3_column_int(stmt, 2);
        out->namespaces = sqlite3_column_int(stmt, 3);
        out->models = sqlite3_column_int(stmt, 4);
        out->links = sqlite3_column_int(stmt, 5);
    }
    sqlite3_finalize(stmt);

    /* Get DB file size */
    sqlite3_stmt *ps = NULL;
    if (sqlite3_prepare_v2(k->db, "PRAGMA page_count;", -1, &ps, NULL) == SQLITE_OK) {
        if (sqlite3_step(ps) == SQLITE_ROW) {
            int64_t pages = sqlite3_column_int64(ps, 0);
            sqlite3_finalize(ps);
            if (sqlite3_prepare_v2(k->db, "PRAGMA page_size;", -1, &ps, NULL) == SQLITE_OK) {
                if (sqlite3_step(ps) == SQLITE_ROW) {
                    int64_t page_size = sqlite3_column_int64(ps, 0);
                    out->db_size_bytes = pages * page_size;
                }
            }
        }
    }
    if (ps) sqlite3_finalize(ps);

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * PUBLIC API — MEMORY MANAGEMENT
 * ═══════════════════════════════════════════════════════════════════ */

void kk_free_results(kk_result *results, int count) {
    if (!results) return;
    for (int i = 0; i < count; i++) {
        free((char *)results[i].path);
        free((char *)results[i].namespace_name);
        free((char *)results[i].scope_name);
        free((char *)results[i].section_title);
        free((char *)results[i].text);
        free((char *)results[i].sha256);
        free((char *)results[i].ingest_ts);
        free((char *)results[i].diff_summary);
    }
    free(results);
}

/* ═══════════════════════════════════════════════════════════════════
 * PUBLIC API — JSON EXPORT
 * ═══════════════════════════════════════════════════════════════════ */

char *kk_state_to_json(kk_ctx *k) {
    if (!kk_is_ready(k)) return xstrdup("{\"error\":\"not_ready\"}");

    kk_stats stats;
    kk_get_stats(k, &stats);

    StrBuf sb;
    sb_init(&sb);
    sb_append(&sb, "{");
    sb_append(&sb, "\"ready\":true,");
    sb_appendf(&sb, "\"documents\":%d,", stats.documents);
    sb_appendf(&sb, "\"versions\":%d,", stats.versions);
    sb_appendf(&sb, "\"chunks\":%d,", stats.chunks);
    sb_appendf(&sb, "\"namespaces\":%d,", stats.namespaces);
    sb_appendf(&sb, "\"models\":%d,", stats.models);
    sb_appendf(&sb, "\"links\":%d,", stats.links);
    sb_appendf(&sb, "\"db_size_bytes\":%" PRId64 ",", stats.db_size_bytes);
    sb_appendf(&sb, "\"has_embedder\":%s,", k->embedder ? "true" : "false");
    sb_appendf(&sb, "\"has_hebbian_bridge\":%s,", k->bridge ? "true" : "false");

    /* List attached models */
    sb_append(&sb, "\"attached_models\":[");
    sqlite3_stmt *stmt = NULL;
    const char *models_sql = "SELECT model_name, scope_default, namespace_default, query_profile_default FROM model_registry WHERE is_active=1 ORDER BY model_name ASC;";
    if (sqlite3_prepare_v2(k->db, models_sql, -1, &stmt, NULL) == SQLITE_OK) {
        int first = 1;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            if (!first) sb_append(&sb, ",");
            first = 0;
            sb_append(&sb, "{");
            sb_append(&sb, "\"model_name\":"); json_append_string(&sb, (const char *)sqlite3_column_text(stmt, 0)); sb_append(&sb, ",");
            sb_append(&sb, "\"scope\":"); json_append_string(&sb, (const char *)sqlite3_column_text(stmt, 1)); sb_append(&sb, ",");
            sb_append(&sb, "\"namespace\":"); json_append_string(&sb, (const char *)sqlite3_column_text(stmt, 2)); sb_append(&sb, ",");
            sb_append(&sb, "\"profile\":"); json_append_string(&sb, (const char *)sqlite3_column_text(stmt, 3));
            sb_append(&sb, "}");
        }
        sqlite3_finalize(stmt);
    }
    sb_append(&sb, "],");

    /* List namespaces */
    sb_append(&sb, "\"namespace_list\":[");
    const char *ns_sql = "SELECT namespace, scope, description FROM namespace_manifest ORDER BY scope ASC, namespace ASC;";
    if (sqlite3_prepare_v2(k->db, ns_sql, -1, &stmt, NULL) == SQLITE_OK) {
        int first = 1;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            if (!first) sb_append(&sb, ",");
            first = 0;
            sb_append(&sb, "{");
            sb_append(&sb, "\"namespace\":"); json_append_string(&sb, (const char *)sqlite3_column_text(stmt, 0)); sb_append(&sb, ",");
            sb_append(&sb, "\"scope\":"); json_append_string(&sb, (const char *)sqlite3_column_text(stmt, 1)); sb_append(&sb, ",");
            sb_append(&sb, "\"description\":"); json_append_string(&sb, (const char *)sqlite3_column_text(stmt, 2));
            sb_append(&sb, "}");
        }
        sqlite3_finalize(stmt);
    }
    sb_append(&sb, "]");

    sb_append(&sb, "}");
    return sb.data;
}

char *kk_results_to_json(const kk_result *results, int count,
                         const char *query, const char *model_name) {
    StrBuf sb;
    sb_init(&sb);
    sb_append(&sb, "{");
    sb_append(&sb, "\"packet_schema_version\":"); json_append_string(&sb, KK_PACKET_SCHEMA_VERSION); sb_append(&sb, ",");
    sb_append(&sb, "\"query\":"); json_append_string(&sb, query ? query : ""); sb_append(&sb, ",");
    sb_append(&sb, "\"model_name\":"); json_append_string(&sb, model_name ? model_name : ""); sb_append(&sb, ",");
    sb_appendf(&sb, "\"result_count\":%d,", count);
    sb_append(&sb, "\"results\":[");
    for (int i = 0; i < count; i++) {
        if (i) sb_append(&sb, ",");
        sb_append(&sb, "{");
        sb_appendf(&sb, "\"chunk_id\":%d,", results[i].chunk_id);
        sb_appendf(&sb, "\"doc_id\":%d,", results[i].doc_id);
        sb_appendf(&sb, "\"version_num\":%d,", results[i].version_num);
        sb_appendf(&sb, "\"ordinal\":%d,", results[i].ordinal);
        sb_append(&sb, "\"path\":"); json_append_string(&sb, results[i].path); sb_append(&sb, ",");
        sb_append(&sb, "\"namespace\":"); json_append_string(&sb, results[i].namespace_name); sb_append(&sb, ",");
        sb_append(&sb, "\"scope\":"); json_append_string(&sb, results[i].scope_name); sb_append(&sb, ",");
        sb_append(&sb, "\"section_title\":"); json_append_string(&sb, results[i].section_title); sb_append(&sb, ",");
        sb_append(&sb, "\"text\":"); json_append_string(&sb, results[i].text); sb_append(&sb, ",");
        sb_appendf(&sb, "\"resonance\":%.6f,", results[i].resonance);
        sb_appendf(&sb, "\"lexical\":%.6f,", results[i].lexical);
        sb_appendf(&sb, "\"recency\":%.6f,", results[i].recency);
        sb_appendf(&sb, "\"trust\":%.6f,", results[i].trust);
        sb_appendf(&sb, "\"linkage\":%.6f,", results[i].linkage);
        sb_appendf(&sb, "\"freshness\":%.6f,", results[i].freshness);
        sb_appendf(&sb, "\"hebbian_boost\":%.6f,", results[i].hebbian_boost);
        sb_append(&sb, "\"sha256\":"); json_append_string(&sb, results[i].sha256); sb_append(&sb, ",");
        sb_append(&sb, "\"ingest_ts\":"); json_append_string(&sb, results[i].ingest_ts); sb_append(&sb, ",");
        sb_appendf(&sb, "\"char_delta\":%d,", results[i].char_delta);
        sb_appendf(&sb, "\"change_ratio\":%.6f,", results[i].change_ratio);
        sb_append(&sb, "\"diff_summary\":"); json_append_string(&sb, results[i].diff_summary);
        sb_append(&sb, "}");
    }
    sb_append(&sb, "]}");
    return sb.data;
}

/* ═══════════════════════════════════════════════════════════════════
 * STANDALONE CLI
 * ═══════════════════════════════════════════════════════════════════ */

#ifdef KK_STANDALONE

static void print_chunk_excerpt(const char *text, size_t limit) {
    size_t len = strlen(text);
    if (len <= limit) {
        printf("%s\n", text);
        return;
    }
    fwrite(text, 1, limit, stdout);
    printf("...\n");
}

static void render_query_results(const char *mode, const char *query_text, const char *access_scope, const char *namespace_filter, QueryResult *results, int count, const ScorePolicy *policy) {
    char policy_buf[256];
    format_score_policy(policy, policy_buf, sizeof(policy_buf));

    if (!strcmp(mode, "compressed")) {
        StrBuf sb;
        sb_init(&sb);
        sb_append(&sb, "{");
        sb_append(&sb, "\"packet_schema_version\":"); json_append_string(&sb, KK_PACKET_SCHEMA_VERSION); sb_append(&sb, ",");
        sb_append(&sb, "\"packet_mode\":"); json_append_string(&sb, KK_DEFAULT_PACKET_MODE); sb_append(&sb, ",");
        sb_append(&sb, "\"query\":"); json_append_string(&sb, query_text); sb_append(&sb, ",");
        sb_append(&sb, "\"scope\":"); json_append_string(&sb, access_scope); sb_append(&sb, ",");
        sb_append(&sb, "\"namespace\":"); json_append_string(&sb, namespace_filter ? namespace_filter : ""); sb_append(&sb, ",");
        sb_append(&sb, "\"retrieval_mode\":"); json_append_string(&sb, mode); sb_append(&sb, ",");
        append_score_policy_json(&sb, policy); sb_append(&sb, ",");
        sb_append(&sb, "\"results\":[");
        for (int i = 0; i < count; i++) {
            if (i) sb_append(&sb, ",");
            append_result_json(&sb, &results[i], query_text);
        }
        sb_append(&sb, "]}\n");
        fputs(sb.data, stdout);
        free(sb.data);
        return;
    }

    for (int i = 0; i < count; i++) {
        char reason[768];
        explain_score(&results[i], query_text, reason, sizeof(reason));
        printf("result %d\n", i + 1);
        printf("  doc: %s\n", results[i].path);
        printf("  version: %d (version_id=%d sha=%s)\n", results[i].version_num, results[i].version_id, results[i].sha256);
        printf("  chunk: %d section: %s\n", results[i].chunk_id, results[i].section_title);
        printf("  namespace: %s\n", results[i].namespace_name);
        printf("  scope: %s\n", results[i].scope_name);
        printf("  policy: %s\n", policy_buf);
        printf("  score: resonance=%.3f lexical=%.3f lexical_norm=%.3f recency=%.3f linkage=%.3f trust=%.3f\n",
               results[i].resonance, results[i].lexical, results[i].lexical_norm, results[i].recency, results[i].linkage, results[i].trust);
        printf("  weighted: lexical=%.3f recency=%.3f trust=%.3f linkage=%.3f scope=%.3f namespace=%.3f freshness=%.3f\n",
               results[i].weighted_lexical, results[i].weighted_recency, results[i].weighted_trust,
               results[i].weighted_linkage, results[i].weighted_scope, results[i].weighted_namespace,
               results[i].weighted_freshness);
        printf("  lineage: %s (char_delta=%+d token_delta=%+d change_ratio=%.2f)\n",
               results[i].diff_summary, results[i].char_delta, results[i].token_delta, results[i].change_ratio);
        printf("  why: %s\n", reason);
        if (!strcmp(mode, "citation")) {
            printf("  citation: file=%s version=%d chunk=%d ingest_ts=%s last_seen_ts=%s\n",
                   results[i].filename, results[i].version_num, results[i].chunk_id,
                   results[i].ingest_ts, results[i].last_seen_ts);
        }
        printf("  text: ");
        print_chunk_excerpt(results[i].raw_text, 260);
        printf("\n");
    }
}

static void sigint_handler(int signo) {
    (void)signo;
    kk_stop = 1;
}

static void sleep_interval(int seconds) {
    struct timespec req = {seconds, 0}, rem = {0, 0};
    while (!kk_stop && nanosleep(&req, &rem) != 0) {
        if (errno != EINTR) break;
        req = rem;
    }
}

static void print_watch_cycle_banner(int cycle, int interval_sec) {
    time_t now = time(NULL);
    struct tm tmv;
    gmtime_r(&now, &tmv);
    char stamp[64];
    strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%SZ", &tmv);
    printf("watch cycle %d @ %s interval=%ds\n", cycle, stamp, interval_sec);
}

static void usage(void) {
    fprintf(stderr,
            "Knowledge Kernel CLI (library mode)\n"
            "Usage:\n"
            "  kk init <db>\n"
            "  kk ingest <db> <dir> <namespace> [scope]\n"
            "  kk watch <db> <dir> <namespace> [scope] [interval_sec] [cycles]\n"
            "  kk query <db> <query> <access_scope> <top_k> [mode] [namespace_filter]\n"
            "  kk attach-model <db> <model_name> <scope_default> <namespace_default>\n"
            "  kk update-model <db> <model_name> <scope_default> <namespace_default>\n"
            "  kk model-set-profile <db> <model_name> <tiny|balanced|deep>\n"
            "  kk list-models <db>\n"
            "  kk detach-model <db> <model_name>\n"
            "  kk ask <db> <model_name> <query> [top_k]\n"
            "  kk namespace-set <db> <namespace> <scope> <description>\n"
            "  kk namespace-list <db>\n"
            "  kk check-integrity <db>\n"
            "  kk rebuild-fts <db>\n"
            "  kk stats <db>\n"
            "\nModes: raw | citation | compressed\n"
            "Scopes: public | shared:<name> | private:<model>\n"
            "Env: KK_SCORE_POLICY=lexical=0.40,recency=0.10,trust=0.10,linkage=0.16,scope=0.10,namespace=0.08,freshness=0.06\n");
    exit(1);
}

int main(int argc, char **argv) {
    if (argc < 2) usage();

    /* Commands that don't need db */
    if (!strcmp(argv[1], "profiles")) {
        const QueryProfile *profiles = query_profiles();
        for (int i = 0; i < query_profile_count(); i++) {
            printf("profile=%s\n", profiles[i].name);
            printf("  result_cap=%d\n", profiles[i].result_cap);
            printf("  excerpt_chars=%d\n", profiles[i].excerpt_chars);
        }
        return 0;
    }

    if (argc < 3) usage();

    const char *db_path = argv[2];

    if (!strcmp(argv[1], "init")) {
        kk_ctx *k = kk_open(db_path);
        if (!k) { fprintf(stderr, "fatal: cannot open %s\n", db_path); return 1; }
        kk_close(k);
        printf("initialized knowledge kernel: %s\n", db_path);
        return 0;
    }

    if (!strcmp(argv[1], "ingest")) {
        if (argc < 5 || argc > 6) usage();
        const char *scope = (argc == 6) ? argv[5] : argv[4];
        require_scope(scope);
        kk_ctx *k = kk_open(db_path);
        if (!k) die("cannot open db");
        int count = kk_ingest_dir(k, argv[3], argv[4], scope);
        printf("ingest complete: %d file(s) updated in namespace=%s scope=%s\n", count, argv[4], scope);
        kk_close(k);
        return 0;
    }

    if (!strcmp(argv[1], "watch")) {
        if (argc < 5 || argc > 8) usage();
        const char *scope = (argc >= 6) ? argv[5] : argv[4];
        require_scope(scope);
        int interval_sec = (argc >= 7) ? atoi(argv[6]) : KK_DEFAULT_WATCH_INTERVAL;
        int cycles = (argc >= 8) ? atoi(argv[7]) : 0;
        if (interval_sec <= 0) interval_sec = KK_DEFAULT_WATCH_INTERVAL;
        kk_ctx *k = kk_open(db_path);
        if (!k) die("cannot open db");
        signal(SIGINT, sigint_handler);
        int cycle = 0;
        while (!kk_stop && (cycles <= 0 || cycle < cycles)) {
            cycle++;
            print_watch_cycle_banner(cycle, interval_sec);
            int count = kk_ingest_dir(k, argv[3], argv[4], scope);
            printf("watch updated: %d file(s) namespace=%s scope=%s\n", count, argv[4], scope);
            if (cycles > 0 && cycle >= cycles) break;
            sleep_interval(interval_sec);
        }
        kk_close(k);
        return 0;
    }

    if (!strcmp(argv[1], "query")) {
        if (argc < 6 || argc > 8) usage();
        const char *query_text = argv[3];
        const char *access_scope = argv[4];
        int top_k = atoi(argv[5]);
        const char *mode = (argc >= 7) ? argv[6] : "citation";
        const char *namespace_filter = (argc >= 8) ? argv[7] : NULL;
        require_scope(access_scope);
        require_mode(mode);
        kk_ctx *k = kk_open(db_path);
        if (!k) die("cannot open db");
        log_retrieval(k->db, query_text, access_scope, namespace_filter, mode, top_k);
        QueryResult *results = NULL;
        ScorePolicy policy;
        int count = fetch_results(k, query_text, access_scope, namespace_filter, top_k, &results, &policy);
        if (strcmp(mode, "compressed")) {
            printf("query: %s\n", query_text);
            printf("access_scope: %s\n", access_scope);
            printf("namespace_filter: %s\n", namespace_filter ? namespace_filter : "<none>");
            printf("mode: %s\n", mode);
            printf("hits: %d\n\n", count);
        }
        render_query_results(mode, query_text, access_scope, namespace_filter, results, count, &policy);
        free_query_results(results, count);
        kk_close(k);
        return 0;
    }

    if (!strcmp(argv[1], "attach-model")) {
        if (argc != 6) usage();
        validate_model_binding(argv[3], argv[4], argv[5]);
        kk_ctx *k = kk_open(db_path);
        if (!k) die("cannot open db");
        require_manifest_bound_namespace(k->db, argv[5], argv[4]);
        if (kk_attach_model(k, argv[3], argv[4], argv[5]) != 0) {
            kk_close(k);
            die("attach-model failed");
        }
        printf("attached model=%s scope_default=%s namespace_default=%s\n", argv[3], argv[4], argv[5]);
        kk_close(k);
        return 0;
    }

    if (!strcmp(argv[1], "update-model")) {
        if (argc != 6) usage();
        validate_model_binding(argv[3], argv[4], argv[5]);
        kk_ctx *k = kk_open(db_path);
        if (!k) die("cannot open db");
        if (kk_update_model(k, argv[3], argv[4], argv[5]) != 0) {
            kk_close(k);
            die("update-model failed");
        }
        printf("updated model=%s scope_default=%s namespace_default=%s\n", argv[3], argv[4], argv[5]);
        kk_close(k);
        return 0;
    }

    if (!strcmp(argv[1], "model-set-profile")) {
        if (argc != 5) usage();
        require_query_profile(argv[4]);
        kk_profile prof = KK_PROFILE_BALANCED;
        if (!strcmp(argv[4], "tiny")) prof = KK_PROFILE_TINY;
        else if (!strcmp(argv[4], "deep")) prof = KK_PROFILE_DEEP;
        kk_ctx *k = kk_open(db_path);
        if (!k) die("cannot open db");
        if (kk_set_model_profile(k, argv[3], prof) != 0) {
            kk_close(k);
            die("model-set-profile failed");
        }
        printf("model profile updated model=%s query_profile_default=%s\n", argv[3], argv[4]);
        kk_close(k);
        return 0;
    }

    if (!strcmp(argv[1], "list-models")) {
        kk_ctx *k = kk_open(db_path);
        if (!k) die("cannot open db");
        sqlite3_stmt *stmt = NULL;
        const char *sql = "SELECT model_name, scope_default, namespace_default, retrieval_mode_default, packet_mode_default, query_profile_default, created_ts, updated_ts, notes FROM model_registry WHERE is_active=1 ORDER BY model_name ASC;";
        if (sqlite3_prepare_v2(k->db, sql, -1, &stmt, NULL) != SQLITE_OK) die_sqlite(k->db, "prepare list-models");
        printf("model_name\tscope_default\tnamespace_default\tretrieval_mode_default\tpacket_mode_default\tquery_profile_default\tcreated_ts\tupdated_ts\tnotes\n");
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            printf("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
                   sqlite3_column_text(stmt, 0), sqlite3_column_text(stmt, 1), sqlite3_column_text(stmt, 2),
                   sqlite3_column_text(stmt, 3), sqlite3_column_text(stmt, 4), sqlite3_column_text(stmt, 5),
                   sqlite3_column_text(stmt, 6), sqlite3_column_text(stmt, 7), sqlite3_column_text(stmt, 8));
        }
        sqlite3_finalize(stmt);
        kk_close(k);
        return 0;
    }

    if (!strcmp(argv[1], "detach-model")) {
        if (argc != 4) usage();
        kk_ctx *k = kk_open(db_path);
        if (!k) die("cannot open db");
        if (kk_detach_model(k, argv[3]) != 0) {
            kk_close(k);
            die("detach-model failed");
        }
        printf("detached model=%s\n", argv[3]);
        kk_close(k);
        return 0;
    }

    if (!strcmp(argv[1], "ask")) {
        if (argc < 5 || argc > 6) usage();
        int top_k = (argc == 6) ? atoi(argv[5]) : 5;
        kk_ctx *k = kk_open(db_path);
        if (!k) die("cannot open db");
        kk_result *results = NULL;
        int count = kk_ask(k, argv[3], argv[4], top_k, &results);
        if (count < 0) {
            fprintf(stderr, "ask failed for model=%s\n", argv[3]);
            kk_close(k);
            return 2;
        }
        char *json = kk_results_to_json(results, count, argv[4], argv[3]);
        fputs(json, stdout);
        fputs("\n", stdout);
        free(json);
        kk_free_results(results, count);
        kk_close(k);
        return 0;
    }

    if (!strcmp(argv[1], "namespace-set")) {
        if (argc != 6) usage();
        kk_ctx *k = kk_open(db_path);
        if (!k) die("cannot open db");
        if (kk_set_namespace(k, argv[3], argv[4], argv[5]) != 0) {
            kk_close(k);
            die("namespace-set failed");
        }
        printf("namespace-manifest namespace=%s scope=%s updated\n", argv[3], argv[4]);
        kk_close(k);
        return 0;
    }

    if (!strcmp(argv[1], "namespace-list")) {
        kk_ctx *k = kk_open(db_path);
        if (!k) die("cannot open db");
        sqlite3_stmt *stmt = NULL;
        const char *sql = "SELECT namespace, scope, description, COALESCE(owner_model, ''), created_ts, updated_ts FROM namespace_manifest ORDER BY scope ASC, namespace ASC;";
        if (sqlite3_prepare_v2(k->db, sql, -1, &stmt, NULL) != SQLITE_OK) die_sqlite(k->db, "prepare namespace-list");
        printf("namespace\tscope\towner_model\tcreated_ts\tupdated_ts\tdescription\n");
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            printf("%s\t%s\t%s\t%s\t%s\t%s\n",
                   sqlite3_column_text(stmt, 0), sqlite3_column_text(stmt, 1), sqlite3_column_text(stmt, 3),
                   sqlite3_column_text(stmt, 4), sqlite3_column_text(stmt, 5), sqlite3_column_text(stmt, 2));
        }
        sqlite3_finalize(stmt);
        kk_close(k);
        return 0;
    }

    if (!strcmp(argv[1], "check-integrity")) {
        kk_ctx *k = kk_open(db_path);
        if (!k) die("cannot open db");
        int issues = kk_check_integrity(k);
        printf("integrity check: %d issue(s)\n", issues);
        kk_close(k);
        return issues > 0 ? 2 : 0;
    }

    if (!strcmp(argv[1], "rebuild-fts")) {
        kk_ctx *k = kk_open(db_path);
        if (!k) die("cannot open db");
        if (kk_rebuild_fts(k) != 0) {
            kk_close(k);
            die("rebuild-fts failed");
        }
        printf("rebuild-fts complete\n");
        kk_close(k);
        return 0;
    }

    if (!strcmp(argv[1], "stats")) {
        kk_ctx *k = kk_open(db_path);
        if (!k) die("cannot open db");
        kk_stats s;
        kk_get_stats(k, &s);
        printf("documents: %d\n", s.documents);
        printf("versions: %d\n", s.versions);
        printf("chunks: %d\n", s.chunks);
        printf("namespaces: %d\n", s.namespaces);
        printf("models: %d\n", s.models);
        printf("links: %d\n", s.links);
        printf("db_size_bytes: %" PRId64 "\n", s.db_size_bytes);
        kk_close(k);
        return 0;
    }

    usage();
    return 0;
}

#endif /* KK_STANDALONE */
