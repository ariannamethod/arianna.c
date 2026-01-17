// external_brain.c — GPT-2 30M Knowledge Subordinate
// Pure C inference, no PyTorch, no dependencies beyond libc + math
//
// "Brain knows WHAT to say, Arianna knows HOW to say it"
//
// ═══════════════════════════════════════════════════════════════════════════════
// build: gcc -O3 -c external_brain.c -o external_brain.o
// ═══════════════════════════════════════════════════════════════════════════════

#include "external_brain.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

// ═══════════════════════════════════════════════════════════════════════════════
// MATH HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

// IEEE 754 half-precision to single-precision conversion
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        // Subnormal or zero
        if (mant == 0) {
            uint32_t bits = sign;
            float result;
            memcpy(&result, &bits, sizeof(float));
            return result;
        }
        // Subnormal: normalize
        while ((mant & 0x400) == 0) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= 0x3FF;
    } else if (exp == 31) {
        // Inf or NaN
        uint32_t bits = sign | 0x7F800000 | (mant << 13);
        float result;
        memcpy(&result, &bits, sizeof(float));
        return result;
    }

    // Normal number
    exp = exp + (127 - 15);
    uint32_t bits = sign | (exp << 23) | (mant << 13);
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

static inline float gelu(float x) {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float SQRT_2_PI = 0.7978845608f;
    float x3 = x * x * x;
    float inner = SQRT_2_PI * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

static void softmax(float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

static void layer_norm(float* out, const float* x, const float* weight,
                       const float* bias, int n) {
    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;

    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= n;

    // Normalize
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++) {
        out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

static void matmul(float* out, const float* x, const float* w,
                   int n, int d) {
    // out[d] = x[n] @ w[n, d]
    for (int j = 0; j < d; j++) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += x[i] * w[i * d + j];
        }
        out[j] = sum;
    }
}

static void matmul_add_bias(float* out, const float* x, const float* w,
                            const float* b, int n, int d) {
    for (int j = 0; j < d; j++) {
        float sum = b[j];
        for (int i = 0; i < n; i++) {
            sum += x[i] * w[i * d + j];
        }
        out[j] = sum;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BPE TOKENIZER
// ═══════════════════════════════════════════════════════════════════════════════

// Simple string hash for word lookup
static unsigned int hash_str(const char* s) {
    unsigned int h = 5381;
    while (*s) {
        h = ((h << 5) + h) + (unsigned char)*s;
        s++;
    }
    return h;
}

// Parse JSON vocab file
static int parse_vocab_json(BPE_Tokenizer* tok, const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[brain] cannot open vocab: %s\n", path);
        return -1;
    }

    // Get file size
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* buf = (char*)malloc(size + 1);
    if (!buf) { fclose(f); return -1; }
    fread(buf, 1, size, f);
    buf[size] = 0;
    fclose(f);

    // Count entries (roughly)
    int n_entries = 0;
    for (long i = 0; i < size; i++) {
        if (buf[i] == ':') n_entries++;
    }

    // Allocate vocab and hash table
    tok->vocab_size = GPT2_VOCAB_SIZE;
    tok->vocab = (char**)calloc(tok->vocab_size, sizeof(char*));

    tok->hash_size = tok->vocab_size * 2;  // load factor 0.5
    tok->word_keys = (char**)calloc(tok->hash_size, sizeof(char*));
    tok->word_ids = (int*)malloc(tok->hash_size * sizeof(int));
    for (int i = 0; i < tok->hash_size; i++) tok->word_ids[i] = -1;

    // Parse: "token": id format
    char* p = buf;
    while (*p) {
        // Find opening quote
        while (*p && *p != '"') p++;
        if (!*p) break;
        p++;  // skip "

        // Extract token
        char* tok_start = p;
        while (*p && *p != '"') {
            if (*p == '\\' && *(p+1)) p += 2;  // skip escaped chars
            else p++;
        }
        if (!*p) break;

        int tok_len = (int)(p - tok_start);
        p++;  // skip closing "

        // Find colon and number
        while (*p && *p != ':') p++;
        if (!*p) break;
        p++;

        // Parse ID
        while (*p && isspace(*p)) p++;
        int id = atoi(p);

        // Store if valid
        if (id >= 0 && id < tok->vocab_size) {
            // Decode escaped sequences
            char* token = (char*)malloc(tok_len + 1);
            int j = 0;
            for (int i = 0; i < tok_len; i++) {
                if (tok_start[i] == '\\' && i + 1 < tok_len) {
                    char c = tok_start[i + 1];
                    if (c == 'n') { token[j++] = '\n'; i++; }
                    else if (c == 't') { token[j++] = '\t'; i++; }
                    else if (c == 'r') { token[j++] = '\r'; i++; }
                    else if (c == '"') { token[j++] = '"'; i++; }
                    else if (c == '\\') { token[j++] = '\\'; i++; }
                    else if (c == 'u') {
                        // Unicode escape \uXXXX
                        if (i + 5 < tok_len) {
                            char hex[5] = {tok_start[i+2], tok_start[i+3],
                                          tok_start[i+4], tok_start[i+5], 0};
                            int code = (int)strtol(hex, NULL, 16);
                            // Simple UTF-8 encoding
                            if (code < 0x80) {
                                token[j++] = (char)code;
                            } else if (code < 0x800) {
                                token[j++] = 0xC0 | (code >> 6);
                                token[j++] = 0x80 | (code & 0x3F);
                            } else {
                                token[j++] = 0xE0 | (code >> 12);
                                token[j++] = 0x80 | ((code >> 6) & 0x3F);
                                token[j++] = 0x80 | (code & 0x3F);
                            }
                            i += 5;
                        }
                    }
                    else { token[j++] = tok_start[i]; }
                } else {
                    token[j++] = tok_start[i];
                }
            }
            token[j] = 0;

            tok->vocab[id] = token;

            // Add to hash table
            unsigned int h = hash_str(token) % tok->hash_size;
            while (tok->word_keys[h] != NULL) {
                h = (h + 1) % tok->hash_size;
            }
            tok->word_keys[h] = token;
            tok->word_ids[h] = id;
        }

        // Skip to next entry
        while (*p && *p != ',' && *p != '}') p++;
        if (*p == ',') p++;
    }

    free(buf);

    tok->eos_token_id = 50256;
    tok->bos_token_id = 50256;

    fprintf(stderr, "[brain] loaded vocab: %d tokens\n", tok->vocab_size);
    return 0;
}

// Parse BPE merges file
static int parse_merges(BPE_Tokenizer* tok, const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[brain] cannot open merges: %s\n", path);
        return -1;
    }

    // Count lines
    int n_lines = 0;
    char line[512];
    while (fgets(line, sizeof(line), f)) n_lines++;
    fseek(f, 0, SEEK_SET);

    // Skip header line
    fgets(line, sizeof(line), f);
    n_lines--;

    tok->n_merges = n_lines;
    tok->merge_a = (char**)malloc(n_lines * sizeof(char*));
    tok->merge_b = (char**)malloc(n_lines * sizeof(char*));

    int idx = 0;
    while (fgets(line, sizeof(line), f) && idx < n_lines) {
        // Remove newline
        int len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
            line[--len] = 0;
        }

        // Split by space
        char* space = strchr(line, ' ');
        if (space) {
            *space = 0;
            tok->merge_a[idx] = strdup(line);
            tok->merge_b[idx] = strdup(space + 1);
            idx++;
        }
    }
    tok->n_merges = idx;

    fclose(f);
    fprintf(stderr, "[brain] loaded %d BPE merges\n", tok->n_merges);
    return 0;
}

// Lookup token in hash table
static int token_to_id(BPE_Tokenizer* tok, const char* token) {
    unsigned int h = hash_str(token) % tok->hash_size;
    int start = h;
    while (tok->word_keys[h] != NULL) {
        if (strcmp(tok->word_keys[h], token) == 0) {
            return tok->word_ids[h];
        }
        h = (h + 1) % tok->hash_size;
        if (h == start) break;
    }
    return -1;
}

// GPT-2 byte-to-unicode mapping (simplified)
// GPT-2 uses a special mapping where printable ASCII maps directly,
// and other bytes map to unicode codepoints starting at 256
static void byte_to_unicode_char(unsigned char byte, char* out) {
    // Printable ASCII (33-126) and some others map directly
    if ((byte >= 33 && byte <= 126) ||
        (byte >= 161 && byte <= 172) ||
        (byte >= 174 && byte <= 255)) {
        out[0] = (char)byte;
        out[1] = 0;
    } else if (byte == 32) {
        // Space maps to Ġ (U+0120) in GPT-2
        out[0] = (char)0xC4;
        out[1] = (char)0xA0;
        out[2] = 0;
    } else if (byte == 10) {
        // Newline maps to Ċ (U+010A)
        out[0] = (char)0xC4;
        out[1] = (char)0x8A;
        out[2] = 0;
    } else {
        // Other bytes map to unicode 256 + offset
        int code = 256 + byte;
        if (code < 0x800) {
            out[0] = 0xC0 | (code >> 6);
            out[1] = 0x80 | (code & 0x3F);
            out[2] = 0;
        } else {
            out[0] = (char)byte;  // fallback
            out[1] = 0;
        }
    }
}

// BPE encode (byte-level GPT-2 style)
int brain_encode(ExternalBrain* brain, const char* text, int* out_tokens, int max_len) {
    BPE_Tokenizer* tok = &brain->tokenizer;
    int n = 0;
    const char* p = text;

    while (*p && n < max_len - 1) {
        // Find longest matching token starting at p
        int best_len = 0;
        int best_id = -1;

        // Build unicode representation of next few bytes
        char unicode_buf[128];
        int buf_pos = 0;
        const char* q = p;
        int byte_len = 0;

        while (*q && byte_len < 32 && buf_pos < 120) {
            char uc[4];
            byte_to_unicode_char((unsigned char)*q, uc);
            int uc_len = strlen(uc);
            memcpy(unicode_buf + buf_pos, uc, uc_len);
            buf_pos += uc_len;
            byte_len++;
            q++;
        }
        unicode_buf[buf_pos] = 0;

        // Try matching from longest to shortest
        for (int try_len = buf_pos; try_len >= 1; try_len--) {
            char try_buf[128];
            memcpy(try_buf, unicode_buf, try_len);
            try_buf[try_len] = 0;

            int id = token_to_id(tok, try_buf);
            if (id >= 0) {
                // Count how many original bytes this covers
                int bytes_used = 0;
                int uc_pos = 0;
                const char* check = p;
                while (uc_pos < try_len && *check) {
                    char uc[4];
                    byte_to_unicode_char((unsigned char)*check, uc);
                    uc_pos += strlen(uc);
                    check++;
                    bytes_used++;
                }
                best_len = bytes_used;
                best_id = id;
                break;
            }
        }

        if (best_id >= 0) {
            out_tokens[n++] = best_id;
            p += best_len;
        } else {
            // Single byte fallback
            char uc[4];
            byte_to_unicode_char((unsigned char)*p, uc);
            int id = token_to_id(tok, uc);
            if (id >= 0) {
                out_tokens[n++] = id;
            }
            p++;
        }
    }

    return n;
}

// Decode tokens to text
int brain_decode(ExternalBrain* brain, const int* tokens, int n_tokens,
                 char* out_text, int max_len) {
    BPE_Tokenizer* tok = &brain->tokenizer;
    int pos = 0;

    for (int i = 0; i < n_tokens && pos < max_len - 1; i++) {
        int id = tokens[i];
        if (id >= 0 && id < tok->vocab_size && tok->vocab[id]) {
            const char* token = tok->vocab[id];

            // Handle Ġ prefix (space marker)
            const char* p = token;
            if ((unsigned char)p[0] == 0xC4 && (unsigned char)p[1] == 0xA0) {
                if (pos < max_len - 1) out_text[pos++] = ' ';
                p += 2;
            }

            // Copy rest of token
            while (*p && pos < max_len - 1) {
                out_text[pos++] = *p++;
            }
        }
    }

    out_text[pos] = 0;
    return pos;
}

// ═══════════════════════════════════════════════════════════════════════════════
// MODEL LOADING
// ═══════════════════════════════════════════════════════════════════════════════

int brain_load_model(ExternalBrain* brain, const char* path) {
    GPT2_Model* m = &brain->model;

    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[brain] cannot open model: %s\n", path);
        return -1;
    }

    // Read header (7 ints)
    int header[7];
    if (fread(header, sizeof(int), 7, f) != 7) {
        fprintf(stderr, "[brain] cannot read header\n");
        fclose(f);
        return -1;
    }

    m->dim = header[0];
    m->n_layers = header[1];
    m->n_heads = header[2];
    m->vocab_size = header[3];
    m->max_seq = header[4];
    m->hidden_dim = header[5];
    m->use_fp16 = header[6];

    fprintf(stderr, "[brain] model: dim=%d layers=%d heads=%d vocab=%d seq=%d hidden=%d fp16=%d\n",
            m->dim, m->n_layers, m->n_heads, m->vocab_size, m->max_seq, m->hidden_dim, m->use_fp16);

    // Allocate weights
    int dim = m->dim;
    int n_layers = m->n_layers;
    int vocab_size = m->vocab_size;
    int max_seq = m->max_seq;
    int hidden_dim = m->hidden_dim;

    // Helper to read weights (handles fp16 conversion)
    #define READ_WEIGHTS(ptr, count) do { \
        ptr = (float*)malloc((count) * sizeof(float)); \
        if (m->use_fp16) { \
            uint16_t* tmp = (uint16_t*)malloc((count) * sizeof(uint16_t)); \
            fread(tmp, sizeof(uint16_t), count, f); \
            for (int _i = 0; _i < (count); _i++) { \
                ptr[_i] = fp16_to_fp32(tmp[_i]); \
            } \
            free(tmp); \
        } else { \
            fread(ptr, sizeof(float), count, f); \
        } \
    } while(0)

    // Read embeddings
    READ_WEIGHTS(m->wte, vocab_size * dim);
    READ_WEIGHTS(m->wpe, max_seq * dim);

    // Read layers
    m->layers = (GPT2_Layer*)calloc(n_layers, sizeof(GPT2_Layer));
    for (int l = 0; l < n_layers; l++) {
        GPT2_Layer* layer = &m->layers[l];

        READ_WEIGHTS(layer->ln1_weight, dim);
        READ_WEIGHTS(layer->ln1_bias, dim);
        READ_WEIGHTS(layer->c_attn_weight, dim * 3 * dim);
        READ_WEIGHTS(layer->c_attn_bias, 3 * dim);
        READ_WEIGHTS(layer->c_proj_weight, dim * dim);
        READ_WEIGHTS(layer->c_proj_bias, dim);
        READ_WEIGHTS(layer->ln2_weight, dim);
        READ_WEIGHTS(layer->ln2_bias, dim);
        READ_WEIGHTS(layer->c_fc_weight, dim * hidden_dim);
        READ_WEIGHTS(layer->c_fc_bias, hidden_dim);
        READ_WEIGHTS(layer->c_proj2_weight, hidden_dim * dim);
        READ_WEIGHTS(layer->c_proj2_bias, dim);
    }

    // Final LayerNorm
    READ_WEIGHTS(m->ln_f_weight, dim);
    READ_WEIGHTS(m->ln_f_bias, dim);

    #undef READ_WEIGHTS

    fclose(f);

    // Allocate runtime buffers
    int max_buf_seq = 512;  // max sequence we'll process at once
    m->x = (float*)calloc(max_buf_seq * dim, sizeof(float));
    m->x_buf = (float*)calloc(max_buf_seq * dim, sizeof(float));
    m->qkv = (float*)calloc(max_buf_seq * 3 * dim, sizeof(float));
    m->attn_out = (float*)calloc(max_buf_seq * dim, sizeof(float));
    m->ffn_buf = (float*)calloc(max_buf_seq * hidden_dim, sizeof(float));
    m->logits = (float*)calloc(vocab_size, sizeof(float));

    // KV cache
    m->key_cache = (float*)calloc(n_layers * max_seq * dim, sizeof(float));
    m->value_cache = (float*)calloc(n_layers * max_seq * dim, sizeof(float));
    m->cache_len = 0;

    fprintf(stderr, "[brain] model loaded successfully\n");
    return 0;
}

int brain_load_tokenizer(ExternalBrain* brain, const char* vocab_path, const char* merges_path) {
    if (parse_vocab_json(&brain->tokenizer, vocab_path) < 0) return -1;
    if (parse_merges(&brain->tokenizer, merges_path) < 0) return -1;
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// FORWARD PASS
// ═══════════════════════════════════════════════════════════════════════════════

void brain_forward(ExternalBrain* brain, int* tokens, int n_tokens, int start_pos) {
    GPT2_Model* m = &brain->model;
    int dim = m->dim;
    int n_heads = m->n_heads;
    int head_dim = dim / n_heads;
    int hidden_dim = m->hidden_dim;

    // Get embeddings: x = wte[tokens] + wpe[positions]
    for (int t = 0; t < n_tokens; t++) {
        int pos = start_pos + t;
        int token = tokens[t];
        float* x_t = m->x + t * dim;

        for (int i = 0; i < dim; i++) {
            x_t[i] = m->wte[token * dim + i] + m->wpe[pos * dim + i];
        }
    }

    // Process each layer
    for (int l = 0; l < m->n_layers; l++) {
        GPT2_Layer* layer = &m->layers[l];

        // For each position
        for (int t = 0; t < n_tokens; t++) {
            float* x_t = m->x + t * dim;
            float* x_buf_t = m->x_buf + t * dim;
            float* qkv_t = m->qkv + t * 3 * dim;
            float* attn_t = m->attn_out + t * dim;
            float* ffn_t = m->ffn_buf + t * hidden_dim;

            int pos = start_pos + t;

            // LayerNorm 1
            layer_norm(x_buf_t, x_t, layer->ln1_weight, layer->ln1_bias, dim);

            // QKV projection
            matmul_add_bias(qkv_t, x_buf_t, layer->c_attn_weight, layer->c_attn_bias, dim, 3 * dim);

            float* q = qkv_t;
            float* k = qkv_t + dim;
            float* v = qkv_t + 2 * dim;

            // Store K, V in cache
            float* k_cache = m->key_cache + l * m->max_seq * dim + pos * dim;
            float* v_cache = m->value_cache + l * m->max_seq * dim + pos * dim;
            memcpy(k_cache, k, dim * sizeof(float));
            memcpy(v_cache, v, dim * sizeof(float));

            // Multi-head attention
            memset(attn_t, 0, dim * sizeof(float));
            float scale = 1.0f / sqrtf((float)head_dim);

            for (int h = 0; h < n_heads; h++) {
                float* q_h = q + h * head_dim;
                float* out_h = attn_t + h * head_dim;

                // Compute attention scores for all cached positions
                float scores[GPT2_MAX_SEQ];
                int n_ctx = pos + 1;  // include current position

                for (int j = 0; j < n_ctx; j++) {
                    float* k_j = m->key_cache + l * m->max_seq * dim + j * dim + h * head_dim;
                    float dot = 0.0f;
                    for (int i = 0; i < head_dim; i++) {
                        dot += q_h[i] * k_j[i];
                    }
                    scores[j] = dot * scale;
                }

                // Softmax
                softmax(scores, n_ctx);

                // Weighted sum of values
                for (int j = 0; j < n_ctx; j++) {
                    float* v_j = m->value_cache + l * m->max_seq * dim + j * dim + h * head_dim;
                    for (int i = 0; i < head_dim; i++) {
                        out_h[i] += scores[j] * v_j[i];
                    }
                }
            }

            // Output projection
            float proj_out[GPT2_DIM];
            matmul_add_bias(proj_out, attn_t, layer->c_proj_weight, layer->c_proj_bias, dim, dim);

            // Residual
            for (int i = 0; i < dim; i++) {
                x_t[i] += proj_out[i];
            }

            // LayerNorm 2
            layer_norm(x_buf_t, x_t, layer->ln2_weight, layer->ln2_bias, dim);

            // FFN: fc -> gelu -> proj
            matmul_add_bias(ffn_t, x_buf_t, layer->c_fc_weight, layer->c_fc_bias, dim, hidden_dim);
            for (int i = 0; i < hidden_dim; i++) {
                ffn_t[i] = gelu(ffn_t[i]);
            }

            float ffn_out[GPT2_DIM];
            matmul_add_bias(ffn_out, ffn_t, layer->c_proj2_weight, layer->c_proj2_bias, hidden_dim, dim);

            // Residual
            for (int i = 0; i < dim; i++) {
                x_t[i] += ffn_out[i];
            }
        }
    }

    // Final LayerNorm (only for last position)
    int last_t = n_tokens - 1;
    float* x_last = m->x + last_t * dim;
    float x_norm[GPT2_DIM];
    layer_norm(x_norm, x_last, m->ln_f_weight, m->ln_f_bias, dim);

    // Logits: x @ wte.T (weight tying)
    for (int v = 0; v < m->vocab_size; v++) {
        float dot = 0.0f;
        for (int i = 0; i < dim; i++) {
            dot += x_norm[i] * m->wte[v * dim + i];
        }
        m->logits[v] = dot;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SAMPLING
// ═══════════════════════════════════════════════════════════════════════════════

// Simple random (should use better RNG in production)
static float random_float(void) {
    return (float)rand() / (float)RAND_MAX;
}

int brain_sample(ExternalBrain* brain, float temperature, int top_k, float top_p) {
    GPT2_Model* m = &brain->model;
    float* logits = m->logits;
    int vocab_size = m->vocab_size;

    // Apply temperature
    if (temperature > 0.0f && temperature != 1.0f) {
        for (int i = 0; i < vocab_size; i++) {
            logits[i] /= temperature;
        }
    }

    // Apply repetition penalty to recent tokens
    if (brain->n_tokens > 0 && brain->repetition_penalty > 1.0f) {
        int lookback = brain->n_tokens < 32 ? brain->n_tokens : 32;
        for (int i = 0; i < lookback; i++) {
            int tok = brain->tokens[brain->n_tokens - 1 - i];
            if (logits[tok] > 0) {
                logits[tok] /= brain->repetition_penalty;
            } else {
                logits[tok] *= brain->repetition_penalty;
            }
        }
    }

    // Top-k filtering
    if (top_k > 0 && top_k < vocab_size) {
        // Find k-th largest value
        float threshold = -1e30f;
        for (int iter = 0; iter < top_k; iter++) {
            float max_val = -1e30f;
            for (int i = 0; i < vocab_size; i++) {
                if (logits[i] > max_val && logits[i] > threshold) {
                    // Check if we've seen a higher value this pass
                    int seen = 0;
                    for (int j = 0; j < i; j++) {
                        if (logits[j] > logits[i] && logits[j] <= threshold) {
                            seen = 1;
                            break;
                        }
                    }
                    if (!seen) max_val = logits[i];
                }
            }
            if (max_val > -1e30f) threshold = max_val;
        }

        // Zero out below threshold
        for (int i = 0; i < vocab_size; i++) {
            if (logits[i] < threshold) logits[i] = -1e30f;
        }
    }

    // Softmax
    softmax(logits, vocab_size);

    // Top-p (nucleus) sampling
    if (top_p > 0.0f && top_p < 1.0f) {
        // Sort indices by probability (descending)
        int* indices = (int*)malloc(vocab_size * sizeof(int));
        for (int i = 0; i < vocab_size; i++) indices[i] = i;

        // Simple insertion sort (good enough for top-p)
        for (int i = 1; i < vocab_size; i++) {
            int j = i;
            while (j > 0 && logits[indices[j]] > logits[indices[j-1]]) {
                int tmp = indices[j];
                indices[j] = indices[j-1];
                indices[j-1] = tmp;
                j--;
            }
        }

        // Find cutoff
        float cumsum = 0.0f;
        int cutoff = vocab_size;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += logits[indices[i]];
            if (cumsum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }

        // Zero out below cutoff
        for (int i = cutoff; i < vocab_size; i++) {
            logits[indices[i]] = 0.0f;
        }

        // Renormalize
        float sum = 0.0f;
        for (int i = 0; i < vocab_size; i++) sum += logits[i];
        if (sum > 0.0f) {
            for (int i = 0; i < vocab_size; i++) logits[i] /= sum;
        }

        free(indices);
    }

    // Sample from distribution
    float r = random_float();
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += logits[i];
        if (cumsum >= r) {
            return i;
        }
    }

    return vocab_size - 1;  // fallback
}

// ═══════════════════════════════════════════════════════════════════════════════
// GENERATION
// ═══════════════════════════════════════════════════════════════════════════════

int brain_generate_draft(ExternalBrain* brain, const char* prompt,
                         char* output, int max_output_len,
                         int max_new_tokens) {
    if (!brain->loaded) {
        strcpy(output, "[brain not loaded]");
        return -1;
    }

    // Encode prompt
    int prompt_tokens[512];
    int n_prompt = brain_encode(brain, prompt, prompt_tokens, 512);

    if (n_prompt == 0) {
        strcpy(output, "[empty prompt]");
        return -1;
    }

    // Initialize token buffer
    brain->n_tokens = n_prompt;
    memcpy(brain->tokens, prompt_tokens, n_prompt * sizeof(int));

    // Reset KV cache
    brain->model.cache_len = 0;

    // Process prompt
    brain_forward(brain, brain->tokens, brain->n_tokens, 0);
    brain->model.cache_len = brain->n_tokens;

    // Generate new tokens
    int generated_tokens[512];
    int n_generated = 0;

    for (int i = 0; i < max_new_tokens && brain->n_tokens < brain->max_tokens; i++) {
        // Sample next token
        int next_token = brain_sample(brain, brain->temperature,
                                      brain->top_k, brain->top_p);

        // Check for EOS
        if (next_token == brain->tokenizer.eos_token_id) {
            break;
        }

        // Add to sequence
        brain->tokens[brain->n_tokens] = next_token;
        generated_tokens[n_generated++] = next_token;
        brain->n_tokens++;

        // Forward pass for next position
        brain_forward(brain, &next_token, 1, brain->n_tokens - 1);
        brain->model.cache_len = brain->n_tokens;
    }

    // Decode generated tokens
    brain_decode(brain, generated_tokens, n_generated, output, max_output_len);

    return n_generated;
}

// ═══════════════════════════════════════════════════════════════════════════════
// INIT/FREE
// ═══════════════════════════════════════════════════════════════════════════════

int brain_init(ExternalBrain* brain) {
    memset(brain, 0, sizeof(ExternalBrain));

    brain->max_tokens = GPT2_MAX_SEQ;
    brain->tokens = (int*)calloc(brain->max_tokens, sizeof(int));

    // Default generation config
    brain->temperature = 0.8f;
    brain->top_p = 0.95f;
    brain->top_k = 50;
    brain->repetition_penalty = 1.2f;

    brain->loaded = 0;
    return 0;
}

void brain_free(ExternalBrain* brain) {
    GPT2_Model* m = &brain->model;
    BPE_Tokenizer* tok = &brain->tokenizer;

    // Free model weights
    free(m->wte);
    free(m->wpe);
    free(m->ln_f_weight);
    free(m->ln_f_bias);

    for (int l = 0; l < m->n_layers; l++) {
        GPT2_Layer* layer = &m->layers[l];
        free(layer->ln1_weight);
        free(layer->ln1_bias);
        free(layer->c_attn_weight);
        free(layer->c_attn_bias);
        free(layer->c_proj_weight);
        free(layer->c_proj_bias);
        free(layer->ln2_weight);
        free(layer->ln2_bias);
        free(layer->c_fc_weight);
        free(layer->c_fc_bias);
        free(layer->c_proj2_weight);
        free(layer->c_proj2_bias);
    }
    free(m->layers);

    // Free runtime buffers
    free(m->x);
    free(m->x_buf);
    free(m->qkv);
    free(m->attn_out);
    free(m->ffn_buf);
    free(m->logits);
    free(m->key_cache);
    free(m->value_cache);

    // Free tokenizer
    for (int i = 0; i < tok->vocab_size; i++) {
        free(tok->vocab[i]);
    }
    free(tok->vocab);
    free(tok->word_keys);
    free(tok->word_ids);

    for (int i = 0; i < tok->n_merges; i++) {
        free(tok->merge_a[i]);
        free(tok->merge_b[i]);
    }
    free(tok->merge_a);
    free(tok->merge_b);

    free(brain->tokens);

    memset(brain, 0, sizeof(ExternalBrain));
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONVENIENCE LOADER
// ═══════════════════════════════════════════════════════════════════════════════

int brain_load(ExternalBrain* brain, const char* weights_dir) {
    char path[512];

    // Load model weights
    snprintf(path, sizeof(path), "%s/gpt2_30m.bin", weights_dir);
    if (brain_load_model(brain, path) < 0) return -1;

    // Load tokenizer
    char vocab_path[512], merges_path[512];
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", weights_dir);
    snprintf(merges_path, sizeof(merges_path), "%s/merges.txt", weights_dir);
    if (brain_load_tokenizer(brain, vocab_path, merges_path) < 0) return -1;

    brain->loaded = 1;
    fprintf(stderr, "[brain] fully loaded and ready\n");
    return 0;
}
