// gpt2_30m.c — Minimal GPT2-30M inference for Pandora
// "External brain for vocabulary extraction"
//
// ═══════════════════════════════════════════════════════════════════════════════

#include "gpt2_30m.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// ═══════════════════════════════════════════════════════════════════════════════
// MATH HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

static void layer_norm(float* out, const float* x, const float* g, const float* b, int n) {
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;

    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= n;

    float std = sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++) {
        out[i] = g[i] * (x[i] - mean) / std + b[i];
    }
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

static float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static void matmul(float* out, const float* a, const float* b, int m, int n, int k) {
    // out[m,k] = a[m,n] @ b[n,k]
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            for (int l = 0; l < n; l++) {
                sum += a[i * n + l] * b[l * k + j];
            }
            out[i * k + j] = sum;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MODEL INIT
// ═══════════════════════════════════════════════════════════════════════════════

int gpt2_30m_init(GPT2_30M* model) {
    memset(model, 0, sizeof(GPT2_30M));

    int d = GPT2_30M_EMBED_DIM;
    int v = GPT2_30M_VOCAB_SIZE;
    int c = GPT2_30M_CONTEXT_LEN;

    // Allocate embeddings
    model->wte = calloc(v * d, sizeof(float));
    model->wpe = calloc(c * d, sizeof(float));

    if (!model->wte || !model->wpe) {
        fprintf(stderr, "[gpt2_30m] failed to allocate embeddings\n");
        return -1;
    }

    // Allocate blocks
    for (int l = 0; l < GPT2_30M_N_LAYERS; l++) {
        model->blocks[l].ln1_g = calloc(d, sizeof(float));
        model->blocks[l].ln1_b = calloc(d, sizeof(float));
        model->blocks[l].attn_qkv = calloc(3 * d * d, sizeof(float));
        model->blocks[l].attn_proj = calloc(d * d, sizeof(float));
        model->blocks[l].ln2_g = calloc(d, sizeof(float));
        model->blocks[l].ln2_b = calloc(d, sizeof(float));
        model->blocks[l].mlp_fc = calloc(4 * d * d, sizeof(float));
        model->blocks[l].mlp_proj = calloc(d * 4 * d, sizeof(float));

        // Initialize layer norm to ones
        for (int i = 0; i < d; i++) {
            model->blocks[l].ln1_g[i] = 1.0f;
            model->blocks[l].ln2_g[i] = 1.0f;
        }
    }

    // Final layer norm
    model->ln_f_g = calloc(d, sizeof(float));
    model->ln_f_b = calloc(d, sizeof(float));
    for (int i = 0; i < d; i++) model->ln_f_g[i] = 1.0f;

    // Scratch buffers
    model->scratch1 = calloc(c * d, sizeof(float));
    model->scratch2 = calloc(c * d, sizeof(float));
    model->qkv_buf = calloc(c * 3 * d, sizeof(float));
    model->attn_buf = calloc(GPT2_30M_N_HEADS * c * c, sizeof(float));

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// LOAD WEIGHTS
// ═══════════════════════════════════════════════════════════════════════════════

int gpt2_30m_load(GPT2_30M* model, const char* path) {
    if (!model->wte) {
        if (gpt2_30m_init(model) != 0) return -1;
    }

    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[gpt2_30m] cannot open %s\n", path);
        return -1;
    }

    int d = GPT2_30M_EMBED_DIM;
    int v = GPT2_30M_VOCAB_SIZE;
    int c = GPT2_30M_CONTEXT_LEN;

    // Read header (magic + dims)
    int header[8];
    if (fread(header, sizeof(int), 8, f) != 8) {
        fprintf(stderr, "[gpt2_30m] failed to read header\n");
        fclose(f);
        return -1;
    }

    // Verify magic
    if (header[0] != 0x47503230) {  // 'GP20'
        fprintf(stderr, "[gpt2_30m] invalid magic: expected GP20\n");
        fclose(f);
        return -1;
    }

    // Read embeddings
    fread(model->wte, sizeof(float), v * d, f);
    fread(model->wpe, sizeof(float), c * d, f);

    // Read blocks
    for (int l = 0; l < GPT2_30M_N_LAYERS; l++) {
        fread(model->blocks[l].ln1_g, sizeof(float), d, f);
        fread(model->blocks[l].ln1_b, sizeof(float), d, f);
        fread(model->blocks[l].attn_qkv, sizeof(float), 3 * d * d, f);
        fread(model->blocks[l].attn_proj, sizeof(float), d * d, f);
        fread(model->blocks[l].ln2_g, sizeof(float), d, f);
        fread(model->blocks[l].ln2_b, sizeof(float), d, f);
        fread(model->blocks[l].mlp_fc, sizeof(float), 4 * d * d, f);
        fread(model->blocks[l].mlp_proj, sizeof(float), d * 4 * d, f);
    }

    // Read final layer norm
    fread(model->ln_f_g, sizeof(float), d, f);
    fread(model->ln_f_b, sizeof(float), d, f);

    fclose(f);
    model->loaded = 1;
    fprintf(stderr, "[gpt2_30m] loaded weights from %s\n", path);
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// FREE
// ═══════════════════════════════════════════════════════════════════════════════

void gpt2_30m_free(GPT2_30M* model) {
    free(model->wte);
    free(model->wpe);

    for (int l = 0; l < GPT2_30M_N_LAYERS; l++) {
        free(model->blocks[l].ln1_g);
        free(model->blocks[l].ln1_b);
        free(model->blocks[l].attn_qkv);
        free(model->blocks[l].attn_proj);
        free(model->blocks[l].ln2_g);
        free(model->blocks[l].ln2_b);
        free(model->blocks[l].mlp_fc);
        free(model->blocks[l].mlp_proj);
    }

    free(model->ln_f_g);
    free(model->ln_f_b);
    free(model->scratch1);
    free(model->scratch2);
    free(model->qkv_buf);
    free(model->attn_buf);

    memset(model, 0, sizeof(GPT2_30M));
}

// ═══════════════════════════════════════════════════════════════════════════════
// FORWARD PASS
// ═══════════════════════════════════════════════════════════════════════════════

float* gpt2_30m_forward(GPT2_30M* model, const int* tokens, int n_tokens) {
    if (!model->loaded || n_tokens <= 0) return NULL;
    if (n_tokens > GPT2_30M_CONTEXT_LEN) n_tokens = GPT2_30M_CONTEXT_LEN;

    int d = GPT2_30M_EMBED_DIM;
    int T = n_tokens;
    int nh = GPT2_30M_N_HEADS;
    int hd = d / nh;

    float* x = model->scratch1;  // [T, d]

    // Embed tokens + positions
    for (int t = 0; t < T; t++) {
        int tok = tokens[t];
        if (tok < 0 || tok >= GPT2_30M_VOCAB_SIZE) tok = 0;

        for (int i = 0; i < d; i++) {
            x[t * d + i] = model->wte[tok * d + i] + model->wpe[t * d + i];
        }
    }

    float* y = model->scratch2;

    // Process blocks
    for (int l = 0; l < GPT2_30M_N_LAYERS; l++) {
        // Layer norm 1
        for (int t = 0; t < T; t++) {
            layer_norm(&y[t * d], &x[t * d],
                      model->blocks[l].ln1_g, model->blocks[l].ln1_b, d);
        }

        // Compute QKV
        float* qkv = model->qkv_buf;
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < 3 * d; i++) {
                float sum = 0.0f;
                for (int j = 0; j < d; j++) {
                    sum += y[t * d + j] * model->blocks[l].attn_qkv[j * 3 * d + i];
                }
                qkv[t * 3 * d + i] = sum;
            }
        }

        // Multi-head attention
        float* attn = model->attn_buf;

        for (int h = 0; h < nh; h++) {
            // Compute attention scores
            for (int t1 = 0; t1 < T; t1++) {
                for (int t2 = 0; t2 <= t1; t2++) {  // causal mask
                    float score = 0.0f;
                    for (int i = 0; i < hd; i++) {
                        float q = qkv[t1 * 3 * d + h * hd + i];
                        float k = qkv[t2 * 3 * d + d + h * hd + i];
                        score += q * k;
                    }
                    attn[h * T * T + t1 * T + t2] = score / sqrtf((float)hd);
                }
                // Mask future
                for (int t2 = t1 + 1; t2 < T; t2++) {
                    attn[h * T * T + t1 * T + t2] = -1e9f;
                }

                // Softmax
                softmax(&attn[h * T * T + t1 * T], T);
            }
        }

        // Apply attention to values
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < nh; h++) {
                for (int i = 0; i < hd; i++) {
                    float sum = 0.0f;
                    for (int t2 = 0; t2 < T; t2++) {
                        float v = qkv[t2 * 3 * d + 2 * d + h * hd + i];
                        sum += attn[h * T * T + t * T + t2] * v;
                    }
                    y[t * d + h * hd + i] = sum;
                }
            }
        }

        // Project attention output
        for (int t = 0; t < T; t++) {
            float tmp[GPT2_30M_EMBED_DIM];
            for (int i = 0; i < d; i++) {
                float sum = 0.0f;
                for (int j = 0; j < d; j++) {
                    sum += y[t * d + j] * model->blocks[l].attn_proj[j * d + i];
                }
                tmp[i] = sum;
            }
            // Residual
            for (int i = 0; i < d; i++) {
                x[t * d + i] += tmp[i];
            }
        }

        // Layer norm 2
        for (int t = 0; t < T; t++) {
            layer_norm(&y[t * d], &x[t * d],
                      model->blocks[l].ln2_g, model->blocks[l].ln2_b, d);
        }

        // MLP
        for (int t = 0; t < T; t++) {
            float hidden[4 * GPT2_30M_EMBED_DIM];

            // FC
            for (int i = 0; i < 4 * d; i++) {
                float sum = 0.0f;
                for (int j = 0; j < d; j++) {
                    sum += y[t * d + j] * model->blocks[l].mlp_fc[j * 4 * d + i];
                }
                hidden[i] = gelu(sum);
            }

            // Proj
            for (int i = 0; i < d; i++) {
                float sum = 0.0f;
                for (int j = 0; j < 4 * d; j++) {
                    sum += hidden[j] * model->blocks[l].mlp_proj[j * d + i];
                }
                x[t * d + i] += sum;  // Residual
            }
        }
    }

    // Final layer norm (only for last token)
    layer_norm(&x[(T-1) * d], &x[(T-1) * d], model->ln_f_g, model->ln_f_b, d);

    // Output logits (project to vocab)
    static float logits[GPT2_30M_VOCAB_SIZE];
    for (int v = 0; v < GPT2_30M_VOCAB_SIZE; v++) {
        float sum = 0.0f;
        for (int i = 0; i < d; i++) {
            sum += x[(T-1) * d + i] * model->wte[v * d + i];
        }
        logits[v] = sum;
    }

    return logits;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SAMPLING
// ═══════════════════════════════════════════════════════════════════════════════

int gpt2_30m_sample(const float* logits, int vocab_size, float temperature) {
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }

    // SECURITY: prevent division by zero
    if (temperature <= 0.0f) temperature = 1.0f;

    // Copy and apply temperature
    float probs[GPT2_30M_VOCAB_SIZE];
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = logits[i] / temperature;
    }

    // Softmax
    softmax(probs, vocab_size);

    // Sample
    float r = (float)rand() / RAND_MAX;
    float cum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cum += probs[i];
        if (r < cum) return i;
    }

    return vocab_size - 1;
}

// ═══════════════════════════════════════════════════════════════════════════════
// GENERATION
// ═══════════════════════════════════════════════════════════════════════════════

int gpt2_30m_generate(
    GPT2_30M* model,
    const int* prompt_tokens,
    int n_prompt,
    int* output_tokens,
    int max_tokens,
    float temperature
) {
    if (!model->loaded) return 0;

    // Copy prompt
    int tokens[GPT2_30M_CONTEXT_LEN];
    int n = n_prompt;
    if (n > GPT2_30M_CONTEXT_LEN) n = GPT2_30M_CONTEXT_LEN;
    memcpy(tokens, prompt_tokens, n * sizeof(int));

    // Generate
    int generated = 0;
    while (generated < max_tokens && n < GPT2_30M_CONTEXT_LEN) {
        float* logits = gpt2_30m_forward(model, tokens, n);
        if (!logits) break;

        int next = gpt2_30m_sample(logits, GPT2_30M_VOCAB_SIZE, temperature);
        tokens[n++] = next;
        output_tokens[generated++] = next;

        // Stop on EOS
        if (next == 50256) break;
    }

    return generated;
}

// ═══════════════════════════════════════════════════════════════════════════════
// VOCAB
// ═══════════════════════════════════════════════════════════════════════════════

int gpt2_vocab_load(GPT2Vocab* vocab, const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return -1;

    vocab->id_to_token = calloc(GPT2_30M_VOCAB_SIZE, sizeof(char*));
    if (!vocab->id_to_token) {
        fclose(f);
        return -1;
    }

    // Simple JSON-like parsing for vocab.json
    // Format: {"token": id, ...}
    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        // Parse "token": id pairs
        char* p = line;
        while (*p) {
            if (*p == '"') {
                char* start = ++p;
                while (*p && *p != '"') p++;
                if (!*p) break;

                int len = p - start;
                char* token = malloc(len + 1);
                memcpy(token, start, len);
                token[len] = '\0';

                // Find id
                p++;
                while (*p && (*p == ':' || *p == ' ')) p++;
                int id = atoi(p);

                if (id >= 0 && id < GPT2_30M_VOCAB_SIZE) {
                    vocab->id_to_token[id] = token;
                } else {
                    free(token);
                }

                // Skip to next
                while (*p && *p != ',') p++;
            } else {
                p++;
            }
        }
    }

    fclose(f);
    vocab->loaded = 1;
    return 0;
}

void gpt2_vocab_free(GPT2Vocab* vocab) {
    if (vocab->id_to_token) {
        for (int i = 0; i < GPT2_30M_VOCAB_SIZE; i++) {
            free(vocab->id_to_token[i]);
        }
        free(vocab->id_to_token);
    }
    memset(vocab, 0, sizeof(GPT2Vocab));
}

const char* gpt2_vocab_decode(GPT2Vocab* vocab, int token_id) {
    if (!vocab->loaded || token_id < 0 || token_id >= GPT2_30M_VOCAB_SIZE) {
        return NULL;
    }
    return vocab->id_to_token[token_id];
}
