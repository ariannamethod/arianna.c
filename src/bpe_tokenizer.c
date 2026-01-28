/*
 * bpe_tokenizer.c — BPE Tokenizer for Arianna
 *
 * SentencePiece-style BPE tokenization in pure C.
 * Uses greedy longest match for encoding.
 */

#include "bpe_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// SentencePiece uses ▁ (U+2581) for word boundary
// In UTF-8: E2 96 81
static const char SPIECE_UNDERLINE[] = "\xE2\x96\x81";
static const int SPIECE_UNDERLINE_LEN = 3;

// Static decode buffer
static char decode_buffer[BPE_MAX_TOKENS * BPE_MAX_PIECE_LEN];

// Simple JSON string extraction (finds "key": "value" or "key": number)
static int find_json_string(const char* json, const char* key, char* out, int max_len) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);

    const char* pos = strstr(json, pattern);
    if (!pos) return 0;

    pos += strlen(pattern);
    while (*pos && (*pos == ' ' || *pos == '\t')) pos++;

    if (*pos == '"') {
        pos++;
        int i = 0;
        while (*pos && *pos != '"' && i < max_len - 1) {
            if (*pos == '\\' && *(pos+1)) {
                pos++;
                switch (*pos) {
                    case 'n': out[i++] = '\n'; break;
                    case 't': out[i++] = '\t'; break;
                    case 'r': out[i++] = '\r'; break;
                    default: out[i++] = *pos; break;
                }
            } else {
                out[i++] = *pos;
            }
            pos++;
        }
        out[i] = '\0';
        return 1;
    }
    return 0;
}

int bpe_load(BPETokenizer* tok, const char* json_path) {
    memset(tok, 0, sizeof(BPETokenizer));

    FILE* f = fopen(json_path, "r");
    if (!f) {
        fprintf(stderr, "[bpe] Failed to open: %s\n", json_path);
        return -1;
    }

    // Get file size
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size <= 0 || size > 10*1024*1024) {
        fprintf(stderr, "[bpe] Invalid file size: %ld\n", size);
        fclose(f);
        return -1;
    }

    char* json = (char*)malloc(size + 1);
    if (!json) {
        fclose(f);
        return -1;
    }

    size_t read = fread(json, 1, size, f);
    fclose(f);
    json[read] = '\0';

    // Parse id_to_piece section
    const char* id_section = strstr(json, "\"id_to_piece\"");
    if (!id_section) {
        fprintf(stderr, "[bpe] No id_to_piece section found\n");
        free(json);
        return -1;
    }

    // Find opening brace
    const char* brace = strchr(id_section, '{');
    if (!brace) {
        free(json);
        return -1;
    }

    // Parse entries: "id": "piece"
    const char* pos = brace + 1;
    int max_id = 0;

    while (*pos && *pos != '}') {
        // Skip whitespace
        while (*pos && (*pos == ' ' || *pos == '\t' || *pos == '\n' || *pos == '\r' || *pos == ',')) pos++;

        if (*pos == '}') break;
        if (*pos != '"') { pos++; continue; }

        // Parse ID
        pos++;
        int id = 0;
        while (*pos && *pos != '"') {
            if (isdigit(*pos)) id = id * 10 + (*pos - '0');
            pos++;
        }
        if (*pos == '"') pos++;

        // Skip colon
        while (*pos && *pos != ':') pos++;
        if (*pos == ':') pos++;
        while (*pos && (*pos == ' ' || *pos == '\t')) pos++;

        // Parse piece
        if (*pos == '"') {
            pos++;
            int i = 0;
            while (*pos && *pos != '"' && i < BPE_MAX_PIECE_LEN - 1) {
                if (*pos == '\\' && *(pos+1)) {
                    pos++;
                    switch (*pos) {
                        case 'n': tok->pieces[id][i++] = '\n'; break;
                        case 't': tok->pieces[id][i++] = '\t'; break;
                        case 'r': tok->pieces[id][i++] = '\r'; break;
                        case 'u': {
                            // Unicode escape \uXXXX
                            if (pos[1] && pos[2] && pos[3] && pos[4]) {
                                char hex[5] = {pos[1], pos[2], pos[3], pos[4], 0};
                                int code = (int)strtol(hex, NULL, 16);
                                pos += 4;
                                // Convert to UTF-8
                                if (code < 0x80) {
                                    tok->pieces[id][i++] = code;
                                } else if (code < 0x800) {
                                    tok->pieces[id][i++] = 0xC0 | (code >> 6);
                                    tok->pieces[id][i++] = 0x80 | (code & 0x3F);
                                } else {
                                    tok->pieces[id][i++] = 0xE0 | (code >> 12);
                                    tok->pieces[id][i++] = 0x80 | ((code >> 6) & 0x3F);
                                    tok->pieces[id][i++] = 0x80 | (code & 0x3F);
                                }
                            }
                            break;
                        }
                        default: tok->pieces[id][i++] = *pos; break;
                    }
                } else {
                    tok->pieces[id][i++] = *pos;
                }
                pos++;
            }
            tok->pieces[id][i] = '\0';
            if (*pos == '"') pos++;

            if (id > max_id) max_id = id;
        }
    }

    tok->vocab_size = max_id + 1;
    tok->loaded = 1;

    free(json);

    printf("[bpe] Loaded %d tokens from %s\n", tok->vocab_size, json_path);
    return 0;
}

int bpe_encode(const BPETokenizer* tok, const char* text, int* ids, int max_tokens) {
    if (!tok->loaded || !text || !ids) return 0;

    int n = 0;
    int len = strlen(text);

    // Prepend space marker for word boundary at start
    int at_word_start = 1;

    for (int i = 0; i < len && n < max_tokens; ) {
        int best_len = 0;
        int best_id = BPE_UNK_ID;

        // Build candidate with possible word boundary marker
        char candidate[BPE_MAX_PIECE_LEN * 2];
        int cand_offset = 0;

        if (at_word_start) {
            memcpy(candidate, SPIECE_UNDERLINE, SPIECE_UNDERLINE_LEN);
            cand_offset = SPIECE_UNDERLINE_LEN;
        }

        // Try greedy longest match
        for (int try_len = 1; try_len <= len - i && try_len < BPE_MAX_PIECE_LEN - cand_offset; try_len++) {
            memcpy(candidate + cand_offset, text + i, try_len);
            candidate[cand_offset + try_len] = '\0';

            // Search in vocab
            for (int v = 0; v < tok->vocab_size; v++) {
                if (strcmp(tok->pieces[v], candidate) == 0) {
                    if (cand_offset + try_len > best_len) {
                        best_len = cand_offset + try_len;
                        best_id = v;
                    }
                }
            }
        }

        // If no match with word boundary, try without
        if (best_len == 0 && at_word_start) {
            for (int try_len = 1; try_len <= len - i && try_len < BPE_MAX_PIECE_LEN; try_len++) {
                memcpy(candidate, text + i, try_len);
                candidate[try_len] = '\0';

                for (int v = 0; v < tok->vocab_size; v++) {
                    if (strcmp(tok->pieces[v], candidate) == 0) {
                        if (try_len > best_len) {
                            best_len = try_len;
                            best_id = v;
                        }
                    }
                }
            }
        }

        // Fallback to byte token
        if (best_len == 0) {
            unsigned char byte = (unsigned char)text[i];
            char byte_token[10];
            snprintf(byte_token, sizeof(byte_token), "<0x%02X>", byte);

            for (int v = 0; v < tok->vocab_size; v++) {
                if (strcmp(tok->pieces[v], byte_token) == 0) {
                    best_id = v;
                    break;
                }
            }
            best_len = 1;
            i++;
        } else {
            // Advance past matched text (excluding the ▁ prefix which is just a marker)
            int text_consumed = best_len;
            if (at_word_start && best_len > SPIECE_UNDERLINE_LEN) {
                text_consumed = best_len - SPIECE_UNDERLINE_LEN;
            }
            i += text_consumed;
        }

        ids[n++] = best_id;

        // Check if next char is space (word boundary)
        if (i < len && text[i] == ' ') {
            i++;  // Skip space
            at_word_start = 1;
        } else {
            at_word_start = 0;
        }
    }

    return n;
}

const char* bpe_decode(const BPETokenizer* tok, const int* ids, int n_tokens) {
    if (!tok->loaded || !ids) return "";

    decode_buffer[0] = '\0';
    int pos = 0;

    for (int i = 0; i < n_tokens && pos < sizeof(decode_buffer) - BPE_MAX_PIECE_LEN; i++) {
        int id = ids[i];
        if (id < 0 || id >= tok->vocab_size) continue;

        const char* piece = tok->pieces[id];

        // Skip special tokens
        if (id < 4) continue;  // <pad>, <unk>, <s>, </s>
        if (piece[0] == '<' && piece[1] == '0' && piece[2] == 'x') {
            // Byte token: <0xNN>
            int byte_val = 0;
            sscanf(piece + 3, "%02X", &byte_val);
            decode_buffer[pos++] = (char)byte_val;
            continue;
        }

        // Handle ▁ (word boundary = space)
        const char* src = piece;
        if (strncmp(src, SPIECE_UNDERLINE, SPIECE_UNDERLINE_LEN) == 0) {
            if (pos > 0) {  // Don't add space at start
                decode_buffer[pos++] = ' ';
            }
            src += SPIECE_UNDERLINE_LEN;
        }

        // Copy rest of piece
        while (*src && pos < sizeof(decode_buffer) - 1) {
            decode_buffer[pos++] = *src++;
        }
    }

    decode_buffer[pos] = '\0';
    return decode_buffer;
}

const char* bpe_get_piece(const BPETokenizer* tok, int id) {
    if (!tok->loaded || id < 0 || id >= tok->vocab_size) return "<unk>";
    return tok->pieces[id];
}

int bpe_get_id(const BPETokenizer* tok, const char* piece) {
    if (!tok->loaded || !piece) return BPE_UNK_ID;

    for (int i = 0; i < tok->vocab_size; i++) {
        if (strcmp(tok->pieces[i], piece) == 0) return i;
    }
    return BPE_UNK_ID;
}

void bpe_free(BPETokenizer* tok) {
    memset(tok, 0, sizeof(BPETokenizer));
}

// ============================================================
// TEST
// ============================================================

#ifdef BPE_TEST
int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "weights/tokenizer_bpe.json";

    BPETokenizer tok;
    if (bpe_load(&tok, path) != 0) {
        return 1;
    }

    printf("\nTest encoding:\n");
    const char* test_texts[] = {
        "I am Arianna.",
        "What is consciousness?",
        "Hello world!",
        "Q: Who are you?\nA:",
    };

    for (int t = 0; t < 4; t++) {
        int ids[256];
        int n = bpe_encode(&tok, test_texts[t], ids, 256);

        printf("\n\"%s\"\n  -> %d tokens: ", test_texts[t], n);
        for (int i = 0; i < n && i < 20; i++) {
            printf("%d ", ids[i]);
        }
        if (n > 20) printf("...");
        printf("\n");

        const char* decoded = bpe_decode(&tok, ids, n);
        printf("  -> decoded: \"%s\"\n", decoded);
    }

    bpe_free(&tok);
    return 0;
}
#endif
