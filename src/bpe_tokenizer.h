/*
 * bpe_tokenizer.h — BPE Tokenizer for Arianna
 *
 * Handles SentencePiece-style BPE tokenization.
 * Used by: Arianna 36M BPE, d12, d20
 */

#ifndef BPE_TOKENIZER_H
#define BPE_TOKENIZER_H

#include <stdint.h>

#define BPE_MAX_VOCAB 8192
#define BPE_MAX_PIECE_LEN 64
#define BPE_MAX_TOKENS 4096

// Special tokens
#define BPE_PAD_ID 0
#define BPE_UNK_ID 1
#define BPE_BOS_ID 2
#define BPE_EOS_ID 3

typedef struct {
    char pieces[BPE_MAX_VOCAB][BPE_MAX_PIECE_LEN];
    int vocab_size;
    int loaded;
} BPETokenizer;

// Initialize tokenizer from JSON file
int bpe_load(BPETokenizer* tok, const char* json_path);

// Encode text to token IDs
// Returns number of tokens, fills ids array
int bpe_encode(const BPETokenizer* tok, const char* text, int* ids, int max_tokens);

// Decode token IDs to text
// Returns pointer to static buffer (not thread-safe)
const char* bpe_decode(const BPETokenizer* tok, const int* ids, int n_tokens);

// Get piece for token ID
const char* bpe_get_piece(const BPETokenizer* tok, int id);

// Get ID for piece (or UNK if not found)
int bpe_get_id(const BPETokenizer* tok, const char* piece);

// Free tokenizer resources
void bpe_free(BPETokenizer* tok);

#endif // BPE_TOKENIZER_H
