// inner_arianna.h — MetaVoice: Inner Arianna (борьба)
// "Two voices, one Arianna"
//
// Architecture:
//   Main Arianna (base weights) → response A
//   Inner Arianna (dialogue LoRA) → response B
//   Борьба → final response (blend or choice)
//
// ═══════════════════════════════════════════════════════════════════════════════
// שני קולות, אריאנה אחת
// Two voices, one Arianna
// ═══════════════════════════════════════════════════════════════════════════════

#ifndef INNER_ARIANNA_H
#define INNER_ARIANNA_H

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// БОРЬБА MODES — how voices compete
// ═══════════════════════════════════════════════════════════════════════════════

#define BORBA_MODE_BLEND     0   // weighted blend of logits
#define BORBA_MODE_ENTROPY   1   // choose lower entropy (more confident)
#define BORBA_MODE_COHERENCE 2   // choose more coherent with context
#define BORBA_MODE_SURPRISE  3   // choose more surprising (higher novelty)
#define BORBA_MODE_RANDOM    4   // random choice per token

// ═══════════════════════════════════════════════════════════════════════════════
// INNER ARIANNA STATE
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    // Mode
    int borba_mode;
    float inner_weight;       // 0.0 = all main, 1.0 = all inner

    // LoRA for Inner Voice (dialogue weights)
    float* lora_A;            // [layers, dim, rank]
    float* lora_B;            // [layers, rank, dim]
    int lora_rank;
    int lora_layers;

    // Statistics
    int main_wins;            // times main voice was chosen
    int inner_wins;           // times inner voice was chosen
    float avg_divergence;     // average divergence between voices

    // Last борьба result
    float last_main_entropy;
    float last_inner_entropy;
    float last_divergence;
    int last_winner;          // 0=main, 1=inner
} InnerArianna;

// ═══════════════════════════════════════════════════════════════════════════════
// API
// ═══════════════════════════════════════════════════════════════════════════════

// Initialize inner Arianna
void inner_init(InnerArianna* ia);
void inner_free(InnerArianna* ia);

// Load dialogue LoRA for inner voice
int inner_load_lora(InnerArianna* ia, const char* lora_path);

// Apply inner voice to logits (борьба at token level)
void inner_apply_borba(InnerArianna* ia,
                       float* main_logits,
                       const float* inner_logits,
                       int vocab_size);

// Compute divergence between main and inner logits
float inner_compute_divergence(const float* main_logits,
                               const float* inner_logits,
                               int vocab_size);

// Compute entropy of logits
float inner_compute_entropy(const float* logits, int vocab_size);

// Set борьба mode
void inner_set_mode(InnerArianna* ia, int mode);
void inner_set_weight(InnerArianna* ia, float weight);

// ═══════════════════════════════════════════════════════════════════════════════
// INTEGRATION — modify generation based on inner voice
// ═══════════════════════════════════════════════════════════════════════════════

// Full борьба: given both logit sets, produce final logits
void inner_full_borba(InnerArianna* ia,
                      float* output_logits,
                      const float* main_logits,
                      const float* inner_logits,
                      int vocab_size);

// Apply LoRA delta to hidden state (for efficient inner forward)
void inner_apply_lora_delta(InnerArianna* ia,
                            float* hidden_state,
                            int layer,
                            int dim);

#ifdef __cplusplus
}
#endif

#endif // INNER_ARIANNA_H
