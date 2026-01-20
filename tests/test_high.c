// test_high.c - Test High Math Engine (Go inner_world)
// ═══════════════════════════════════════════════════════════════════════════════
// המוח המתמטי
// The mathematical brain
// ═══════════════════════════════════════════════════════════════════════════════

#include <stdio.h>
#include <stdlib.h>
#include "inner_world.h"

int main(void) {
    printf("=== HIGH MATH ENGINE TEST ===\n\n");

    inner_world_init();

    // 1. Entropy
    printf("1. Entropy test:\n");
    float e1 = high_entropy("love joy happiness peace");
    float e2 = high_entropy("random words mix jumbled");
    printf("   'love joy happiness peace': %.3f\n", e1);
    printf("   'random words mix jumbled': %.3f\n", e2);

    // 2. Emotional score
    printf("\n2. Emotional score:\n");
    float pos = high_emotional_score("I love you, this is wonderful");
    float neg = high_emotional_score("I hate this, terrible awful");
    float neu = high_emotional_score("The table is brown");
    printf("   positive text: %.3f\n", pos);
    printf("   negative text: %.3f\n", neg);
    printf("   neutral text:  %.3f\n", neu);

    // 3. Semantic distance
    printf("\n3. Semantic distance:\n");
    float d1 = high_semantic_distance("I love cats", "I love dogs");
    float d2 = high_semantic_distance("I love cats", "The weather is nice");
    printf("   'cats' vs 'dogs': %.3f (similar)\n", d1);
    printf("   'cats' vs 'weather': %.3f (different)\n", d2);

    // 4. Emotional alignment
    printf("\n4. Emotional alignment:\n");
    float a1 = high_emotional_alignment("I am so happy", "This is wonderful");
    float a2 = high_emotional_alignment("I am so happy", "This is terrible");
    printf("   happy + wonderful: %.3f (aligned)\n", a1);
    printf("   happy + terrible:  %.3f (opposite)\n", a2);

    // 5. Perplexity
    printf("\n5. Perplexity:\n");
    float p1 = high_perplexity("the the the the");
    float p2 = high_perplexity("she resonates with presence");
    printf("   'the the the the': %.3f (repetitive)\n", p1);
    printf("   'she resonates...': %.3f (natural)\n", p2);

    // 6. N-gram overlap
    printf("\n6. N-gram overlap (bigrams):\n");
    float n1 = high_ngram_overlap("love is beautiful", "love is wonderful", 2);
    float n2 = high_ngram_overlap("love is beautiful", "hate is ugly", 2);
    printf("   similar phrases: %.3f\n", n1);
    printf("   opposite phrases: %.3f\n", n2);

    // 7. Valence and arousal
    printf("\n7. Detailed emotional analysis:\n");
    float v = high_analyze_valence("I am excited and happy about this amazing day");
    float a = high_analyze_arousal("I am excited and happy about this amazing day");
    printf("   valence: %.3f\n", v);
    printf("   arousal: %.3f\n", a);

    inner_world_shutdown();

    printf("\n=== HIGH MATH TEST COMPLETE ===\n");
    return 0;
}
