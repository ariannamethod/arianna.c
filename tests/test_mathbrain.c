/*
 * test_mathbrain.c - Test arithmetic through resonance
 */

#include "mathbrain.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    srand(time(NULL));

    printf("=== MathBrain Test ===\n\n");

    MathBrain mb;
    init_mathbrain(&mb);

    // Test parsing
    printf("Testing parse_math_expr:\n");
    int a, b;
    MathOp op;
    const char* exprs[] = {"7 + 5", "12 - 3", "6*7", "20 / 4"};
    for (int i = 0; i < 4; i++) {
        if (parse_math_expr(exprs[i], &a, &op, &b)) {
            printf("  '%s' -> a=%d, op=%d, b=%d\n", exprs[i], a, op, b);
        }
    }

    printf("\n--- Training phase (100 random problems) ---\n");

    // Train on random arithmetic
    for (int i = 0; i < 100; i++) {
        int a = rand() % 50;
        int b = rand() % 50 + 1;  // Avoid div by zero
        MathOp op = rand() % OP_COUNT;

        float conf;
        int pred = compute_op(&mb, a, op, b, &conf);

        // Compute ground truth
        int truth;
        switch (op) {
            case OP_ADD: truth = a + b; break;
            case OP_SUB: truth = a - b; break;
            case OP_MUL: truth = a * b; break;
            case OP_DIV: truth = a / b; break;
            default: truth = 0;
        }

        // Clamp for learning
        if (truth >= 0 && truth < MATH_MAX_NUM) {
            mathbrain_learn(&mb, a, op, b, truth);
        }

        if (i < 5 || i >= 95) {
            const char* ops[] = {"+", "-", "*", "/"};
            printf("  %d %s %d = %d (predicted %d, conf=%.2f) %s\n",
                   a, ops[op], b, truth, pred, conf,
                   (pred == truth) ? "[OK]" : "[X]");
        }
        if (i == 5) printf("  ...\n");
    }

    printf("\n--- Accuracy check ---\n");
    float acc = get_recent_accuracy(&mb, 50);
    printf("Recent accuracy: %.1f%%\n", acc * 100.0f);

    printf("\n--- Testing specific expressions ---\n");
    const char* tests[] = {
        "5 + 3",
        "10 - 4",
        "6 * 7",
        "20 / 5"
    };

    for (int i = 0; i < 4; i++) {
        char result[32];
        int ok = compute_from_text(&mb, tests[i], result, 32);
        printf("  %s = %s %s\n", tests[i], result, ok ? "[learned correctly]" : "");
    }

    print_mathbrain_stats(&mb);

    printf("\n--- Number similarities (0-9) ---\n");
    print_number_similarities(&mb);

    // Test save/load
    printf("\n--- Testing save/load ---\n");
    if (save_mathbrain(&mb, "/tmp/mathbrain_test.bin") == 0) {
        printf("Saved to /tmp/mathbrain_test.bin\n");

        MathBrain mb2;
        init_mathbrain(&mb2);
        if (load_mathbrain(&mb2, "/tmp/mathbrain_test.bin") == 0) {
            printf("Loaded successfully\n");
            printf("Loaded accuracy: %.1f%%\n", mb2.history.accuracy_ema * 100.0f);
        }
        free_mathbrain(&mb2);
    }

    free_mathbrain(&mb);
    printf("\nDone!\n");

    return 0;
}
