#!/bin/bash
# Run all C-only tests (without Go library dependency)

echo "═══════════════════════════════════════════════════════════════════"
echo "  ARIANNA.C TEST SUITE (C-ONLY, NO GO)"
echo "  Testing 10 core modules"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

PASSED=0
FAILED=0

run_test() {
    local name=$1
    local bin=$2
    echo "[TEST] $name"
    if ./$bin > /dev/null 2>&1; then
        echo "  ✓ PASSED"
        ((PASSED++))
    else
        echo "  ✗ FAILED"
        ((FAILED++))
    fi
    echo ""
}

# Run all C-only tests
run_test "Julia Emotional Gradient" "bin/test_julia"
run_test "MathBrain" "bin/test_mathbrain"
run_test "Pandora N-gram Memory" "bin/test_pandora"
run_test "SelfSense Internal Signals" "bin/test_selfsense"
run_test "Inner Arianna Борьба" "bin/test_inner"
run_test "Quantum Accumulator" "bin/test_accumulator"
run_test "Enhanced Delta System" "bin/test_delta_enhanced"
run_test "Cloud 200K Emotion Detection" "bin/test_cloud"
run_test "AMK Kernel" "bin/test_amk"
run_test "Comprehensive Integration" "bin/test_comprehensive"

echo "═══════════════════════════════════════════════════════════════════"
echo "  RESULTS: $PASSED passed, $FAILED failed"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  Note: 4 tests skipped (require Go libinner_world):"
echo "    - test_blood"
echo "    - test_high"
echo "    - test_amlk"
echo "    - test_inner_world"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "  ✓ ALL C-ONLY TESTS PASSED!"
    exit 0
else
    echo "  ✗ SOME TESTS FAILED"
    exit 1
fi
