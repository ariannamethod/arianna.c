#!/bin/bash
# Run all Arianna tests

# Set library path for Go dylibs
export DYLD_LIBRARY_PATH=./lib:./bin:$DYLD_LIBRARY_PATH

TESTS="test_amlk test_cloud test_comprehensive test_accumulator test_inner test_mathbrain test_pandora test_selfsense test_amk test_delta_enhanced test_blood test_high test_inner_world test_julia test_ariannabody_extended test_sartre test_sartre_comprehensive test_sampling_edge_cases test_delta"

PASSED=0
FAILED=0
TOTAL=0

echo "=========================================="
echo "RUNNING ALL ARIANNA TESTS"
echo "=========================================="

for test in $TESTS; do
    TOTAL=$((TOTAL + 1))
    echo ""
    echo "[$TOTAL] Running $test..."
    
    # Compile test
    make $test > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "  ❌ COMPILATION FAILED"
        FAILED=$((FAILED + 1))
        continue
    fi
    
    # Run test
    ./bin/$test > /tmp/${test}_output.txt 2>&1
    if [ $? -eq 0 ]; then
        echo "  ✅ PASSED"
        PASSED=$((PASSED + 1))
    else
        echo "  ❌ FAILED"
        FAILED=$((FAILED + 1))
        echo "     Output:"
        tail -10 /tmp/${test}_output.txt | sed 's/^/     /'
    fi
done

echo ""
echo "=========================================="
echo "SUMMARY: $PASSED/$TOTAL tests passed"
echo "=========================================="

if [ $FAILED -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED!"
    exit 0
else
    echo "⚠️  $FAILED tests failed"
    exit 1
fi
