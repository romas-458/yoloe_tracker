#!/usr/bin/env bash
# Ablation Study Runner for YOLOe-VP-IoU Tracker
# Runs all ablation experiments A1-A8 on the test set
# Compatible with both bash and sh

DATASET_PATH="/home/peoly/datasets/lasot/test"
TEST_LIST="lasot_ablation_test_2.txt"
OUTPUT_DIR="results_ablation"

echo "========================================="
echo "YOLOe-VP-IoU Ablation Study"
echo "========================================="
echo "Dataset: $DATASET_PATH"
echo "Test list: $TEST_LIST"
echo "Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# List of ablation configs (POSIX-compatible, no arrays)
# Loop through all A*.yaml files in order
for config in yoloe-vp-iou/ablation/A7_dual_memory.yaml \
              yoloe-vp-iou/ablation/A8_full.yaml; do
    # Extract experiment name (A1, A2, etc.)
    exp_name=$(basename "$config" .yaml)

    echo "----------------------------------------"
    echo "Running experiment: $exp_name"
    echo "Config: $config"
    echo "----------------------------------------"

    # Run evaluation
    python scripts/modular_evaluation.py \
        -d "$DATASET_PATH" \
        -o "$OUTPUT_DIR/results_${exp_name}.json" \
        --tracker-config "$config" \
        --dataset lasot \
        --test-list "$TEST_LIST" \
        --num-frames 10000 \
        --visualize

    if [ $? -eq 0 ]; then
        echo "✓ $exp_name completed successfully"
    else
        echo "✗ $exp_name failed"
    fi
    echo ""
done

echo "========================================="
echo "Ablation study complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================="

# Generate comparison report
echo ""
echo "Generating comparison report..."
if command -v python &> /dev/null; then
    python scripts/compare_ablation_results.py \
        --results-dir "$OUTPUT_DIR" \
        --output "$OUTPUT_DIR/ablation_comparison.csv" 2>/dev/null || echo "⚠ Comparison script requires pandas"
fi

echo "Done!"
