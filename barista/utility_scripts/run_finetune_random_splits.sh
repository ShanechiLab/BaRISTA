#!/bin/bash

# Usage:
# ./run_finetune_random_splits.sh --spe coords --checkpoint "pretrained_models/chans_chans.ckpt" --session HOLDSUBJ_2_HS2_6 --gpu 1 --exp sentence_onset
# ./run_finetune_random_splits.sh --spe destrieux --checkpoint "pretrained_models/parcels_chans.ckpt" --session HOLDSUBJ_2_HS2_6 --gpu 2 --exp speech_vs_nonspeech

# Default values
GPU=0
SEEDS=(0 1 2 3 4)
SESSION=""
CHECKPOINT=""
DATASET_CONFIG="barista/config/braintreebank.yaml"
TRAIN_CONFIG="barista/config/train.yaml"
MODEL_CONFIG="barista/config/model.yaml"
SPATIAL_GROUPING="coords"
EXPERIMENT="sentence_onset"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --session)
            SESSION="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --seeds)
            IFS=',' read -ra SEEDS <<< "$2"
            shift 2
            ;;
        --dataset_config)
            DATASET_CONFIG="$2"
            shift 2
            ;;
        --train_config)
            TRAIN_CONFIG="$2"
            shift 2
            ;;
        --exp)
            EXPERIMENT="$2"
            shift 2
            ;;
        --spe)
            SPATIAL_GROUPING="$2"
            shift 2
            ;;
        --model_config)
            MODEL_CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$SESSION" ]; then
    echo "Error: --session is required"
    exit 1
fi


NUM_SEEDS=${#SEEDS[@]}

# Create output directory
OUTPUT_DIR="results/${SESSION}_${EXPERIMENT}_model${SPATIAL_GROUPING}$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Sequential Multi-Seed Fine-tuning"
echo "=========================================="
echo "Session: $SESSION"
echo "Checkpoint: $CHECKPOINT"
echo "GPU: $GPU"
echo "Seeds: ${SEEDS[@]}"
echo "Number of runs: $NUM_SEEDS"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Arrays to store results
VAL_AUCS=()
BEST_TEST_AUCS=()
LAST_TEST_AUCS=()
FAILED_SEEDS=()

# Run jobs sequentially
for i in $(seq 0 $(($NUM_SEEDS - 1))); do
    SEED=${SEEDS[$i]}
    
    LOG_FILE="$OUTPUT_DIR/seed_${SEED}.log"
    
    echo "=========================================="
    echo "Running job $((i+1))/$NUM_SEEDS: Seed=$SEED"
    echo "=========================================="
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Run training
    CUDA_VISIBLE_DEVICES=$GPU python barista/train.py \
        --dataset_config "$DATASET_CONFIG" \
        --train_config "$TRAIN_CONFIG" \
        --model_config "$MODEL_CONFIG" \
        --override \
            seed=$SEED \
            device=cuda:0 \
            checkpoint_path="$CHECKPOINT" \
            force_nonoverlap=True \
            experiment="$EXPERIMENT" \
            tokenizer.spatial_grouping="$SPATIAL_GROUPING" \
            "finetune_sessions=['$SESSION']" \
        2>&1 | tee "$LOG_FILE"
    
    # Check if job completed successfully
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo ""
        echo "✓ Job $((i+1)) completed successfully"
        
        # Extract results from log file
        VAL_AUC=$(grep "BEST VAL AUC" "$LOG_FILE" | awk '{print $NF}')
        BEST_TEST_AUC=$(grep "^BEST TEST AUC" "$LOG_FILE" | tail -1 | awk '{print $NF}')
        LAST_TEST_AUC=$(grep "LAST TEST AUC" "$LOG_FILE" | awk '{print $NF}')
        
        if [ ! -z "$VAL_AUC" ] && [ ! -z "$BEST_TEST_AUC" ] && [ ! -z "$LAST_TEST_AUC" ]; then
            VAL_AUCS+=($VAL_AUC)
            BEST_TEST_AUCS+=($BEST_TEST_AUC)
            LAST_TEST_AUCS+=($LAST_TEST_AUC)
            echo "  Val AUC: $VAL_AUC"
            echo "  Test AUC: $BEST_TEST_AUC"
            echo "  Last Test AUC: $LAST_TEST_AUC"
        else
            echo "  Warning: Could not extract AUC values"
            FAILED_SEEDS+=($SEED)
        fi
    else
        echo ""
        echo "✗ Job $((i+1)) failed"
        FAILED_SEEDS+=($SEED)
    fi
    
    echo ""
done

echo "=========================================="
echo "All jobs completed!"
echo "=========================================="
echo ""

# Calculate statistics using Python
STATS_SCRIPT="$OUTPUT_DIR/calculate_stats.py"
cat > "$STATS_SCRIPT" << 'EOF'
import sys
import numpy as np

def calculate_stats(values):
    if len(values) == 0:
        return None, None
    arr = np.array(values, dtype=float)
    return np.mean(arr), np.std(arr)

# Read values from command line
val_aucs = [float(x) for x in sys.argv[1].split(',') if x]
best_test_aucs = [float(x) for x in sys.argv[2].split(',') if x]
last_test_aucs = [float(x) for x in sys.argv[3].split(',') if x]

val_mean, val_std = calculate_stats(val_aucs)
best_test_mean, best_test_std = calculate_stats(best_test_aucs)
last_test_mean, last_test_std = calculate_stats(last_test_aucs)

print(f"VAL_MEAN={val_mean:.4f}")
print(f"VAL_STD={val_std:.4f}")
print(f"BEST_TEST_MEAN={best_test_mean:.4f}")
print(f"BEST_TEST_STD={best_test_std:.4f}")
print(f"LAST_TEST_MEAN={last_test_mean:.4f}")
print(f"LAST_TEST_STD={last_test_std:.4f}")

# Print individual values
print("\nIndividual Results:")
for i, (val, test, last_test) in enumerate(zip(val_aucs, best_test_aucs, last_test_aucs), 1):
    print(f"  Run {i}: Val AUC = {val:.4f}, Best Test AUC = {test:.4f}, Last Test AUC = {last_test:.4f}")
EOF

# Convert arrays to comma-separated strings
VAL_AUCS_STR=$(IFS=,; echo "${VAL_AUCS[*]}")
BEST_TEST_AUCS_STR=$(IFS=,; echo "${BEST_TEST_AUCS[*]}")
LAST_TEST_AUCS_STR=$(IFS=,; echo "${LAST_TEST_AUCS[*]}")

# Calculate and display statistics
if [ ${#BEST_TEST_AUCS[@]} -gt 0 ]; then
    echo "=========================================="
    echo "FINAL RESULTS"
    echo "=========================================="
    
    STATS_OUTPUT=$(python "$STATS_SCRIPT" "$VAL_AUCS_STR" "$BEST_TEST_AUCS_STR" "$LAST_TEST_AUCS_STR")
    echo "$STATS_OUTPUT"
    
    VAL_MEAN=$(awk -F= '/^VAL_MEAN=/{print $2; exit}' <<<"$STATS_OUTPUT")
    VAL_STD=$(awk -F= '/^VAL_STD=/{print $2; exit}' <<<"$STATS_OUTPUT")
    BEST_TEST_MEAN=$(awk -F= '/^BEST_TEST_MEAN=/{print $2; exit}' <<<"$STATS_OUTPUT")
    BEST_TEST_STD=$(awk -F= '/^BEST_TEST_STD=/{print $2; exit}' <<<"$STATS_OUTPUT")
    LAST_TEST_MEAN=$(awk -F= '/^LAST_TEST_MEAN=/{print $2; exit}' <<<"$STATS_OUTPUT")
    LAST_TEST_STD=$(awk -F= '/^LAST_TEST_STD=/{print $2; exit}' <<<"$STATS_OUTPUT")

    echo ""
    echo "Summary:"
    echo "  Validation AUC: ${VAL_MEAN} ± ${VAL_STD}"
    echo "  Best Test AUC:       ${BEST_TEST_MEAN} ± ${BEST_TEST_STD}"
    echo "  Last Test AUC:       ${LAST_TEST_MEAN} ± ${LAST_TEST_STD}"
    echo ""
    echo "Successful runs: ${#BEST_TEST_AUCS[@]}/$NUM_SEEDS"
    
    if [ ${#FAILED_SEEDS[@]} -gt 0 ]; then
        echo "Failed seeds: ${FAILED_SEEDS[@]}"
    fi
    
    echo "=========================================="
    
    # Save summary to file
    SUMMARY_FILE="$OUTPUT_DIR/summary.txt"
    {
        echo "Summary Report - $(date)"
        echo "=================================="
        echo "Session: $SESSION"
        echo "Checkpoint: $CHECKPOINT"
        echo "GPU: $GPU"
        echo "Seeds: ${SEEDS[@]}"
        echo ""
        echo "FINAL RESULTS"
        echo "=================================="
        echo "$STATS_OUTPUT"
        echo ""
        echo "Summary:"
        echo "  Validation AUC: ${VAL_MEAN} ± ${VAL_STD}"
        echo "  Test AUC:       ${BEST_TEST_MEAN} ± ${BEST_TEST_STD}"
        echo "  Last Test AUC:       ${LAST_TEST_MEAN} ± ${LAST_TEST_STD}"
        echo ""
        echo "Successful runs: ${#BEST_TEST_AUCS[@]}/$NUM_SEEDS"
        if [ ${#FAILED_SEEDS[@]} -gt 0 ]; then
            echo "Failed seeds: ${FAILED_SEEDS[@]}"
        fi
    } > "$SUMMARY_FILE"
    
    echo ""
    echo "Summary saved to: $SUMMARY_FILE"
    echo "All logs saved to: $OUTPUT_DIR"
else
    echo "ERROR: No successful runs completed"
    exit 1
fi

# Clean up temporary script
rm "$STATS_SCRIPT"

# Exit with error if any jobs failed
if [ ${#FAILED_SEEDS[@]} -gt 0 ]; then
    exit 1
fi

exit 0
