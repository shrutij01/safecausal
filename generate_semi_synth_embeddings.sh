#!/bin/bash

# Generate contrastive embeddings for semi-synthetic datasets with Gemma and Pythia
# Semi-synthetic datasets: eng-french, eng-german, masc-fem-eng

# Configuration
DATASETS=("eng-french" "eng-german" "masc-fem-eng")
MODELS=("google/gemma-2-2b-it" "EleutherAI/pythia-70m-deduped")
LAYERS=(25 5)  # Layer 25 for Gemma, Layer 5 for Pythia
POOLING_METHOD="last_token"
NUM_SAMPLES=100
SPLIT=0.9
BATCH_SIZE=128

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STORE_EMBEDDINGS_SCRIPT="${SCRIPT_DIR}/ssae/store_embeddings.py"

# Check if store_embeddings.py exists
if [ ! -f "$STORE_EMBEDDINGS_SCRIPT" ]; then
    echo "Error: store_embeddings.py not found at $STORE_EMBEDDINGS_SCRIPT"
    exit 1
fi

# Create logs directory
mkdir -p logs

echo "Starting contrastive embedding generation for semi-synthetic datasets..."
echo "Datasets: ${DATASETS[*]}"
echo "Models: ${MODELS[*]}"
echo "=================================================="

# Counter for tracking progress
total_jobs=$((${#DATASETS[@]} * ${#MODELS[@]}))
current_job=0

# Generate embeddings for each dataset-model combination
for dataset in "${DATASETS[@]}"; do
    for i in "${!MODELS[@]}"; do
        model="${MODELS[$i]}"
        layer="${LAYERS[$i]}"

        ((current_job++))
        echo ""
        echo "[$current_job/$total_jobs] Processing: $dataset with $model (layer $layer)"
        echo "=================================================="

        # Set model-specific name for output
        if [[ "$model" == "google/gemma-2-2b-it" ]]; then
            model_name="gemma2"
        elif [[ "$model" == "EleutherAI/pythia-70m-deduped" ]]; then
            model_name="pythia70m"
        fi

        # Generate timestamp for unique log files
        timestamp=$(date +"%Y%m%d_%H%M%S")
        log_file="logs/embeddings_${dataset}_${model_name}_${timestamp}.log"

        echo "Command: python $STORE_EMBEDDINGS_SCRIPT"
        echo "  --dataset $dataset"
        echo "  --model_id $model"
        echo "  --layer $layer"
        echo "  --pooling-method $POOLING_METHOD"
        echo "  --num-samples $NUM_SAMPLES"
        echo "  --split $SPLIT"
        echo "  --batch-size $BATCH_SIZE"
        echo "Log file: $log_file"
        echo ""

        # Run the embedding generation
        python "$STORE_EMBEDDINGS_SCRIPT" \
            --dataset "$dataset" \
            --model_id "$model" \
            --layer "$layer" \
            --pooling-method "$POOLING_METHOD" \
            --num-samples "$NUM_SAMPLES" \
            --split "$SPLIT" \
            --batch-size "$BATCH_SIZE" \
            2>&1 | tee "$log_file"

        # Check if the command was successful
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "✅ Successfully generated embeddings for $dataset with $model"
        else
            echo "❌ Failed to generate embeddings for $dataset with $model"
            echo "Check log file: $log_file"
        fi

        echo "=================================================="
    done
done

echo ""
echo "🎉 Contrastive embedding generation completed!"
echo "Generated embeddings for ${#DATASETS[@]} datasets with ${#MODELS[@]} models"
echo "Total jobs processed: $total_jobs"
echo "Check logs directory for detailed output"
echo ""
echo "Expected output files will be saved to:"
echo "/network/scratch/j/joshi.shruti/ssae/{dataset_name}/{dataset}_{model}_{layer}_{pooling_method}.h5"
echo "/network/scratch/j/joshi.shruti/ssae/{dataset_name}/{dataset}_{model}_{layer}_{pooling_method}.yaml"