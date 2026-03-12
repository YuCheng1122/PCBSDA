#!/bin/bash
# Full W2V pipeline: train embeddings -> embed graphs -> train GNN
# Usage: bash ours/scripts/run_w2v_experiments.sh [cbow|skipgram|fasttext|all]

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

MODELS=${1:-all}

# Create log directory
LOG_DIR="ours/outputs/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/w2v_pipeline_${TIMESTAMP}.log"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Resolve model list
if [ "$MODELS" = "all" ]; then
    MODEL_LIST=("cbow" "skipgram" "fasttext")
else
    MODEL_LIST=("$MODELS")
fi

log "======================================"
log "W2V Experiment Pipeline"
log "======================================"
log "Models: ${MODEL_LIST[*]}"
log "Log file: $LOG_FILE"

# Step 1: Train word embedding models
log ""
log "======================================"
log "Step 1: Training Word Embedding Models"
log "======================================"

for model in "${MODEL_LIST[@]}"; do
    log "Training $model..."
    bash ours/scripts/train_word_embeddings.sh "$model" 2>&1 | tee -a "$LOG_FILE"
    log "Done: $model"
done

# Step 2: Embed graphs with trained models
log ""
log "======================================"
log "Step 2: Embedding Graphs"
log "======================================"

for model in "${MODEL_LIST[@]}"; do
    log "Embedding graphs with $model..."
    python -m ours.src.embedding.batch_embedding_w2v --model "$model" 2>&1 | tee -a "$LOG_FILE"
    log "Done: $model"
done

# Step 3: Train GNN for each embedding model
log ""
log "======================================"
log "Step 3: Training GNN Models"
log "======================================"

for model in "${MODEL_LIST[@]}"; do
    log "Training GNN with $model embeddings..."
    python -m ours.src.gnn.w2v_training --model "$model" 2>&1 | tee -a "$LOG_FILE"
    log "Done: $model"
done

log ""
log "======================================"
log "Pipeline Complete!"
log "======================================"
log "Results:"
for model in "${MODEL_LIST[@]}"; do
    log "  Embedding model: ours/outputs/models/embedding/$model/"
    log "  Embedded graphs: ours/outputs/embedded_graphs/$model/"
    log "  GNN model:       ours/outputs/models/gnn/$model/"
    log "  GNN results:     ours/outputs/results/gnn/$model/"
done
log "Log: $LOG_FILE"
