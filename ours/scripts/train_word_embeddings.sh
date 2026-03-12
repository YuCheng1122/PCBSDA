#!/bin/bash
# Train word embedding models: CBOW, Skip-gram, FastText
# Usage: bash ours/scripts/train_word_embeddings.sh [cbow|skipgram|fasttext|all]

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

METHODS=${1:-all}

run_cbow() {
    echo "=========================================="
    echo "  Training CBOW"
    echo "=========================================="
    python -m ours.src.embedding.cbow.main
    echo "CBOW training complete."
    echo ""
}

run_skipgram() {
    echo "=========================================="
    echo "  Training Skip-gram"
    echo "=========================================="
    python -m ours.src.embedding.skipgram.main
    echo "Skip-gram training complete."
    echo ""
}

run_fasttext() {
    echo "=========================================="
    echo "  Training FastText"
    echo "=========================================="
    python -m ours.src.embedding.fast_text.main
    echo "FastText training complete."
    echo ""
}

case "$METHODS" in
    cbow)
        run_cbow
        ;;
    skipgram)
        run_skipgram
        ;;
    fasttext)
        run_fasttext
        ;;
    all)
        run_cbow
        run_skipgram
        run_fasttext
        echo "=========================================="
        echo "  All word embedding models trained!"
        echo "=========================================="
        ;;
    *)
        echo "Usage: $0 [cbow|skipgram|fasttext|all]"
        exit 1
        ;;
esac
