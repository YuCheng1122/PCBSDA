import os
import pickle
import numpy as np
from gensim.models import Word2Vec, FastText
from tqdm import tqdm
import json
import argparse


BASE_PATH = "/home/tommy/Project/PCBSDA"

# Input: raw gpickle graphs (no embedding yet)
RAW_GRAPH_DIR = f"{BASE_PATH}/ours/outputs/raw_data/gnn/gpickle"

# Output: embedded gpickle graphs (per model)
OUTPUT_BASE = f"{BASE_PATH}/ours/outputs/embedded_graphs"

# Trained embedding models
MODELS_BASE = f"{BASE_PATH}/ours/outputs/models/embedding"


def load_w2v_model(model_name):
    """Load trained word2vec-style model, return KeyedVectors."""
    model_dir = os.path.join(MODELS_BASE, model_name)
    if model_name == 'fasttext':
        model = FastText.load(os.path.join(model_dir, 'fasttext_model.model'))
    elif model_name == 'cbow':
        model = Word2Vec.load(os.path.join(model_dir, 'cbow_model.model'))
    elif model_name == 'skipgram':
        model = Word2Vec.load(os.path.join(model_dir, 'skipgram_model.model'))
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    return model.wv


def get_sentence_embedding(sentence, model):
    """Get sentence embedding by averaging word embeddings."""
    words = sentence.split()
    embeddings = []

    for word in words:
        if word in model:
            embeddings.append(model[word])

    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.vector_size)


def find_all_gpickle_files(base_path):
    """Find all gpickle files recursively."""
    gpickle_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.gpickle'):
                gpickle_files.append(os.path.join(root, file))
    return gpickle_files


def process_single_graph(graph_path, model):
    """Embed all nodes in a single graph using the w2v model."""
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)

    for node_id, node_data in graph.nodes(data=True):
        tokens = node_data.get('tokens', [])
        sentence = " ".join(tokens) if tokens else ""
        node_data['x'] = get_sentence_embedding(sentence, model)
        node_data.pop('function_name', None)
        node_data.pop('tokens', None)

    return graph


def batch_process_graphs(model_name):
    """Batch process all graph files with a given embedding model."""
    print(f"Loading {model_name} model...")
    model = load_w2v_model(model_name)

    gpickle_files = find_all_gpickle_files(RAW_GRAPH_DIR)
    print(f"Found {len(gpickle_files)} gpickle files")

    output_dir = os.path.join(OUTPUT_BASE, model_name)
    processed_count = 0
    failed_count = 0

    for file_path in tqdm(gpickle_files, desc=f"Embedding with {model_name}"):
        try:
            graph = process_single_graph(file_path, model)

            rel_path = os.path.relpath(file_path, RAW_GRAPH_DIR)
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, 'wb') as f:
                pickle.dump(graph, f)

            processed_count += 1

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            failed_count += 1

    stats = {
        'model_name': model_name,
        'total_files': len(gpickle_files),
        'processed_files': processed_count,
        'failed_files': failed_count,
        'embedding_dim': model.vector_size
    }

    stats_path = os.path.join(output_dir, f"processing_stats_{model_name}.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Results saved to: {output_dir}")
    print(f"Processed: {processed_count}, Failed: {failed_count}")


def main():
    parser = argparse.ArgumentParser(description='Batch embed graphs with w2v models')
    parser.add_argument('--model', type=str, default='all',
                        choices=['fasttext', 'cbow', 'skipgram', 'all'],
                        help='Which embedding model to use (default: all)')
    args = parser.parse_args()

    models = ['cbow', 'skipgram', 'fasttext'] if args.model == 'all' else [args.model]

    for model_name in models:
        print(f"\n=== Processing with {model_name} ===")
        batch_process_graphs(model_name)


if __name__ == "__main__":
    main()
