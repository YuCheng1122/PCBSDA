import os
import pickle
import torch
import numpy as np
from transformers import RobertaForMaskedLM, AutoTokenizer
from tqdm import tqdm
import json


BASE_PATH = "/home/tommy/Project/PCBSDA"
MODEL_PATH = f"{BASE_PATH}/ours/outputs/models/embedding/roberta/model_epoch_20"
RAW_GRAPH_DIR = f"{BASE_PATH}/ours/outputs/raw_data/gnn/gpickle"
OUTPUT_DIR = f"{BASE_PATH}/ours/outputs/embedded_graphs/roberta_20"


def load_pretrained_model(model_path=MODEL_PATH):
    """Load pretrained RoBERTa model and tokenizer."""
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = RobertaForMaskedLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded on device: {device}, hidden_size: {model.config.hidden_size}")
    return model, tokenizer, device


def get_embeddings_batch(sentences, model, tokenizer, device, batch_size=256):
    """Get mean-pooled embeddings for a list of sentences."""
    all_embeddings = []

    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]

        inputs = tokenizer(
            batch_sentences,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.roberta(**inputs)
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            embeddings = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
            all_embeddings.append(embeddings.cpu().numpy())

    if not all_embeddings:
        return np.array([])

    return np.concatenate(all_embeddings, axis=0)


def process_single_graph(graph_path, model, tokenizer, device):
    """Embed all nodes in a single graph, same output format as W2V."""
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)

    node_ids = []
    sentences = []
    for node_id, node_data in graph.nodes(data=True):
        tokens = node_data.get('tokens', [])
        sentence = " ".join(tokens) if tokens else ""
        node_ids.append(node_id)
        sentences.append(sentence)

    if sentences:
        all_embeddings = get_embeddings_batch(sentences, model, tokenizer, device)
    else:
        all_embeddings = np.array([])

    for i, node_id in enumerate(node_ids):
        node_data = graph.nodes[node_id]
        if all_embeddings.size > 0:
            node_data['x'] = all_embeddings[i].astype(np.float32)
        else:
            node_data['x'] = np.zeros(model.config.hidden_size, dtype=np.float32)
        node_data.pop('function_name', None)
        node_data.pop('tokens', None)

    return graph


def find_all_gpickle_files(base_path):
    """Find all gpickle files recursively."""
    gpickle_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.gpickle'):
                gpickle_files.append(os.path.join(root, file))
    return gpickle_files


def batch_process_graphs():
    """Batch process all graph files with RoBERTa embedding."""
    model, tokenizer, device = load_pretrained_model()

    gpickle_files = find_all_gpickle_files(RAW_GRAPH_DIR)
    print(f"Found {len(gpickle_files)} gpickle files")

    processed_count = 0
    failed_count = 0

    for file_path in tqdm(gpickle_files, desc="Embedding with roberta"):
        try:
            graph = process_single_graph(file_path, model, tokenizer, device)

            rel_path = os.path.relpath(file_path, RAW_GRAPH_DIR)
            output_path = os.path.join(OUTPUT_DIR, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, 'wb') as f:
                pickle.dump(graph, f)

            processed_count += 1

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            failed_count += 1

    stats = {
        'model_name': 'roberta',
        'total_files': len(gpickle_files),
        'processed_files': processed_count,
        'failed_files': failed_count,
        'embedding_dim': model.config.hidden_size
    }

    stats_path = os.path.join(OUTPUT_DIR, "processing_stats_roberta.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Processed: {processed_count}, Failed: {failed_count}")


if __name__ == "__main__":
    batch_process_graphs()
