"""
Embed FCGAT graphs using per-architecture Word2Vec CBOW models.

Each architecture has its own CBOW model (trained on that arch's Pcode corpus).
Node tokens are averaged into a feature vector stored as node['x'].

Input:
  CSV:       datasets/csv/single_arch_dataset.csv
  Raw graphs: experiment/outputs/raw_data/single-architecture/FCGAT/GNN/gpickle_single/{prefix}/{file}.gpickle
  CBOW:      experiment/outputs/models/single-architecture/FCGAT/word2vec/{arch}/cbow_model.model

Output:
  Embedded:  experiment/outputs/embedded_graphs/single-architecture/FCGAT/{arch}/{prefix}/{file}.gpickle

Run from PCBSDA root:
    python experiment/single-architecture/FCGAT/batch_embed_graphs.py
    python experiment/single-architecture/FCGAT/batch_embed_graphs.py --arch x86_64
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm

BASE_PATH   = "/home/tommy/Project/PCBSDA"
CSV_PATH    = f"{BASE_PATH}/datasets/csv/single_arch_dataset.csv"
RAW_DIR     = f"{BASE_PATH}/experiment/outputs/raw_data/single-architecture/FCGAT/GNN/gpickle_single"
MODEL_DIR   = f"{BASE_PATH}/experiment/outputs/models/single-architecture/FCGAT/word2vec"
OUTPUT_DIR  = f"{BASE_PATH}/experiment/outputs/embedded_graphs/single-architecture/FCGAT"
ALL_ARCHS   = ["Intel", "ARM-32", "x86_64", "MIPS"]

# All architectures share the same output directory.
# Each file has a unique hash name so there is no collision.
# load_graphs_from_df filters by CPU via the CSV, so no per-arch subdir is needed.


def get_node_embedding(tokens, wv):
    """Average token embeddings; return zero vector if no token is in vocab."""
    vectors = [wv[t] for t in tokens if t in wv]
    if vectors:
        return np.mean(vectors, axis=0).astype(np.float32)
    return np.zeros(wv.vector_size, dtype=np.float32)


def embed_arch(arch: str):
    model_path = os.path.join(MODEL_DIR, arch, "cbow_model.model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"CBOW model not found: {model_path}\n"
            "Run train_word2vec.py first."
        )

    print(f"\n{'='*60}")
    print(f"Arch: {arch}")
    print(f"  model : {model_path}")

    wv = Word2Vec.load(model_path).wv
    print(f"  vocab : {len(wv):,}  dim: {wv.vector_size}")

    df = pd.read_csv(CSV_PATH)
    arch_df = df[df["CPU"] == arch].reset_index(drop=True)
    print(f"  files : {len(arch_df):,}")

    out_dir = Path(OUTPUT_DIR)
    processed = skipped = missing = 0

    for _, row in tqdm(arch_df.iterrows(), total=len(arch_df), desc=arch):
        file_name = row["file_name"]
        prefix    = file_name[:2]

        src_path = Path(RAW_DIR) / prefix / f"{file_name}.gpickle"
        dst_path = out_dir       / prefix / f"{file_name}.gpickle"

        if dst_path.exists():
            skipped += 1
            continue

        if not src_path.exists():
            missing += 1
            continue

        with open(src_path, "rb") as f:
            graph = pickle.load(f)

        if graph.number_of_nodes() == 0:
            missing += 1
            continue

        for _, node_data in graph.nodes(data=True):
            tokens = node_data.get("tokens", [])
            node_data["x"] = get_node_embedding(tokens, wv)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_path, "wb") as f:
            pickle.dump(graph, f)

        processed += 1

    stats = {
        "arch": arch,
        "total_in_csv": len(arch_df),
        "processed": processed,
        "skipped_exists": skipped,
        "missing_source": missing,
        "embedding_dim": wv.vector_size,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"embedding_stats_{arch}.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  processed={processed}  skipped(exists)={skipped}  missing={missing}")
    print(f"  output   : {out_dir}")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Embed FCGAT graphs with per-arch CBOW"
    )
    parser.add_argument(
        "--arch", type=str, default=None,
        help=f"Target arch. If omitted, runs all: {ALL_ARCHS}",
    )
    args = parser.parse_args()

    archs = [args.arch] if args.arch else ALL_ARCHS
    for arch in archs:
        embed_arch(arch)

    print("\nAll done.")


if __name__ == "__main__":
    main()
