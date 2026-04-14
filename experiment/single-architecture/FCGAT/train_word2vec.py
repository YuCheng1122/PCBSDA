"""
Train Word2Vec CBOW on Pcode corpus for FCGAT experiment.
Paper settings: vector_size=100, window=2, epochs=100

Run from PCBSDA root:
    python experiment/single-architecture/FCGAT/train_word2vec.py --arch x86_64
    python experiment/single-architecture/FCGAT/train_word2vec.py  # all architectures
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import argparse
import logging
import pickle
from pathlib import Path
from typing import Union

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from datasets import Dataset, load_from_disk
from datasets import disable_progress_bar as hf_disable_progress_bar

# Suppress noisy progress output
hf_disable_progress_bar()
logging.getLogger("gensim").setLevel(logging.WARNING)


class EpochLogger(CallbackAny2Vec):
    def __init__(self, total: int, arch: str):
        self.epoch  = 0
        self.total  = total
        self.arch   = arch

    def on_epoch_end(self, model):
        self.epoch += 1
        print(f"\r[{self.arch}] epoch {self.epoch}/{self.total}", end="", flush=True)
        if self.epoch == self.total:
            print()

BASE_PATH = "/home/tommy/Project/PCBSDA"
CORPUS_DIR = f"{BASE_PATH}/experiment/outputs/raw_data/single-architecture/FCGAT/corpus"
OUTPUT_DIR = f"{BASE_PATH}/experiment/outputs/models/single-architecture/FCGAT/word2vec"
ALL_ARCHS  = ["Intel", "ARM-32", "x86_64", "MIPS"]

# Paper settings (Section V-A)
VECTOR_SIZE = 100
WINDOW      = 2
EPOCHS      = 100
MIN_COUNT   = 3
WORKERS     = os.cpu_count()
SEED        = 42


def load_corpus_dataset(corpus_path: Union[str, Path]) -> Dataset:
    corpus_path = Path(corpus_path)
    processed_path = corpus_path.parent / f"{corpus_path.stem}_processed"
    if processed_path.exists():
        dataset = load_from_disk(str(processed_path))
        return dataset

    print(f"[cache] building Arrow cache → {processed_path}")

    def data_generator():
        with open(corpus_path, "rb") as f:
            while True:
                try:
                    batch = pickle.load(f)
                    if isinstance(batch, list):
                        for sentence in batch:
                            if isinstance(sentence, list):
                                yield {"text": " ".join(sentence)}
                except EOFError:
                    break
                except Exception as e:
                    print(f"Error reading batch: {e}")
                    break

    dataset = Dataset.from_generator(data_generator)
    dataset.save_to_disk(str(processed_path))
    return dataset


def train_arch(arch: str):
    corpus_path = os.path.join(CORPUS_DIR, f"corpus_{arch}.pkl")
    output_dir  = os.path.join(OUTPUT_DIR, arch)

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    print(f"[{arch}] loading corpus ...")
    dataset = load_corpus_dataset(corpus_path)
    sentences = [item["text"].split() for item in dataset]
    print(f"[{arch}] {len(sentences):,} sentences  →  training CBOW "
          f"(vector_size={VECTOR_SIZE}, window={WINDOW}, epochs={EPOCHS}) ...")

    model = Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=WORKERS,
        sg=0,        # CBOW
        epochs=EPOCHS,
        seed=SEED,
        callbacks=[EpochLogger(EPOCHS, arch)],
    )

    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, "cbow_model.model"))
    model.wv.save(os.path.join(output_dir, "cbow_vectors.kv"))
    print(f"[{arch}] done  vocab={len(model.wv):,}  →  {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Word2Vec CBOW for FCGAT (paper settings)")
    parser.add_argument("--arch", type=str, default=None,
                        help=f"Target arch. If omitted, runs all: {ALL_ARCHS}")
    args = parser.parse_args()

    archs = [args.arch] if args.arch else ALL_ARCHS
    for arch in archs:
        train_arch(arch)


if __name__ == "__main__":
    main()
