"""
trainEmbedding.py — Train fastText skipgram embeddings per architecture.

Usage:
    python trainEmbedding.py --arch x86_64
    python trainEmbedding.py --arch arm_32
    python trainEmbedding.py --arch mips_32
    python trainEmbedding.py --all

Output: <OUTPUT_DIR>/<arch>.vec  (word2vec text format, for use with map_embeddings.py)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import fasttext

# ── paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get('PCBSDA_UNIMAP_DATA_ROOT', PROJECT_ROOT / 'experiment' / 'outputs' / 'raw_data' / 'cross-architecture' / 'unimap'))
CORPUS_DIR = DATA_ROOT / 'corpus_norm'
OUTPUT_DIR = PROJECT_ROOT / 'experiment' / 'cross-architecture' / 'unimap' / 'embeddings'

ARCH_CORPUS = {
    'x86_64':  CORPUS_DIR / 'x86_64_corpus_norm.txt',
    'arm_32':  CORPUS_DIR / 'arm_32_corpus_norm.txt',
    'mips_32': CORPUS_DIR / 'mips_32_corpus_norm.txt',
}

# ── fastText hyperparameters ──────────────────────────────────────────────────

FT_PARAMS = dict(
    model='skipgram',
    dim=200,
    epoch=5,
    lr=0.05,
    minCount=5,
    wordNgrams=1,
    minn=0,             # disable character n-grams — each instruction is an atomic token
    maxn=0,
    thread=os.cpu_count() or 4,
    verbose=2,
)

def train_arch(arch: str) -> None:
    corpus = ARCH_CORPUS[arch]
    if not corpus.exists():
        print(f'[{arch}] Corpus not found: {corpus}', file=sys.stderr)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_vec = OUTPUT_DIR / f'{arch}.vec'

    print(f'[{arch}] Training fastText skipgram dim=200 on {corpus}')
    t0 = time.time()

    model = fasttext.train_unsupervised(
        str(corpus),
        **FT_PARAMS,
    )

    # Save .bin alongside for later inspection (optional)
    model.save_model(str(OUTPUT_DIR / f'{arch}.bin'))

    # Export .vec (word2vec text format) — used by map_embeddings.py
    words = model.get_words()
    with open(out_vec, 'w', encoding='utf-8') as f:
        f.write(f'{len(words)} {model.get_dimension()}\n')
        for w in words:
            vec = model.get_word_vector(w)
            f.write(w + ' ' + ' '.join(f'{v:.6f}' for v in vec) + '\n')

    elapsed = time.time() - t0
    print(f'[{arch}] Done in {elapsed:.1f}s  →  {out_vec}  ({len(words):,} words)')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train fastText skipgram embeddings for instruction tokens'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--arch', choices=list(ARCH_CORPUS), help='Single architecture')
    group.add_argument('--all', action='store_true', help='Train all three architectures')
    args = parser.parse_args()

    archs = list(ARCH_CORPUS) if args.all else [args.arch]
    for arch in archs:
        train_arch(arch)


if __name__ == '__main__':
    main()
