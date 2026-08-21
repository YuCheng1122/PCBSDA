"""
embedSequences.py — embed instruction sequences using mapped fastText .vec

Reads config.py to determine which pairs / CPUs to embed.

Usage:
    python embedSequences.py                        # embed all pairs in config
    python embedSequences.py --pair x86_64,ARM-32  # single cross pair
    python embedSequences.py --mode mono --cpu x86_64  # single mono CPU
    python embedSequences.py --force               # re-embed even if cache exists
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import get_lstm_config


def load_vec(path: str) -> dict:
    word2vec = {}
    with open(path, encoding='utf-8') as f:
        n, dim = map(int, f.readline().split())
        for line in f:
            parts = line.rstrip().split(' ')
            word2vec[parts[0]] = np.array(parts[1:], dtype=np.float32)
    return word2vec


def embed_file(txt_path: str, word2vec: dict, max_len: int, embed_dim: int) -> tuple:
    with open(txt_path, encoding='utf-8', errors='replace') as f:
        tokens = f.read().split()
    tokens = tokens[:max_len]
    mat = np.zeros((max_len, embed_dim), dtype=np.float32)
    oov = 0
    for i, tok in enumerate(tokens):
        vec = word2vec.get(tok)
        if vec is not None:
            mat[i] = vec
        else:
            oov += 1
    return mat, len(tokens), oov


def embed_cpus(df, cpus, vec_map, seq_dir, cpu_to_dir, cache_dir, max_len, embed_dim, force):
    """Embed all files for the given cpu list into cache_dir."""
    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subset = df[df['CPU'].isin(cpus)].reset_index(drop=True)
    skipped = total_tokens = total_oov = 0

    for _, row in tqdm(subset.iterrows(), total=len(subset), desc=f"Embedding {cpus}"):
        sha      = row['file_name']
        cpu      = row['CPU']
        arch_dir = cpu_to_dir[cpu]

        out_path = out_dir / f"{sha}.npy"
        if out_path.exists() and not force:
            skipped += 1
            continue

        txt_path = Path(seq_dir) / arch_dir / f"{sha}.txt"
        if not txt_path.exists():
            continue

        mat, n_tok, n_oov = embed_file(str(txt_path), vec_map[arch_dir], max_len, embed_dim)
        total_tokens += n_tok
        total_oov    += n_oov
        np.save(str(out_path), mat)

    total_saved = len(list(out_dir.glob('*.npy')))
    oov_rate    = total_oov / total_tokens if total_tokens > 0 else 0.0
    print(f"  → {total_saved} files in {out_dir}  (skipped {skipped} cached)")
    print(f"  OOV: {total_oov}/{total_tokens} ({oov_rate:.2%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',  choices=['cross', 'mono'], default=None)
    parser.add_argument('--pair',  default=None, help='e.g. x86_64,ARM-32')
    parser.add_argument('--cpu',   default=None, help='e.g. x86_64  (mono only)')
    parser.add_argument('--force', action='store_true', help='re-embed even if cache exists')
    args = parser.parse_args()

    cfg        = get_lstm_config()
    mode       = args.mode or cfg['mode']
    cpu_to_dir = cfg['cpu_to_dir']
    seq_dir    = cfg['seq_dir']
    max_len    = cfg['max_len']
    embed_dim  = cfg['embed_dim']

    df = pd.read_csv(cfg['csv_path'])
    print(f"CSV: {len(df)} rows  CPUs: {sorted(df['CPU'].unique())}")

    # vec cache: arch_dir → word2vec dict (loaded lazily)
    _vec_cache = {}
    def get_vec(path, arch_dir):
        if arch_dir not in _vec_cache:
            print(f"  Loading vec {arch_dir} from {path} ...")
            _vec_cache[arch_dir] = load_vec(path)
        return _vec_cache[arch_dir]

    if mode == 'cross':
        pairs = cfg['cross_pairs']
        if args.pair:
            src, tgt = args.pair.split(',')
            pairs = [(src.strip(), tgt.strip())]

        for src_cpu, tgt_cpu in pairs:
            pc       = cfg['pair_cfg_fn'](src_cpu, tgt_cpu)
            sd       = cpu_to_dir[src_cpu]
            td       = cpu_to_dir[tgt_cpu]
            tag      = f"{sd}_{td}"
            vec_src  = get_vec(pc['vec_src'], sd)
            vec_tgt  = get_vec(pc['vec_tgt'], td)
            vec_map  = {sd: vec_src, td: vec_tgt}

            print(f"\n[{tag}]  src={src_cpu}  tgt={tgt_cpu}")
            embed_cpus(df, [src_cpu, tgt_cpu], vec_map,
                       seq_dir, cpu_to_dir, pc['embed_cache_dir'],
                       max_len, embed_dim, args.force)

    else:  # mono
        cpus = cfg['mono_cpus']
        if args.cpu:
            cpus = [args.cpu]

        for cpu in cpus:
            mc       = cfg['mono_cfg_fn'](cpu)
            arch_dir = cpu_to_dir[cpu]
            vec_map  = {arch_dir: get_vec(mc['vec_src'], arch_dir)}

            print(f"\n[mono_{arch_dir}]  cpu={cpu}")
            embed_cpus(df, [cpu], vec_map,
                       seq_dir, cpu_to_dir, mc['embed_cache_dir'],
                       max_len, embed_dim, args.force)


if __name__ == '__main__':
    main()
