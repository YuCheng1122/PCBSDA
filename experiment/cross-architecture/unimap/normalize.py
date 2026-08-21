"""
normalize.py  —  normalize raw IDA-disassembly corpora for fastText training.

Usage:
    python normalize.py --arch x86_64
    python normalize.py --arch arm_32
    python normalize.py --arch mips_32
    python normalize.py --all
    python normalize.py --all --workers 8
"""

import argparse
import os
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get('PCBSDA_UNIMAP_DATA_ROOT', PROJECT_ROOT / 'experiment' / 'outputs' / 'raw_data' / 'cross-architecture' / 'unimap'))
CORPUS_DIR = DATA_ROOT / 'corpus'
OUTPUT_DIR = DATA_ROOT / 'corpus_norm'

ARCH_CONFIG = {
    'x86_64': {
        'input':  CORPUS_DIR / 'x86_64_corpus.txt',
        'output': OUTPUT_DIR / 'x86_64_corpus_norm.txt',
        'module': 'format.format_x86_64',
        'func':   'normalize_line',
    },
    'arm_32': {
        'input':  CORPUS_DIR / 'arm_32_corpus.txt',
        'output': OUTPUT_DIR / 'arm_32_corpus_norm.txt',
        'module': 'format.format_arm_32',
        'func':   'normalize_line',
    },
    'mips_32': {
        'input':  CORPUS_DIR / 'mips_32_corpus.txt',
        'output': OUTPUT_DIR / 'mips_32_corpus_norm.txt',
        'module': 'format.format_mips_32',
        'func':   'normalize_line',
    },
}

CHUNK_SIZE = 8192  # lines per worker task

# ── worker (module-level so it is picklable) ──────────────────────────────────

_normalize_fn = None  # set in each worker process


def _worker_init(module_name: str, func_name: str) -> None:
    global _normalize_fn
    import importlib
    mod = importlib.import_module(module_name)
    _normalize_fn = getattr(mod, func_name)


def _process_chunk(lines: list) -> list:
    out = []
    for line in lines:
        result = _normalize_fn(line)
        if result:
            out.append(result)
    return out


# ── core ──────────────────────────────────────────────────────────────────────

def count_lines(path: Path) -> int:
    count = 0
    with open(path, 'rb') as f:
        for _ in f:
            count += 1
    return count


def normalize_arch(arch: str, workers: int) -> None:
    cfg = ARCH_CONFIG[arch]
    src  = cfg['input']
    dst  = cfg['output']
    module = cfg['module']
    func   = cfg['func']

    if not src.exists():
        print(f'[{arch}] Input not found: {src}', file=sys.stderr)
        return

    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f'[{arch}] Counting lines ...')
    total = count_lines(src)
    print(f'[{arch}] {total:,} lines  ->  {dst}')

    pool = Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=(module, func),
    )

    written = 0

    with open(src, encoding='utf-8', errors='replace') as fin, \
         open(dst, 'w', encoding='utf-8') as fout, \
         tqdm(total=total, desc=arch, unit='lines', dynamic_ncols=True) as pbar:

        chunk = []
        futures = []

        def flush_chunk():
            nonlocal chunk
            if chunk:
                futures.append(pool.apply_async(_process_chunk, (chunk,)))
                chunk = []

        def drain_futures(drain_all=False):
            nonlocal written
            limit = 0 if drain_all else max(0, len(futures) - workers * 2)
            while len(futures) > limit:
                result_lines = futures.pop(0).get()
                for r in result_lines:
                    fout.write(r + '\n')
                written += len(result_lines)

        lines_read = 0
        for line in fin:
            chunk.append(line)
            lines_read += 1
            if len(chunk) >= CHUNK_SIZE:
                flush_chunk()
                drain_futures()
                pbar.update(CHUNK_SIZE)

        flush_chunk()
        drain_futures(drain_all=True)
        pbar.update(lines_read % CHUNK_SIZE)

    pool.close()
    pool.join()
    print(f'[{arch}] Done — {written:,} sentences written.')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Normalize disassembly corpora for fastText')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--arch', choices=list(ARCH_CONFIG), help='Single architecture to process')
    group.add_argument('--all', action='store_true', help='Process all architectures sequentially')
    parser.add_argument(
        '--workers', type=int, default=max(1, cpu_count() - 1),
        help='Number of worker processes (default: nCPU-1)',
    )
    args = parser.parse_args()

    archs = list(ARCH_CONFIG) if args.all else [args.arch]

    for arch in archs:
        normalize_arch(arch, args.workers)


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).parent))
    main()
