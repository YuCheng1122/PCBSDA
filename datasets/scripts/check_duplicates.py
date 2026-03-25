"""
check_duplicates.py

掃描 gpickle 目錄，找出圖結構與節點 tokens 完全相同的重複樣本。

使用 Weisfeiler-Lehman graph hash (networkx.weisfeiler_lehman_graph_hash)
將每個圖（含結構與節點 tokens）壓縮成一個 hash，相同 hash 即為重複。

用法：
    python check_duplicates.py --gpickle_dir <path> [--output <csv_path>] [--workers <n>]

預設 gpickle_dir: /home/tommy/Project/PCBSDA/ours/outputs/raw_data/gnn/gpickle
"""

import argparse
import pickle
import os
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import warnings
import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
from tqdm import tqdm


def _process_file(fp: Path) -> tuple[str, str] | tuple[None, str]:
    """Worker function: load gpickle and return (hash, filepath) or (None, error_msg)."""
    try:
        with open(fp, "rb") as f:
            G = pickle.load(f)

        H = nx.DiGraph()
        for node, data in G.nodes(data=True):
            tokens = data.get("tokens", [])
            H.add_node(node, label="|".join(tokens))
        for u, v in G.edges():
            H.add_edge(u, v)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h = weisfeiler_lehman_graph_hash(H, node_attr="label")
        return h, str(fp)
    except Exception as e:
        return None, f"{fp}: {e}"


def scan_gpickle_dir(gpickle_dir: Path, workers: int) -> dict[str, list[Path]]:
    files = list(gpickle_dir.rglob("*.gpickle"))
    print(f"找到 {len(files)} 個 gpickle 檔案，使用 {workers} 個 worker 計算 hash...")

    hash_to_files: dict[str, list[Path]] = defaultdict(list)
    errors = []

    with Pool(processes=workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(_process_file, files, chunksize=32),
            total=len(files),
            desc="Hashing",
            unit="file",
        ))

    for h, info in results:
        if h is None:
            errors.append(info)
        else:
            hash_to_files[h].append(Path(info))

    if errors:
        print(f"\n載入失敗 {len(errors)} 個檔案:")
        for err in errors[:10]:
            print(f"  {err}")

    return hash_to_files


def report(hash_to_files: dict[str, list[Path]], output_csv: Path | None = None):
    total = sum(len(v) for v in hash_to_files.values())
    unique = len(hash_to_files)
    dup_groups = {h: v for h, v in hash_to_files.items() if len(v) > 1}
    dup_files = sum(len(v) for v in dup_groups.values())

    print("\n========== 重複檢測結果 ==========")
    print(f"總檔案數       : {total}")
    print(f"唯一圖數       : {unique}")
    print(f"重複組數       : {len(dup_groups)}")
    print(f"涉及重複的檔案 : {dup_files}  (其中多餘副本: {dup_files - len(dup_groups)})")
    print(f"重複率         : {(dup_files - len(dup_groups)) / total * 100:.2f}%")

    if dup_groups:
        print("\n重複組（最多顯示 10 組）:")
        for i, (h, paths) in enumerate(list(dup_groups.items())[:10]):
            print(f"\n  [Group {i+1}] hash={h[:16]}... ({len(paths)} 個檔案)")
            for p in paths:
                print(f"    {p}")

    if output_csv:
        import csv
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["hash", "file_path"])
            for h, paths in dup_groups.items():
                for p in paths:
                    writer.writerow([h, str(p)])
        print(f"\n重複清單已存至: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="檢測 gpickle 資料集中的重複圖")
    parser.add_argument(
        "--gpickle_dir",
        type=Path,
        default=Path("/home/tommy/Project/PCBSDA/ours/outputs/raw_data/gnn/gpickle"),
        help="gpickle 根目錄",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="輸出重複清單 CSV 路徑（可選）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, cpu_count() - 2),
        help="multiprocessing worker 數量（預設 cpu_count - 2）",
    )
    args = parser.parse_args()

    hash_to_files = scan_gpickle_dir(args.gpickle_dir, args.workers)
    report(hash_to_files, args.output)


if __name__ == "__main__":
    main()
