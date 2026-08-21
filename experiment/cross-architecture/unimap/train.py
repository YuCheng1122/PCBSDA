"""
train.py — multi-class malware family classification via UniMap MAIE

Usage:
    python train.py                   # use mode from config.py
    python train.py --mode cross      # cross-architecture (all pairs)
    python train.py --mode mono       # single-architecture (all CPUs)
    python train.py --mode cross --pair x86_64,ARM-32   # single pair
    python train.py --mode mono  --cpu x86_64           # single CPU
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from config import get_lstm_config
from model import PaperLSTM


# ── Dataset ───────────────────────────────────────────────────────────────────

class SeqDataset(Dataset):
    def __init__(self, records: list):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        x = np.load(path).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)


# ── Helpers ───────────────────────────────────────────────────────────────────

def assign_labels(df, families):
    fam2id = {f: i for i, f in enumerate(families)}
    df = df[df['family'].isin(families)].copy()
    df['label'] = df['family'].map(fam2id)
    return df


def build_records(df, cache_dir, cpu_list):
    records = []
    cache = Path(cache_dir)
    for _, row in df.iterrows():
        if row['CPU'] not in cpu_list:
            continue
        p = cache / f"{row['file_name']}.npy"
        if p.exists():
            records.append((str(p), int(row['label'])))
    return records


def compute_val_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total_loss += criterion(model(x), y).item()
    return total_loss / len(loader)


def evaluate(model, loader, device, n_classes):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            all_logits.append(model(x).cpu().numpy())
            all_labels.extend(y.numpy())

    logits = np.vstack(all_logits)
    probs  = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    preds  = logits.argmax(axis=1)
    labels = np.array(all_labels)

    metrics = {
        'acc': accuracy_score(labels, preds),
        'f1':  f1_score(labels, preds, average='macro', zero_division=0),
        '_preds':  preds,
        '_probs':  probs,
        '_labels': labels,
    }
    try:
        metrics['auc'] = roc_auc_score(
            labels, probs,
            multi_class='ovr', average='macro'
        ) if n_classes > 2 else roc_auc_score(labels, probs[:, 1])
    except Exception:
        metrics['auc'] = float('nan')
    return metrics


def print_report(metrics, families, title=''):
    if title:
        print(f"\n{'─'*55}\n{title}")
    print(f"  Accuracy : {metrics['acc']:.4f}")
    print(f"  F1 macro : {metrics['f1']:.4f}")
    print(f"  AUC      : {metrics['auc']:.4f}")
    print(classification_report(
        metrics['_labels'], metrics['_preds'],
        target_names=families, digits=4, zero_division=0,
    ))


def save_seed_log(metrics, families, log_path: Path, seed: int,
                  epoch_log: list, tag: str):
    """Write per-seed detailed log: epoch table + classification report."""
    with open(log_path, 'w') as f:
        f.write(f"tag={tag}  seed={seed}\n")
        f.write(f"acc={metrics['acc']:.6f}  f1={metrics['f1']:.6f}  "
                f"auc={metrics['auc']:.6f}\n\n")

        # epoch table
        f.write(f"{'epoch':>6}  {'train_loss':>10}  {'val_loss':>8}  "
                f"{'val_acc':>7}  {'val_f1':>6}  {'val_auc':>7}\n")
        f.write('─' * 58 + '\n')
        for row in epoch_log:
            f.write(f"{row['epoch']:>6}  {row['train_loss']:>10.4f}  "
                    f"{row['val_loss']:>8.4f}  {row['val_acc']:>7.4f}  "
                    f"{row['val_f1']:>6.4f}  {row['val_auc']:>7.4f}\n")

        f.write('\n')
        f.write(classification_report(
            metrics['_labels'], metrics['_preds'],
            target_names=families, digits=4, zero_division=0,
        ))


def save_results(all_seed_metrics: list, families: list,
                 out_dir: Path, tag: str, mode: str,
                 src_cpu: str, tgt_cpu: str | None, seeds: list):
    """Write results_<ts>.csv and summary_<ts>.json, matching FCGAT format."""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir.mkdir(parents=True, exist_ok=True)

    # per-seed rows for CSV
    rows = []
    for seed, m in zip(seeds, all_seed_metrics):
        rows.append({
            'tag':      tag,
            'seed':     seed,
            'accuracy': round(m['acc'], 6),
            'f1_macro': round(m['f1'],  6),
            'auc':      round(m['auc'], 6),
        })
    pd.DataFrame(rows).to_csv(out_dir / f'results_{ts}.csv', index=False)

    # aggregate summary
    accs = [r['accuracy'] for r in rows]
    f1s  = [r['f1_macro'] for r in rows]
    aucs = [r['auc']      for r in rows]

    summary = {
        'tag':         tag,
        'mode':        mode,
        'source_cpu':  src_cpu,
        'target_cpu':  tgt_cpu,
        'seeds':       seeds,
        'families':    families,
        'avg_accuracy': float(np.mean(accs)),
        'std_accuracy': float(np.std(accs)),
        'avg_f1_macro': float(np.mean(f1s)),
        'std_f1_macro': float(np.std(f1s)),
        'avg_auc':      float(np.mean(aucs)),
        'std_auc':      float(np.std(aucs)),
        'all_results':  rows,
    }
    with open(out_dir / f'summary_{ts}.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


# ── Training loop (single seed) ───────────────────────────────────────────────

def run(cfg, seed, src_cpu, tgt_cpu, pair_cfg, tag):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')

    families  = cfg['families']
    n_classes = len(families)
    is_cross  = tgt_cpu is not None

    df = pd.read_csv(cfg['csv_path'])
    df = assign_labels(df, families)

    cache_dir = pair_cfg['embed_cache_dir']

    if is_cross:
        src_df = df[df['CPU'] == src_cpu]
        tgt_df = df[df['CPU'] == tgt_cpu]
        src_train, src_val = train_test_split(
            src_df, test_size=cfg['val_size'], random_state=seed, stratify=src_df['label']
        )
        train_rec = build_records(src_train, cache_dir, [src_cpu])
        val_rec   = build_records(src_val,   cache_dir, [src_cpu])
        test_rec  = build_records(tgt_df,    cache_dir, [tgt_cpu])
    else:
        cpu_df = df[df['CPU'] == src_cpu]
        train_val, test_df = train_test_split(
            cpu_df, test_size=cfg['test_size'], random_state=seed, stratify=cpu_df['label']
        )
        train_df, val_df = train_test_split(
            train_val, test_size=cfg['val_size'], random_state=seed, stratify=train_val['label']
        )
        train_rec = build_records(train_df, cache_dir, [src_cpu])
        val_rec   = build_records(val_df,   cache_dir, [src_cpu])
        test_rec  = build_records(test_df,  cache_dir, [src_cpu])

    if not train_rec or not val_rec or not test_rec:
        raise RuntimeError(
            f"[{tag} seed={seed}] Empty split: "
            f"train={len(train_rec)} val={len(val_rec)} test={len(test_rec)}\n"
            f"  embed_cache_dir={cache_dir}\n"
            f"  Run embedSequences.py first."
        )

    print(f"  train={len(train_rec)}  val={len(val_rec)}  test={len(test_rec)}  device={device}")

    train_loader = DataLoader(SeqDataset(train_rec), batch_size=cfg['batch_size'],
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(SeqDataset(val_rec),   batch_size=cfg['batch_size'],
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(SeqDataset(test_rec),  batch_size=cfg['batch_size'],
                              shuffle=False, num_workers=4, pin_memory=True)

    model     = PaperLSTM(input_dim=cfg['embed_dim'], use_embedding=False,
                          n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(pair_cfg['model_output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1  = -1.0
    patience_cnt = 0
    epoch_log    = []

    print(f"\n{'Epoch':>6}  {'train_loss':>10}  {'val_loss':>8}  "
          f"{'val_acc':>7}  {'val_f1':>6}  {'val_auc':>7}  {'patience':>8}")
    print('─' * 72)

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss   = compute_val_loss(model, val_loader, criterion, device)
        val_m      = evaluate(model, val_loader, device, n_classes)

        epoch_log.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_loss': val_loss, 'val_acc': val_m['acc'],
            'val_f1': val_m['f1'], 'val_auc': val_m['auc'],
        })

        if val_m['f1'] > best_val_f1:
            best_val_f1  = val_m['f1']
            patience_cnt = 0
            torch.save(model.state_dict(), out_dir / f'best_model_seed{seed}.pt')
            marker = ' *'
        else:
            patience_cnt += 1
            marker = ''

        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>8.4f}  "
              f"{val_m['acc']:>7.4f}  {val_m['f1']:>6.4f}  {val_m['auc']:>7.4f}  "
              f"{patience_cnt:>8}{marker}")

        if patience_cnt >= cfg['patience']:
            print(f"\n  Early stop at epoch {epoch}  (best val_f1={best_val_f1:.4f})")
            break

    model.load_state_dict(torch.load(
        out_dir / f'best_model_seed{seed}.pt', map_location=device, weights_only=True
    ))
    test_m = evaluate(model, test_loader, device, n_classes)

    test_title = f"[{tag} seed={seed}] {'TARGET' if is_cross else 'TEST'} " \
                 f"({tgt_cpu or src_cpu}) Report"
    print_report(test_m, families, title=test_title)

    # per-seed log
    log_path = out_dir / f'log_seed{seed}.txt'
    save_seed_log(test_m, families, log_path, seed, epoch_log, tag)

    return test_m


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['cross', 'mono'], default=None)
    parser.add_argument('--pair', default=None, help='e.g. x86_64,ARM-32')
    parser.add_argument('--cpu',  default=None, help='e.g. x86_64 (mono only)')
    args = parser.parse_args()

    cfg   = get_lstm_config()
    seeds = cfg['seeds']
    mode  = args.mode or cfg['mode']

    if mode == 'cross':
        pairs = cfg['cross_pairs']
        if args.pair:
            src, tgt = args.pair.split(',')
            pairs = [(src.strip(), tgt.strip())]

        print(f"\n{'='*60}")
        print(f"Mode: cross-architecture  |  seeds={seeds}")
        print(f"Pairs: {pairs}")
        print(f"{'='*60}")

        all_pair_summaries = {}
        for src_cpu, tgt_cpu in pairs:
            pc  = cfg['pair_cfg_fn'](src_cpu, tgt_cpu)
            sd  = cfg['cpu_to_dir'][src_cpu]
            td  = cfg['cpu_to_dir'][tgt_cpu]
            tag = f"{sd}→{td}"
            print(f"\n{'─'*55}\n{tag}\n{'─'*55}")

            seed_metrics = []
            for seed in seeds:
                print(f"\n  [seed={seed}]")
                m = run(cfg, seed, src_cpu, tgt_cpu, pc, tag)
                seed_metrics.append(m)

            out_dir = Path(pc['model_output_dir'])
            summary = save_results(seed_metrics, cfg['families'],
                                   out_dir, tag, mode, src_cpu, tgt_cpu, seeds)
            all_pair_summaries[tag] = summary

        print(f"\n{'='*60}\nSummary — cross-architecture  seeds={seeds}")
        print(f"{'Pair':<20}  {'Acc':>12}  {'F1':>12}  {'AUC':>12}")
        print('─' * 62)
        for tag, s in all_pair_summaries.items():
            print(f"{tag:<20}  "
                  f"{s['avg_accuracy']:.4f}±{s['std_accuracy']:.4f}  "
                  f"{s['avg_f1_macro']:.4f}±{s['std_f1_macro']:.4f}  "
                  f"{s['avg_auc']:.4f}±{s['std_auc']:.4f}")

    else:  # mono
        cpus = cfg['mono_cpus']
        if args.cpu:
            cpus = [args.cpu]

        print(f"\n{'='*60}")
        print(f"Mode: mono-architecture  |  seeds={seeds}")
        print(f"CPUs: {cpus}")
        print(f"{'='*60}")

        all_cpu_summaries = {}
        for cpu in cpus:
            mc  = cfg['mono_cfg_fn'](cpu)
            tag = f"mono_{cfg['cpu_to_dir'][cpu]}"
            print(f"\n{'─'*55}\n{tag}\n{'─'*55}")

            seed_metrics = []
            for seed in seeds:
                print(f"\n  [seed={seed}]")
                m = run(cfg, seed, cpu, None, mc, tag)
                seed_metrics.append(m)

            out_dir = Path(mc['model_output_dir'])
            summary = save_results(seed_metrics, cfg['families'],
                                   out_dir, tag, mode, cpu, None, seeds)
            all_cpu_summaries[tag] = summary

        print(f"\n{'='*60}\nSummary — mono-architecture  seeds={seeds}")
        print(f"{'CPU':<20}  {'Acc':>12}  {'F1':>12}  {'AUC':>12}")
        print('─' * 62)
        for tag, s in all_cpu_summaries.items():
            print(f"{tag:<20}  "
                  f"{s['avg_accuracy']:.4f}±{s['std_accuracy']:.4f}  "
                  f"{s['avg_f1_macro']:.4f}±{s['std_f1_macro']:.4f}  "
                  f"{s['avg_auc']:.4f}±{s['std_auc']:.4f}")


if __name__ == '__main__':
    main()
