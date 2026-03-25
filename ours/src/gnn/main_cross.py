"""
main_cross.py
跨架構交叉實驗：依 configs/gnn/gnn_cross.py 中 arch_experiments 定義的
source -> targets 組合輪流跑（x86_64 已跑過，不重複）。

Log output: ours/outputs/logs/gnn/{embedding}/{model_type}/gnn_cross.log
"""

import copy
import sys
import os
import logging
from datetime import datetime

import torch
import numpy as np
import random
from collections import Counter
from torch_geometric.loader import DataLoader

from src.gnn.utils import (
    load_cross_arch_data, train_epoch, evaluate,
    create_gnn_scheduler, test_model, plot_training_curves,
    save_experiment_results, load_test_data_by_arch
)
from src.gnn.models import GCN, GAT
from configs.gnn.gnn_cross import get_gnn_cross_config


# ──────────────────────────────────────────────────────────────
# Logger: 同時輸出到 console 與 log 檔
# ──────────────────────────────────────────────────────────────
def setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("gnn_cross")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(message)s")

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def log(logger, msg=""):
    logger.info(msg)


# ──────────────────────────────────────────────────────────────
# Utilities (same as main.py)
# ──────────────────────────────────────────────────────────────
def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(config, num_classes, device):
    common_args = dict(
        num_node_features=config["num_node_features"],
        hidden_channels=config["hidden_channels"],
        output_channels=config["output_channels"],
        num_classes=num_classes,
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        pooling=config["pooling"],
    )
    if config["model_type"] == "GAT":
        model = GAT(**common_args, heads=config["gat_heads"])
    else:
        model = GCN(**common_args)
    return model.to(device)


def build_scheduler(optimizer, config):
    stype = config["scheduler_type"]
    if stype == "step":
        return create_gnn_scheduler(optimizer, stype,
                                    step_size=config["step_size"],
                                    gamma=config["gamma"])
    elif stype == "plateau":
        return create_gnn_scheduler(optimizer, stype,
                                    patience=config["plateau_patience"],
                                    factor=config["plateau_factor"])
    elif stype == "cosine":
        return create_gnn_scheduler(optimizer, stype,
                                    T_max=config["cosine_T_max"])
    else:
        raise ValueError(f"Unknown scheduler type: {stype}")


# ──────────────────────────────────────────────────────────────
# Single seed experiment (cross-arch only)
# ──────────────────────────────────────────────────────────────
def run_experiment(seed, config, logger):
    set_random_seed(seed)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    os.makedirs(config["model_output_dir"], exist_ok=True)

    batch_size = config["batch_size"]
    epochs     = config["epochs"]
    patience   = config["patience"]

    # --- Load training data (source arch) ---
    train_graphs, val_graphs, label_encoder, num_classes = load_cross_arch_data(
        csv_path=config["csv_path"],
        graph_dir=config["graph_dir"],
        source_cpus=config["source_cpus"],
        target_cpus=config["target_cpus"],
        cache_file=config["cache_file"],
        val_size=config["cross_arch_val_size"],
        random_state=config["random_state"],
        force_reload=config["force_reload"],
        classification=config["classification"],
    )

    train_label_counts = Counter(int(g.y) for g in train_graphs)
    val_label_counts   = Counter(int(g.y) for g in val_graphs)
    log(logger, f"  Train: {len(train_graphs)} samples  {dict(sorted(train_label_counts.items()))}")
    log(logger, f"  Val  : {len(val_graphs)} samples  {dict(sorted(val_label_counts.items()))}")

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True,
                              num_workers=config["num_workers"], pin_memory=config["pin_memory"])
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size, shuffle=False,
                              num_workers=config["num_workers"], pin_memory=config["pin_memory"])

    # --- Model ---
    model     = build_model(config, num_classes, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    if config["classification"]:
        class_counts  = torch.zeros(num_classes)
        for g in train_graphs:
            class_counts[int(g.y)] += 1
        class_weights = (len(train_graphs) / (num_classes * class_counts)).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    scheduler = build_scheduler(optimizer, config)

    # --- Training ---
    train_losses, val_losses_list, val_accuracies = [], [], []
    best_val_loss   = float("inf")
    best_model_state = None
    patience_counter = 0

    log(logger, f"  {'Epoch':>6}  {'ValAcc':>8}  {'ValLoss':>9}")
    log(logger, f"  {'─'*30}")

    for epoch in range(1, epochs + 1):
        train_loss  = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, val_loss = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses_list.append(val_loss)
        val_accuracies.append(val_acc)

        if config["scheduler_type"] == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            log(logger, f"  Early stopping at epoch {epoch}")
            break

        log(logger, f"  {epoch:>6}  {val_acc:>8.4f}  {val_loss:>9.4f}")

    model.load_state_dict(best_model_state)

    # --- Save model ---
    source_str   = "_".join(config["source_cpus"])
    model_fname  = f"gnn_cross_{source_str}_seed{seed}.pt"
    model_path   = os.path.join(config["model_output_dir"], model_fname)
    torch.save({
        "seed": seed, "model_state_dict": best_model_state,
        "config": config, "num_classes": num_classes,
        "label_encoder": label_encoder, "best_val_loss": best_val_loss,
    }, model_path)
    log(logger, f"  Model saved: {model_path}")

    # --- Plot ---
    plot_training_curves(train_losses, val_losses_list, val_accuracies, seed,
                         save_dir=config["plot_dir"])

    # --- Test per target arch ---
    test_graphs_by_arch = load_test_data_by_arch(
        config["csv_path"], config["graph_dir"],
        config["target_cpus"], label_encoder, config["classification"],
        cache_file=config.get("test_cache_file"),
        force_reload=config["force_reload"],
    )

    results = {}
    for cpu, graphs in test_graphs_by_arch.items():
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
        r = test_model(model, loader, device, label_encoder)
        results[cpu] = r
        log(logger, f"  [{cpu}]  Acc={r['accuracy']:.4f}  "
                    f"P={r['precision']:.4f}  R={r['recall']:.4f}  "
                    f"F1m={r['f1_micro']:.4f}  F1M={r['f1_macro']:.4f}  "
                    f"AUC={r['auc']:.4f}")
    return results


# ──────────────────────────────────────────────────────────────
# Aggregate helpers
# ──────────────────────────────────────────────────────────────
def aggregate(all_results, target_cpus):
    summary = {}
    for cpu in target_cpus:
        runs = [r[cpu] for r in all_results if cpu in r]
        if not runs:
            continue
        summary[cpu] = {
            k: {"mean": float(np.mean([r[k] for r in runs])),
                "std":  float(np.std( [r[k] for r in runs]))}
            for k in ["accuracy", "precision", "recall", "f1_micro", "f1_macro", "auc"]
        }
    return summary


# ──────────────────────────────────────────────────────────────
# One source arch: run all seeds, log, aggregate
# ──────────────────────────────────────────────────────────────
def run_source(source_arch, arch_info, base_config, logger):
    EMBEDDING = os.path.basename(base_config["graph_dir"])
    BASE_PATH = "/home/tommy/Project/PCBSDA"

    config = copy.deepcopy(base_config)
    config["source_cpus"]     = [source_arch]
    config["target_cpus"]     = arch_info["targets"]
    config["cache_file"]      = f"{BASE_PATH}/ours/outputs/cache/gnn/{EMBEDDING}/gnn_{source_arch}.pkl"
    config["test_cache_file"] = f"{BASE_PATH}/ours/outputs/cache/gnn/{EMBEDDING}/gnn_{source_arch}_test.pkl"

    seeds = config["seeds"]
    csv_name = os.path.basename(config["csv_path"])
    log(logger)
    log(logger, "=" * 70)
    log(logger, f"  SOURCE: {source_arch}  ->  TARGET: {arch_info['targets']}")
    log(logger, f"  CSV: {csv_name}")
    log(logger, "=" * 70)

    all_results = []
    for i, seed in enumerate(seeds):
        log(logger, f"\n  -- Seed {seed} ({i+1}/{len(seeds)}) --")
        r = run_experiment(seed, config, logger)
        all_results.append(r)

    summary = aggregate(all_results, arch_info["targets"])

    log(logger)
    log(logger, f"  >> {source_arch} Summary ({len(seeds)} seeds)")
    log(logger, f"  {'Target':>10}  {'Acc':>8}  {'±':>7}  {'F1-mac':>8}  {'±':>7}  {'AUC':>8}  {'±':>7}")
    log(logger, f"  {'─'*65}")
    for cpu, m in summary.items():
        log(logger,
            f"  {cpu:>10}  "
            f"{m['accuracy']['mean']:>8.4f}  {m['accuracy']['std']:>7.4f}  "
            f"{m['f1_macro']['mean']:>8.4f}  {m['f1_macro']['std']:>7.4f}  "
            f"{m['auc']['mean']:>8.4f}  {m['auc']['std']:>7.4f}")

    return summary, all_results


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    base_config = get_gnn_cross_config()
    arch_experiments = base_config["arch_experiments"]

    log_path = os.path.join(base_config["log_dir"], "gnn_cross.log")
    logger   = setup_logger(log_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log(logger, "=" * 70)
    log(logger, "=== GNN Cross-Architecture Experiment ===")
    log(logger, f"Timestamp : {timestamp}")
    log(logger, f"Embedding : {os.path.basename(base_config['graph_dir'])}")
    log(logger, f"Model     : {base_config['model_type']}  |  "
                f"Pooling: {base_config['pooling']}  |  "
                f"Layers: {base_config['num_layers']}")
    log(logger, f"Dims      : {base_config['num_node_features']} -> "
                f"{base_config['hidden_channels']} -> "
                f"{base_config['output_channels']}")
    log(logger, f"Scheduler : {base_config['scheduler_type']}")
    log(logger, f"Seeds     : {base_config['seeds']}")
    log(logger, f"CSV       : {os.path.basename(base_config['csv_path'])}")
    log(logger, f"Sources   : {list(arch_experiments.keys())}")
    log(logger, f"MIPS      : target only")
    log(logger, "=" * 70)

    global_summary = {}

    for source_arch, arch_info in arch_experiments.items():
        summary, _ = run_source(source_arch, arch_info, base_config, logger)
        global_summary[source_arch] = summary

    # ── Final cross table ──────────────────────────────────────
    log(logger)
    log(logger, "=" * 70)
    log(logger, "=== FINAL CROSS-ARCHITECTURE SUMMARY (F1-macro) ===")
    log(logger, "=" * 70)
    all_targets = sorted({t for v in arch_experiments.values() for t in v["targets"]})
    header = f"{'Source':>10}" + "".join(f"  {t:>10}" for t in all_targets)
    log(logger, header)
    log(logger, "  " + "─" * (10 + 13 * len(all_targets)))
    for src, tgt_dict in global_summary.items():
        row = f"{src:>10}"
        for tgt in all_targets:
            if tgt in tgt_dict:
                row += f"  {tgt_dict[tgt]['f1_macro']['mean']:>10.4f}"
            else:
                row += f"  {'—':>10}"
        log(logger, row)

    log(logger)
    log(logger, f"Log saved to: {log_path}")

    # ── Save JSON summary ──────────────────────────────────────
    results_for_save = {
        "timestamp": timestamp,
        "embedding": os.path.basename(base_config["graph_dir"]),
        "model_type": base_config["model_type"],
        "pooling": base_config["pooling"],
        "seeds": base_config["seeds"],
        "cross_results": {
            src: {tgt: m for tgt, m in tgt_dict.items()}
            for src, tgt_dict in global_summary.items()
        }
    }
    save_experiment_results(results_for_save, save_dir=base_config["result_dir"])
    log(logger, f"Results saved to: {base_config['result_dir']}")


if __name__ == "__main__":
    main()
