"""
Cross-architecture DANN experiments.

Experiment pairs:
  x86_64  -> Intel
  x86_64  -> MIPS
  ARM-32  -> x86_64
  ARM-32  -> MIPS
  ARM-32  -> Intel
  Intel   -> x86_64
  Intel   -> MIPS
  Intel   -> ARM-32

Each pair runs N random seeds. MIPS is test-only (never source).
Outputs are separated per (source, target) pair.
"""

import copy
import logging
import os

import numpy as np
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from configs.dann.run_all_config import get_run_all_config
from src.transfer_learning.dann.models import GCN_DANN, GAT_DANN
from src.transfer_learning.dann.utils import (
    prepare_dann_data, compute_alpha,
    train_dann_epoch, evaluate, test_model,
    plot_training_curves, save_experiment_results,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pair_tag(src_cpus, tgt_cpus):
    src = "+".join(src_cpus).replace("/", "_")
    tgt = "+".join(tgt_cpus).replace("/", "_")
    return f"{src}_to_{tgt}"


def setup_logger(log_dir, name):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def build_model(config, num_classes):
    model_type = config.get("model_type", "GCN")
    kwargs = dict(
        num_node_features=config["num_node_features"],
        hidden_channels=config["hidden_channels"],
        output_channels=config["output_channels"],
        num_classes=num_classes,
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        pooling=config["pooling"],
    )
    if model_type == "GAT":
        model = GAT_DANN(**kwargs, heads=config.get("gat_heads", 4))
    else:
        model = GCN_DANN(**kwargs)
    return model.to(config["device"])


# ── Single experiment run ──────────────────────────────────────────────────────

def run_experiment(config, random_state, dirs, logger):
    src_train, src_val, _, tgt_test, tgt_all, label_encoder, num_classes = \
        prepare_dann_data(config, random_state)

    device = config["device"]

    src_loader = PyGDataLoader(
        src_train, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"], pin_memory=config["pin_memory"]
    )
    # DANN domain alignment: 全量 target（unlabeled），class label 不使用
    tgt_domain_loader = PyGDataLoader(
        tgt_all, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"], pin_memory=config["pin_memory"]
    )
    val_loader = PyGDataLoader(
        src_val, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=config["pin_memory"]
    )
    target_test_loader = PyGDataLoader(
        tgt_test, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=config["pin_memory"]
    )
    model = build_model(config, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    ce_criterion = torch.nn.CrossEntropyLoss()
    domain_criterion = torch.nn.CrossEntropyLoss()

    lambda_domain = config["lambda_domain"]
    epochs = config["epochs"]
    patience = config["patience"]

    train_losses, val_losses_list, val_accuracies = [], [], []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    logger.info(f"\n--- Training (seed={random_state}) ---")
    header = (f"{'Epoch':>6}  {'Loss':>8}  {'CE_src':>8}  {'Dom_src':>8}  "
              f"{'Dom_tgt':>8}  {'Alpha':>7}  {'SrcValAcc':>10}")
    logger.info(header)
    logger.info("-" * len(header))

    for epoch in range(1, epochs + 1):
        alpha = compute_alpha(epoch, epochs)

        train_loss, ce_src, dom_src, dom_tgt = train_dann_epoch(
            model, src_loader, tgt_domain_loader, optimizer,
            ce_criterion, domain_criterion, device,
            alpha=alpha, lambda_domain=lambda_domain,
        )

        val_accuracy, val_loss = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses_list.append(val_loss)
        val_accuracies.append(val_accuracy)

        logger.info(
            f"{epoch:>6}  {train_loss:>8.4f}  {ce_src:>8.4f}  {dom_src:>8.4f}  "
            f"{dom_tgt:>8.4f}  {alpha:>7.4f}  {val_accuracy:>10.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_model_state)

    os.makedirs(dirs["model_output_dir"], exist_ok=True)
    save_path = os.path.join(dirs["model_output_dir"], f"dann_best_rs{random_state}.pt")
    torch.save(best_model_state, save_path)
    logger.info(f"Model saved to: {save_path}")

    plot_training_curves(train_losses, val_losses_list, val_accuracies,
                         save_dir=dirs["plot_dir"], random_state=random_state)

    logger.info(f"\n--- Final Evaluation (seed={random_state}) ---")
    target_results = test_model(model, target_test_loader, device, label_encoder)
    logger.info(f"Report:\n{target_results['classification_report']}")

    return target_results


# ── Per-pair orchestration ─────────────────────────────────────────────────────

def run_pair(base_config, src_cpus, tgt_cpus, seeds, device):
    tag = _pair_tag(src_cpus, tgt_cpus)

    cfg = copy.deepcopy(base_config)
    cfg["source_cpus"]   = src_cpus
    cfg["target_cpus"]   = tgt_cpus
    cfg["device"]        = device
    cfg["random_states"] = seeds

    # Per-pair cache derived from config's cache_dir
    cache_dir = cfg["cache_dir"]
    src_tag = "+".join(src_cpus).replace("/", "_")
    tgt_tag = "+".join(tgt_cpus).replace("/", "_")
    cfg["source_cache_file"] = f"{cache_dir}/source_{src_tag}.pkl"
    cfg["target_cache_file"] = f"{cache_dir}/target_{tgt_tag}.pkl"

    # Per-pair output dirs
    cfg["model_output_dir"] = f"{base_config['model_output_dir']}/{tag}"
    cfg["plot_dir"]         = f"{base_config['plot_dir']}/{tag}"
    cfg["result_dir"]       = f"{base_config['result_dir']}/{tag}"
    cfg["log_dir"]          = f"{base_config['log_dir']}/{tag}"

    dirs = {k: cfg[k] for k in ("model_output_dir", "plot_dir", "result_dir", "log_dir")}
    logger = setup_logger(cfg["log_dir"], name=f"dann_{tag}")

    mode = "Classification (family)" if cfg["classification"] else "Detection (label)"
    fewshot_desc = (f"{cfg['num_target_samples_per_class']} per class"
                    if "num_target_samples_per_class" in cfg
                    else f"{cfg.get('num_target_samples', '?')} total")

    logger.info(f"=== DANN | {src_cpus} -> {tgt_cpus} ===")
    logger.info(f"Mode: {mode}  |  Seeds: {seeds}")
    logger.info(f"Few-shot: {fewshot_desc}  |  Model: {cfg.get('model_type','GCN')}")
    logger.info(f"Lambda_domain: {cfg['lambda_domain']}")
    logger.info("")

    all_results = []
    for i, rs in enumerate(seeds):
        logger.info(f"\n{'='*60}")
        logger.info(f"Run {i+1}/{len(seeds)}, seed={rs}")
        logger.info(f"{'='*60}")
        results = run_experiment(cfg, rs, dirs, logger)
        all_results.append({'random_state': rs, **results})

    # ── Summary ──────────────────────────────────────────────────────────────
    accs       = [r['accuracy']  for r in all_results]
    f1_macros  = [r['f1_macro']  for r in all_results]
    aucs       = [r['auc']       for r in all_results]
    precisions = [r['precision'] for r in all_results]
    recalls    = [r['recall']    for r in all_results]

    logger.info(f"\n{'='*60}")
    logger.info(f"Summary ({len(seeds)} runs) | {src_cpus} -> {tgt_cpus}")
    logger.info(f"{'='*60}")
    logger.info(f"  Accuracy  : {np.mean(accs):.4f} +/- {np.std(accs):.4f}  "
                f"(min={np.min(accs):.4f}, max={np.max(accs):.4f})")
    logger.info(f"  Precision : {np.mean(precisions):.4f} +/- {np.std(precisions):.4f}")
    logger.info(f"  Recall    : {np.mean(recalls):.4f} +/- {np.std(recalls):.4f}")
    logger.info(f"  F1-macro  : {np.mean(f1_macros):.4f} +/- {np.std(f1_macros):.4f}")
    logger.info(f"  AUC       : {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    for r in all_results:
        logger.info(f"  seed={r['random_state']}: acc={r['accuracy']:.4f}, "
                    f"f1={r['f1_macro']:.4f}, auc={r['auc']:.4f}")

    results_summary = {
        'mode': mode,
        'source_cpus': src_cpus,
        'target_cpus': tgt_cpus,
        'target_fewshot': fewshot_desc,
        'lambda_domain': cfg['lambda_domain'],
        'model_type': cfg.get('model_type', 'GCN'),
        'pooling': cfg['pooling'],
        'mean_accuracy': float(np.mean(accs)),
        'std_accuracy':  float(np.std(accs)),
        'mean_f1_macro': float(np.mean(f1_macros)),
        'std_f1_macro':  float(np.std(f1_macros)),
        'mean_auc':      float(np.mean(aucs)),
        'std_auc':       float(np.std(aucs)),
        'per_run': [
            {'random_state': r['random_state'],
             'accuracy': r['accuracy'],
             'f1_macro': r['f1_macro'],
             'auc': r['auc']}
            for r in all_results
        ],
    }
    timestamp = save_experiment_results(results_summary, save_dir=dirs["result_dir"])
    logger.info(f"\nResults saved with timestamp: {timestamp}")

    return results_summary


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    base_config = get_run_all_config()

    seeds  = base_config["random_states"]
    pairs  = base_config["experiment_pairs"]
    device = torch.device(base_config["device"] if torch.cuda.is_available() else "cpu")

    print(f"\n{'#'*60}")
    print(f"# DANN Cross-Architecture — All Pairs")
    print(f"# Seeds: {seeds}")
    print(f"# Pairs:")
    for src, tgt in pairs:
        print(f"#   {src} -> {tgt}")
    print(f"# Device: {device}")
    print(f"{'#'*60}\n")

    all_summaries = []
    for src_cpus, tgt_cpus in pairs:
        print(f"\n{'='*60}")
        print(f"Starting pair: {src_cpus} -> {tgt_cpus}")
        print(f"{'='*60}")
        summary = run_pair(base_config, src_cpus, tgt_cpus, seeds, device)
        all_summaries.append(summary)

    # ── Global summary across all pairs ──────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"# Final Summary — All Pairs")
    print(f"{'#'*60}")
    print(f"  {'Source':<12} {'Target':<12} {'Acc':>8} {'±':>6} {'F1':>8} {'±':>6} {'AUC':>8} {'±':>6}")
    print(f"  {'-'*66}")
    for s in all_summaries:
        src = "+".join(s['source_cpus'])
        tgt = "+".join(s['target_cpus'])
        print(f"  {src:<12} {tgt:<12} "
              f"{s['mean_accuracy']:>8.4f} {s['std_accuracy']:>6.4f} "
              f"{s['mean_f1_macro']:>8.4f} {s['std_f1_macro']:>6.4f} "
              f"{s['mean_auc']:>8.4f} {s['std_auc']:>6.4f}")


if __name__ == "__main__":
    main()
