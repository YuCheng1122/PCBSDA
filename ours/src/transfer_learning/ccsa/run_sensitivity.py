"""
Sensitivity analysis for CCSA hyperparameters (alpha and margin m).

Experiment: x86_64 -> MIPS (fixed pair)
- Alpha sweep : fix m=0.5, vary alpha in {0.1, 0.2, 0.3, 0.4, 0.5, 0.7}
- Margin sweep: fix alpha=0.3, vary m in {0.1, 0.3, 0.5, 0.7, 1.0}
Each setting runs 10 seeds; results saved as JSON + summary CSV + line plots.
"""

import copy
import json
import logging
import os
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

from configs.ccsa.sensitivity_config import get_sensitivity_config
from src.transfer_learning.ccsa.models import GCN_CCSA, GAT_CCSA
from src.transfer_learning.ccsa.utils import (
    prepare_ccsa_data, CCSAPairDataset, ccsa_pair_collate_fn,
    train_ccsa_epoch, evaluate, test_model,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

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
    kwargs = dict(
        num_node_features=config["num_node_features"],
        hidden_channels=config["hidden_channels"],
        output_channels=config["output_channels"],
        num_classes=num_classes,
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        pooling=config["pooling"],
    )
    if config.get("model_type", "GCN") == "GAT":
        model = GAT_CCSA(**kwargs, heads=config.get("gat_heads", 4))
    else:
        model = GCN_CCSA(**kwargs)
    return model.to(config["device"])


# ── Single seed run ────────────────────────────────────────────────────────────

def run_one_seed(config, random_state, logger):
    src_train, src_val, tgt_train, tgt_test, label_encoder, num_classes = \
        prepare_ccsa_data(config, random_state)

    pair_dataset = CCSAPairDataset(
        src_train, tgt_train, neg_pair_ratio=config["neg_pair_ratio"]
    )
    collate_fn = partial(ccsa_pair_collate_fn,
                         source_graphs=src_train, target_graphs=tgt_train)
    pair_loader = DataLoader(
        pair_dataset, batch_size=config["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=config["pin_memory"]
    )
    val_loader = PyGDataLoader(
        src_val, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=config["pin_memory"]
    )
    tgt_test_loader = PyGDataLoader(
        tgt_test, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=config["pin_memory"]
    )

    model = build_model(config, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    ce_criterion = torch.nn.CrossEntropyLoss()
    device = config["device"]

    alpha   = config["alpha"]
    margin  = config["csa_margin"]
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    max_epochs = config["epochs"]
    for epoch in range(1, max_epochs + 1):
        train_loss, _, _ = train_ccsa_epoch(model, pair_loader, optimizer,
                                            ce_criterion, device, alpha, margin)
        _, val_loss = evaluate(model, val_loader, device)

        print(f"\r  seed={random_state}  epoch {epoch}/{max_epochs}"
              f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
              f"  patience={patience_counter}/{config['patience']}",
              end="", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            print()  # newline after \r
            logger.info(f"  seed={random_state} early stop at epoch {epoch}")
            break
    else:
        print()  # newline after \r

    model.load_state_dict(best_model_state)
    results = test_model(model, tgt_test_loader, device, label_encoder)
    logger.info(f"  seed={random_state}: f1={results['f1_macro']:.4f} "
                f"acc={results['accuracy']:.4f}")
    return results


# ── One hyperparameter setting ─────────────────────────────────────────────────

def run_setting(base_config, alpha, margin, seeds, logger, tag):
    cfg = copy.deepcopy(base_config)
    cfg["alpha"]      = alpha
    cfg["csa_margin"] = margin

    src_cpus = cfg["source_cpus"]
    tgt_cpus = cfg["target_cpus"]
    src_tag  = "+".join(src_cpus).replace("/", "_")
    tgt_tag  = "+".join(tgt_cpus).replace("/", "_")
    pair_tag = f"{src_tag}_to_{tgt_tag}"

    cfg["source_cache_file"] = f"{cfg['cache_dir']}/source_{src_tag}.pkl"
    cfg["target_cache_file"] = f"{cfg['cache_dir']}/target_{tgt_tag}.pkl"

    logger.info(f"\n[{tag}] alpha={alpha}, margin={margin}, seeds={seeds}")

    f1s, accs, aucs = [], [], []
    for rs in seeds:
        r = run_one_seed(cfg, rs, logger)
        f1s.append(r["f1_macro"])
        accs.append(r["accuracy"])
        aucs.append(r["auc"])

    result = {
        "tag": tag,
        "alpha": alpha,
        "margin": margin,
        "pair": pair_tag,
        "seeds": seeds,
        "mean_f1":  float(np.mean(f1s)),
        "std_f1":   float(np.std(f1s)),
        "mean_acc": float(np.mean(accs)),
        "std_acc":  float(np.std(accs)),
        "mean_auc": float(np.mean(aucs)),
        "std_auc":  float(np.std(aucs)),
        "per_seed": [{"seed": s, "f1": f, "acc": a, "auc": u}
                     for s, f, a, u in zip(seeds, f1s, accs, aucs)],
    }
    logger.info(f"  => mean_f1={result['mean_f1']:.4f} ± {result['std_f1']:.4f}")
    return result


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_sensitivity(records, sweep_key, sweep_values, fixed_label,
                     save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    means = [r["mean_f1"] for r in records]
    stds  = [r["std_f1"]  for r in records]
    xs    = sweep_values

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(xs, means, yerr=stds, marker="o", linewidth=2,
                capsize=4, color="steelblue")
    ax.set_xlabel(sweep_key, fontsize=12)
    ax.set_ylabel("F1-macro", fontsize=12)
    ax.set_title(f"Sensitivity to {sweep_key}  ({fixed_label})", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xticks(xs)
    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {path}")


# ── Per-pair sweep ────────────────────────────────────────────────────────────

def run_pair_sweep(base_cfg, src_cpus, tgt_cpus, seeds, device):
    cfg = copy.deepcopy(base_cfg)
    cfg["source_cpus"] = src_cpus
    cfg["target_cpus"] = tgt_cpus
    cfg["device"]      = str(device)

    src_tag  = "+".join(src_cpus).replace("/", "_")
    tgt_tag  = "+".join(tgt_cpus).replace("/", "_")
    pair_tag = f"{src_tag}_to_{tgt_tag}"

    cfg["source_cache_file"] = f"{cfg['cache_dir']}/source_{src_tag}.pkl"
    cfg["target_cache_file"] = f"{cfg['cache_dir']}/target_{tgt_tag}.pkl"

    # per-pair output dirs
    result_dir = os.path.join(base_cfg["result_dir"], pair_tag)
    plot_dir   = os.path.join(base_cfg["plot_dir"],   pair_tag)
    log_dir    = os.path.join(base_cfg["log_dir"],    pair_tag)
    os.makedirs(result_dir, exist_ok=True)

    logger = setup_logger(log_dir, f"sensitivity_{pair_tag}")
    logger.info("=" * 60)
    logger.info(f"CCSA Sensitivity Analysis  ({src_cpus} -> {tgt_cpus})")
    logger.info(f"Seeds: {seeds}")
    logger.info("=" * 60)

    fixed_margin = cfg["csa_margin"]
    fixed_alpha  = cfg["alpha"]

    # alpha sweep
    alpha_records = []
    for a in cfg["alpha_sweep"]:
        r = run_setting(cfg, alpha=a, margin=fixed_margin,
                        seeds=seeds, logger=logger,
                        tag=f"alpha={a}_m={fixed_margin}")
        alpha_records.append(r)

    # margin sweep
    margin_records = []
    for m in cfg["margin_sweep"]:
        r = run_setting(cfg, alpha=fixed_alpha, margin=m,
                        seeds=seeds, logger=logger,
                        tag=f"alpha={fixed_alpha}_m={m}")
        margin_records.append(r)

    # save JSON
    json_path = os.path.join(result_dir, "sensitivity_results.json")
    with open(json_path, "w") as f:
        json.dump({"alpha_sweep": alpha_records, "margin_sweep": margin_records}, f, indent=2)

    # save CSVs
    df_alpha = pd.DataFrame([
        {"pair": pair_tag, "alpha": r["alpha"],
         "mean_f1": r["mean_f1"], "std_f1": r["std_f1"]}
        for r in alpha_records
    ])
    df_margin = pd.DataFrame([
        {"pair": pair_tag, "margin": r["margin"],
         "mean_f1": r["mean_f1"], "std_f1": r["std_f1"]}
        for r in margin_records
    ])
    df_alpha.to_csv(os.path.join(result_dir, "sensitivity_alpha.csv"),  index=False)
    df_margin.to_csv(os.path.join(result_dir, "sensitivity_margin.csv"), index=False)

    # plots
    plot_sensitivity(alpha_records, sweep_key="α",
                     sweep_values=cfg["alpha_sweep"],
                     fixed_label=f"m={fixed_margin}  |  {pair_tag}",
                     save_dir=plot_dir,
                     filename="sensitivity_alpha.png")
    plot_sensitivity(margin_records, sweep_key="margin m",
                     sweep_values=cfg["margin_sweep"],
                     fixed_label=f"α={fixed_alpha}  |  {pair_tag}",
                     save_dir=plot_dir,
                     filename="sensitivity_margin.png")

    # console summary
    logger.info(f"\nAlpha sweep  (fixed m={fixed_margin})")
    logger.info(f"  {'alpha':>6}  {'mean_F1':>8}  {'std_F1':>8}")
    for r in alpha_records:
        logger.info(f"  {r['alpha']:>6.1f}  {r['mean_f1']:>8.4f}  {r['std_f1']:>8.4f}")
    logger.info(f"\nMargin sweep  (fixed α={fixed_alpha})")
    logger.info(f"  {'margin':>6}  {'mean_F1':>8}  {'std_F1':>8}")
    for r in margin_records:
        logger.info(f"  {r['margin']:>6.1f}  {r['mean_f1']:>8.4f}  {r['std_f1']:>8.4f}")

    return alpha_records, margin_records, pair_tag


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    cfg    = get_sensitivity_config()
    seeds  = cfg["random_states"]
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    pairs  = cfg.get("experiment_pairs", [(cfg["source_cpus"], cfg["target_cpus"])])

    os.makedirs(cfg["result_dir"], exist_ok=True)
    top_logger = setup_logger(cfg["log_dir"], "sensitivity_all")
    top_logger.info(f"Running sensitivity for {len(pairs)} pairs, seeds={seeds}")

    all_alpha, all_margin = [], []
    for src_cpus, tgt_cpus in pairs:
        src_tag  = "+".join(src_cpus).replace("/", "_")
        tgt_tag  = "+".join(tgt_cpus).replace("/", "_")
        pair_tag = f"{src_tag}_to_{tgt_tag}"
        json_path = os.path.join(cfg["result_dir"], pair_tag, "sensitivity_results.json")

        if os.path.exists(json_path):
            top_logger.info(f"\n[SKIP] {pair_tag} — results already exist at {json_path}")
            with open(json_path) as f:
                cached = json.load(f)
            all_alpha.extend(cached["alpha_sweep"])
            all_margin.extend(cached["margin_sweep"])
            continue

        top_logger.info(f"\n{'#'*60}")
        top_logger.info(f"# Pair: {src_cpus} -> {tgt_cpus}")
        top_logger.info(f"{'#'*60}")
        a_recs, m_recs, pair_tag = run_pair_sweep(cfg, src_cpus, tgt_cpus, seeds, device)
        all_alpha.extend(a_recs)
        all_margin.extend(m_recs)

    # ── Aggregate CSV across all pairs ────────────────────────────────────────
    df_all_alpha = pd.DataFrame([
        {"pair": r["pair"], "alpha": r["alpha"],
         "mean_f1": r["mean_f1"], "std_f1": r["std_f1"]}
        for r in all_alpha
    ])
    df_all_margin = pd.DataFrame([
        {"pair": r["pair"], "margin": r["margin"],
         "mean_f1": r["mean_f1"], "std_f1": r["std_f1"]}
        for r in all_margin
    ])

    # mean across pairs per hyperparameter value
    agg_alpha  = df_all_alpha.groupby("alpha")[["mean_f1","std_f1"]].mean().reset_index()
    agg_margin = df_all_margin.groupby("margin")[["mean_f1","std_f1"]].mean().reset_index()

    agg_alpha.to_csv(os.path.join(cfg["result_dir"], "sensitivity_alpha_agg.csv"),  index=False)
    agg_margin.to_csv(os.path.join(cfg["result_dir"], "sensitivity_margin_agg.csv"), index=False)

    top_logger.info("\n=== Aggregate alpha sweep (mean across all pairs) ===")
    top_logger.info(agg_alpha.to_string(index=False))
    top_logger.info("\n=== Aggregate margin sweep (mean across all pairs) ===")
    top_logger.info(agg_margin.to_string(index=False))

    # aggregate plots
    alpha_agg_records = [
        {"alpha": row.alpha, "mean_f1": row.mean_f1, "std_f1": row.std_f1}
        for row in agg_alpha.itertuples()
    ]
    margin_agg_records = [
        {"margin": row.margin, "mean_f1": row.mean_f1, "std_f1": row.std_f1}
        for row in agg_margin.itertuples()
    ]
    plot_sensitivity(alpha_agg_records, sweep_key="α",
                     sweep_values=list(agg_alpha["alpha"]),
                     fixed_label=f"m={cfg['csa_margin']}  |  avg {len(pairs)} pairs",
                     save_dir=cfg["plot_dir"],
                     filename="sensitivity_alpha_agg.png")
    plot_sensitivity(margin_agg_records, sweep_key="margin m",
                     sweep_values=list(agg_margin["margin"]),
                     fixed_label=f"α={cfg['alpha']}  |  avg {len(pairs)} pairs",
                     save_dir=cfg["plot_dir"],
                     filename="sensitivity_margin_agg.png")


if __name__ == "__main__":
    main()
