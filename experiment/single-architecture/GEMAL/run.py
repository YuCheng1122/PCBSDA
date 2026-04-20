"""
Single-Architecture Family Classification — GEMAL (GCN + attention readout)

Pipeline per arch (Nested CV):
  1. Outer 5-fold CV  — each fold splits all data into train (80%) and test (20%)
  2. Optuna search    — on outer train fold, 5-fold inner CV, maximise F1-macro
  3. Outer evaluation — retrain with best params on outer train fold, evaluate outer test fold
  4. Summary          — mean ± std across 5 outer folds

Run from PCBSDA root:
    python experiment/single-architecture/GEMAL/run.py
    python experiment/single-architecture/GEMAL/run.py --arch x86_64
    python experiment/single-architecture/GEMAL/run.py --arch x86_64 --tune-only
    python experiment/single-architecture/GEMAL/run.py --arch x86_64 --eval-only
"""

import sys
import os
import time


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import argparse
import copy
import random

import numpy as np
import torch
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch_geometric.loader import DataLoader
import pandas as pd

from config import get_gemal_single_config, ALL_ARCHS
from model import GEMAL
from ours.src.gnn.utils import (
    load_graphs_from_df,
    train_epoch,
    evaluate,
    test_model,
    create_gnn_scheduler,
    save_experiment_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(params: dict, num_node_features: int,
                num_classes: int, device: torch.device) -> GEMAL:
    model = GEMAL(
        num_node_features=num_node_features,
        hidden_channels=params["hidden_channels"],
        num_classes=num_classes,
        dropout=params["dropout"],
        embed_dim=params["embed_dim"],
    )
    return model.to(device)


def build_scheduler(optimizer, params: dict):
    return create_gnn_scheduler(
        optimizer, params["scheduler_type"],
        patience=params.get("plateau_patience", 5),
        factor=params.get("plateau_factor", 0.5),
        step_size=params.get("step_size", 30),
        gamma=params.get("gamma", 0.5),
        T_max=params.get("cosine_T_max", 70),
    )


def make_class_weights(graphs: list, num_classes: int,
                       device: torch.device) -> torch.Tensor:
    counts = torch.zeros(num_classes)
    for g in graphs:
        counts[int(g.y)] += 1
    weights = len(graphs) / (num_classes * counts)
    return weights.to(device)


def load_all_graphs(config: dict):
    df = pd.read_csv(config["csv_path"])
    arch_df = df[df["CPU"].isin(config["source_cpus"])]
    print(f"Total samples for {config['source_cpus']}: {len(arch_df)}")
    graphs, labels = load_graphs_from_df(arch_df, config["graph_dir"],
                                          classification=True)
    return graphs, labels


def encode_labels(graphs: list, labels: list):
    le = LabelEncoder()
    encoded = le.fit_transform(labels)
    for i, g in enumerate(graphs):
        g.y = torch.tensor(encoded[i], dtype=torch.long)
    return le, len(le.classes_)


# ---------------------------------------------------------------------------
# Single-fold training
# ---------------------------------------------------------------------------

def train_fold(train_graphs: list, val_graphs: list, params: dict,
               num_classes: int, config: dict, seed: int = 42,
               max_epochs: int = None):
    set_seed(seed)
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    batch_size   = params["batch_size"]
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True,
                              num_workers=config["num_workers"],
                              pin_memory=config["pin_memory"])
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size, shuffle=False,
                              num_workers=config["num_workers"],
                              pin_memory=config["pin_memory"])

    model        = build_model(params, config["num_node_features"], num_classes, device)
    optimizer    = torch.optim.Adam(model.parameters(),
                                    lr=params["learning_rate"],
                                    weight_decay=params.get("weight_decay", 0.0))
    scheduler    = build_scheduler(optimizer, {**config, **params})
    class_weights = make_class_weights(train_graphs, num_classes, device)
    criterion    = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss    = float("inf")
    best_model_state = None
    patience_counter = 0
    patience         = config["patience"]
    n_epochs         = max_epochs if max_epochs is not None else config["epochs"]

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, val_loss = evaluate(model, val_loader, device)
        epoch_time = time.time() - t0

        if params["scheduler_type"] == "plateau":
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
            break
    model.load_state_dict(best_model_state)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.arange(num_classes)
    val_results = test_model(model, val_loader, device, label_encoder)
    return val_results["f1_macro"], best_model_state


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(dev_graphs: list, dev_labels_encoded: np.ndarray,
                   num_classes: int, config: dict):
    skf      = StratifiedKFold(n_splits=config["optuna_n_splits"], shuffle=True,
                               random_state=config["random_state"])
    ss       = config["search_space"]
    n_trials = config["n_trials"]

    def objective(trial):
        params = {
            # Fixed
            "hidden_channels": config["hidden_channels"],
            "embed_dim":       config["embed_dim"],
            "dropout":         config["dropout"],
            "scheduler_type":  config["scheduler_type"],
            # Searched
            "batch_size":    trial.suggest_categorical("batch_size", ss["batch_size"]),
            "learning_rate": trial.suggest_float("learning_rate",
                                                  ss["learning_rate"][0],
                                                  ss["learning_rate"][1], log=True),
            "weight_decay":  trial.suggest_float("weight_decay",
                                                  ss["weight_decay"][0],
                                                  ss["weight_decay"][1], log=True),
        }

        print(f"\n[Trial {trial.number+1}/{n_trials}] "
              f"lr={params['learning_rate']:.2e}  "
              f"wd={params['weight_decay']:.2e}  "
              f"bs={params['batch_size']}")

        fold_f1s = []
        for fold_idx, (train_idx, val_idx) in enumerate(
                skf.split(dev_graphs, dev_labels_encoded)):
            train_g = [dev_graphs[i] for i in train_idx]
            val_g   = [dev_graphs[i] for i in val_idx]
            f1, _   = train_fold(train_g, val_g, params, num_classes, config,
                                  seed=config["random_state"] + fold_idx,
                                  max_epochs=config["optuna_epochs"])
            fold_f1s.append(f1)
            print(f"  fold {fold_idx+1}/{config['optuna_n_splits']}  "
                  f"F1={f1:.4f}  mean={np.mean(fold_f1s):.4f}")
            trial.report(np.mean(fold_f1s), fold_idx)
            if trial.should_prune():
                print("  → pruned")
                raise optuna.exceptions.TrialPruned()

        return np.mean(fold_f1s)

    return objective


# ---------------------------------------------------------------------------
# Nested CV: Optuna per outer fold, evaluate on outer test set
# ---------------------------------------------------------------------------

def run_nested_cv(all_graphs: list, all_labels_encoded: np.ndarray,
                  label_encoder, num_classes: int, config: dict):
    """
    True Nested CV:
      - Outer loop: n_splits-fold stratified CV
      - For each outer fold:
          1. Run Optuna on outer train set (inner CV) to find best params
          2. Retrain on entire outer train set with best params
          3. Evaluate on outer test set (never seen during Optuna)
    """
    outer_skf = StratifiedKFold(n_splits=config["n_splits"], shuffle=True,
                                random_state=config["random_state"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    fold_results = []
    fold_best_params = []

    for outer_idx, (train_idx, test_idx) in enumerate(
            outer_skf.split(all_graphs, all_labels_encoded)):
        print(f"\n{'─'*60}")
        print(f"[Outer Fold {outer_idx+1}/{config['n_splits']}]  "
              f"train={len(train_idx)}  test={len(test_idx)}")

        outer_train_g = [all_graphs[i] for i in train_idx]
        outer_test_g  = [all_graphs[i] for i in test_idx]
        outer_train_y = all_labels_encoded[train_idx]

        # ── Inner CV: Optuna on outer train set only ──────────────────────
        print(f"  [Optuna] {config['n_trials']} trials, "
              f"{config['optuna_n_splits']}-fold inner CV, "
              f"max {config['optuna_epochs']} epochs/fold")
        pruner  = optuna.pruners.MedianPruner(n_warmup_steps=2)
        sampler = optuna.samplers.TPESampler(seed=config["random_state"])
        study   = optuna.create_study(
            direction="maximize", pruner=pruner, sampler=sampler,
            study_name=f"gemal_{config['source_cpus'][0]}_outer{outer_idx}",
        )

        def trial_callback(study, trial):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                print(f"    → Trial {trial.number+1} done  "
                      f"F1={trial.value:.4f}  best={study.best_value:.4f}")

        objective = make_objective(outer_train_g, outer_train_y, num_classes, config)
        study.optimize(objective, n_trials=config["n_trials"],
                       timeout=config["optuna_timeout"],
                       show_progress_bar=False,
                       callbacks=[trial_callback])

        best_params = {
            "hidden_channels": config["hidden_channels"],
            "embed_dim":       config["embed_dim"],
            "dropout":         config["dropout"],
            "scheduler_type":  config["scheduler_type"],
            **study.best_params,
        }
        print(f"  [Optuna] Best inner F1-macro={study.best_value:.4f}  "
              f"params={study.best_params}")
        fold_best_params.append(best_params)

        # ── Retrain on full outer train set with best params ──────────────
        print(f"  [Retrain] Training on full outer train set …")
        _, best_state = train_fold(
            outer_train_g, outer_test_g, best_params, num_classes, config,
            seed=config["random_state"] + outer_idx,
        )

        # ── Evaluate on outer test set ────────────────────────────────────
        model = build_model(best_params, config["num_node_features"], num_classes, device)
        model.load_state_dict(best_state)
        test_loader = DataLoader(outer_test_g, batch_size=best_params["batch_size"],
                                 shuffle=False)
        fold_res = test_model(model, test_loader, device, label_encoder)
        fold_results.append(fold_res)
        print(f"  [Outer {outer_idx+1}] Acc={fold_res['accuracy']:.4f}  "
              f"F1-macro={fold_res['f1_macro']:.4f}  AUC={fold_res['auc']:.4f}")

    metrics = ["accuracy", "precision", "recall", "f1_micro", "f1_macro", "auc"]
    summary = {}
    for m in metrics:
        vals = [r[m] for r in fold_results]
        summary[f"avg_{m}"] = float(np.mean(vals))
        summary[f"std_{m}"] = float(np.std(vals))

    return summary, fold_results, fold_best_params


# ---------------------------------------------------------------------------
# Main per-arch runner
# ---------------------------------------------------------------------------

def run_arch(arch: str):
    config = get_gemal_single_config(arch)

    os.makedirs(config["optuna_dir"], exist_ok=True)
    os.makedirs(config["result_dir"], exist_ok=True)
    os.makedirs(config["log_dir"],    exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Arch: {arch} | GEMAL (GCN + attention readout)")
    print(f"  hidden={config['hidden_channels']}  embed_dim={config['embed_dim']}  "
          f"dropout={config['dropout']}")
    print(f"{'='*60}")

    all_graphs, all_labels = load_all_graphs(config)
    label_encoder, num_classes = encode_labels(all_graphs, all_labels)
    all_labels_encoded = np.array([int(g.y) for g in all_graphs])
    config["num_node_features"] = all_graphs[0].x.shape[1]
    print(f"Num classes: {num_classes}, Node feature dim: {config['num_node_features']}")

    # ── Nested CV ────────────────────────────────────────────────────────────
    print(f"\n[Nested CV] {config['n_splits']}-fold outer × "
          f"{config['optuna_n_splits']}-fold inner, "
          f"{config['n_trials']} Optuna trials/fold")
    summary, fold_results, fold_best_params = run_nested_cv(
        all_graphs, all_labels_encoded, label_encoder, num_classes, config
    )

    print(f"\n[{arch}] Nested CV Summary ({config['n_splits']}-fold):")
    for m in ["accuracy", "precision", "recall", "f1_micro", "f1_macro", "auc"]:
        print(f"  {m:12s}: {summary[f'avg_{m}']:.4f} ± {summary[f'std_{m}']:.4f}")

    results_dict = {
        "mode":       "Classification (family)",
        "arch_mode":  "單架構",
        "arch":       arch,
        "embedding":  "gemal_cbow",
        "model":      "GEMAL (GCN + attention readout)",
        "source_cpus": config["source_cpus"],
        "n_splits":    config["n_splits"],
        "fold_best_params": fold_best_params,
        **summary,
        "all_results": [
            {
                "fold":       i,
                "best_params": fold_best_params[i],
                "accuracy":   r["accuracy"],
                "precision":  r["precision"],
                "recall":     r["recall"],
                "f1_micro":   r["f1_micro"],
                "f1_macro":   r["f1_macro"],
                "auc":        r["auc"],
            }
            for i, r in enumerate(fold_results)
        ],
    }
    save_experiment_results(results_dict, save_dir=config["result_dir"])
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Single-arch family classification (GEMAL — GCN + attention readout, Nested CV)")
    parser.add_argument("--arch", type=str, default=None,
                        help=f"Target arch. If omitted, runs all: {ALL_ARCHS}")
    args = parser.parse_args()

    archs = [args.arch] if args.arch else ALL_ARCHS
    summaries = {}
    for arch in archs:
        summary = run_arch(arch)
        if summary:
            summaries[arch] = summary

    if len(summaries) > 1:
        print(f"\n{'='*60}")
        print("All Architectures Summary:")
        print(f"{'='*60}")
        for arch, s in summaries.items():
            print(f"  {arch:8s}  Acc: {s['avg_accuracy']:.4f}±{s['std_accuracy']:.4f}  "
                  f"F1-macro: {s['avg_f1_macro']:.4f}±{s['std_f1_macro']:.4f}  "
                  f"AUC: {s['avg_auc']:.4f}±{s['std_auc']:.4f}")


if __name__ == "__main__":
    main()
