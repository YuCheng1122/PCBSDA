"""
Single-Architecture Family Classification — Word2Vec Embedding
- Optuna hyperparameter search (inner CV)
- K-Fold cross-validation for final evaluation

Run from PCBSDA root:
    python experiment/single-architecture/Word2Vec/run.py
    python experiment/single-architecture/Word2Vec/run.py --arch x86_64 --w2v-model cbow
    python experiment/single-architecture/Word2Vec/run.py --arch x86_64 --tune-only
    python experiment/single-architecture/Word2Vec/run.py --arch x86_64 --eval-only
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import argparse
import copy
import json
import pickle
import random

import numpy as np
import torch
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch_geometric.loader import DataLoader

from ours.src.gnn.models import GCN, GAT
from ours.src.gnn.utils import (
    load_graphs_from_df,
    train_epoch,
    evaluate,
    test_model,
    create_gnn_scheduler,
    save_experiment_results,
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from config import get_w2v_single_config, ALL_ARCHS, W2V_MODELS

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(params, num_node_features, num_classes, device):
    common = dict(
        num_node_features=num_node_features,
        hidden_channels=params["hidden_channels"],
        output_channels=params["output_channels"],
        num_classes=num_classes,
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        pooling=params["pooling"],
    )
    if params["model_type"] == "GAT":
        model = GAT(**common, heads=params.get("gat_heads", 4))
    else:
        model = GCN(**common)
    return model.to(device)


def build_scheduler(optimizer, params):
    stype = params["scheduler_type"]
    return create_gnn_scheduler(
        optimizer, stype,
        step_size=params.get("step_size", 30),
        gamma=params.get("gamma", 0.5),
        patience=params.get("plateau_patience", 10),
        factor=params.get("plateau_factor", 0.5),
        T_max=params.get("cosine_T_max", 100),
    )


def make_class_weights(graphs, num_classes, device):
    counts = torch.zeros(num_classes)
    for g in graphs:
        counts[int(g.y)] += 1
    weights = len(graphs) / (num_classes * counts)
    return weights.to(device)


def load_all_graphs(config):
    """Load all graphs for the target architecture (no split yet)."""
    df = pd.read_csv(config["csv_path"])
    arch_df = df[df["CPU"].isin(config["source_cpus"])]
    print(f"Total samples for {config['source_cpus']}: {len(arch_df)}")
    graphs, labels = load_graphs_from_df(arch_df, config["graph_dir"], classification=True)
    return graphs, labels


def encode_labels(graphs, labels):
    """Fit LabelEncoder and assign data.y in-place. Returns label_encoder, num_classes."""
    le = LabelEncoder()
    encoded = le.fit_transform(labels)
    for i, g in enumerate(graphs):
        g.y = torch.tensor(encoded[i], dtype=torch.long)
    return le, len(le.classes_)


# ---------------------------------------------------------------------------
# Single fold training — returns best val F1-macro
# ---------------------------------------------------------------------------

def train_fold(train_graphs, val_graphs, params, num_classes, config, seed=42):
    set_seed(seed)
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    batch_size = params["batch_size"]
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True,
                              num_workers=config["num_workers"], pin_memory=config["pin_memory"])
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False,
                            num_workers=config["num_workers"], pin_memory=config["pin_memory"])

    model = build_model(params, config["num_node_features"], num_classes, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    scheduler = build_scheduler(optimizer, {**config, **params})
    class_weights = make_class_weights(train_graphs, num_classes, device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    patience = config["patience"]

    for epoch in range(1, config["epochs"] + 1):
        train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, val_loss = evaluate(model, val_loader, device)

        if params["scheduler_type"] == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Evaluate best model on val set for metric
    model.load_state_dict(best_model_state)
    label_encoder = LabelEncoder()  # dummy — test_model only needs classes count
    label_encoder.classes_ = np.arange(num_classes)
    val_results = test_model(model, val_loader, device, label_encoder)
    return val_results["f1_macro"], best_model_state


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(dev_graphs, dev_labels_encoded, num_classes, config):
    skf = StratifiedKFold(n_splits=config["optuna_n_splits"], shuffle=True,
                          random_state=config["random_state"])
    ss = config["search_space"]

    def objective(trial):
        params = {
            "model_type": config["model_type"],
            "pooling": config["pooling"],
            "gat_heads": config["gat_heads"],
            "scheduler_type": config["scheduler_type"],
            "hidden_channels": config["hidden_channels"],
            "output_channels": config["output_channels"],
            "batch_size": config["batch_size"],
            "learning_rate": trial.suggest_float("learning_rate", ss["learning_rate"][0],
                                                  ss["learning_rate"][1], log=True),
            "num_layers": trial.suggest_int("num_layers", ss["num_layers"][0], ss["num_layers"][-1]),
            "dropout": trial.suggest_float("dropout", ss["dropout"][0], ss["dropout"][1]),
        }

        fold_f1s = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(dev_graphs, dev_labels_encoded)):
            train_g = [dev_graphs[i] for i in train_idx]
            val_g = [dev_graphs[i] for i in val_idx]
            f1, _ = train_fold(train_g, val_g, params, num_classes, config,
                                seed=config["random_state"] + fold_idx)
            fold_f1s.append(f1)
            trial.report(np.mean(fold_f1s), fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(fold_f1s)

    return objective


# ---------------------------------------------------------------------------
# Final CV evaluation with best params
# ---------------------------------------------------------------------------

def run_final_cv(all_graphs, all_labels_encoded, label_encoder, num_classes, best_params, config):
    skf = StratifiedKFold(n_splits=config["n_splits"], shuffle=True,
                          random_state=config["random_state"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_graphs, all_labels_encoded)):
        print(f"\n  [Fold {fold_idx+1}/{config['n_splits']}]")
        train_g = [all_graphs[i] for i in train_idx]
        val_g = [all_graphs[i] for i in val_idx]
        _, best_state = train_fold(train_g, val_g, best_params, num_classes, config,
                                   seed=config["random_state"] + fold_idx)

        # Evaluate on val fold
        model = build_model(best_params, config["num_node_features"], num_classes, device)
        model.load_state_dict(best_state)
        val_loader = DataLoader(val_g, batch_size=best_params["batch_size"], shuffle=False)
        fold_res = test_model(model, val_loader, device, label_encoder)
        fold_results.append(fold_res)
        print(f"    Acc={fold_res['accuracy']:.4f}  F1-macro={fold_res['f1_macro']:.4f}  "
              f"AUC={fold_res['auc']:.4f}")

    metrics = ["accuracy", "precision", "recall", "f1_micro", "f1_macro", "auc"]
    summary = {}
    for m in metrics:
        vals = [r[m] for r in fold_results]
        summary[f"avg_{m}"] = float(np.mean(vals))
        summary[f"std_{m}"] = float(np.std(vals))

    return summary, fold_results


# ---------------------------------------------------------------------------
# Main per-arch runner
# ---------------------------------------------------------------------------

def run_arch(arch, w2v_model, tune_only=False, eval_only=False, n_trials=None):
    config = get_w2v_single_config(arch, w2v_model)
    if n_trials is not None:
        config["n_trials"] = n_trials

    os.makedirs(config["optuna_dir"], exist_ok=True)
    os.makedirs(config["result_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Arch: {arch} | Embedding: {w2v_model}")
    print(f"{'='*60}")

    # Load data
    all_graphs, all_labels = load_all_graphs(config)
    label_encoder, num_classes = encode_labels(all_graphs, all_labels)
    all_labels_encoded = np.array([int(g.y) for g in all_graphs])
    config["num_node_features"] = all_graphs[0].x.shape[1]
    print(f"Num classes: {num_classes}, Node feature dim: {config['num_node_features']}")

    best_params_path = os.path.join(config["optuna_dir"], "best_params.json")
    study_path = os.path.join(config["optuna_dir"], "study.pkl")

    # --- Optuna phase (inner CV on all data) ---
    if not eval_only:
        print(f"\n[Optuna] Starting search: {config['n_trials']} trials, {config['optuna_n_splits']}-fold inner CV")
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
        sampler = optuna.samplers.TPESampler(seed=config["random_state"])
        study = optuna.create_study(direction="maximize", pruner=pruner, sampler=sampler,
                                    study_name=f"single_{w2v_model}_{arch}")
        objective = make_objective(all_graphs, all_labels_encoded, num_classes, config)
        study.optimize(objective, n_trials=config["n_trials"],
                       timeout=config["optuna_timeout"], show_progress_bar=True)
        print(f"\n[Optuna] Best F1-macro: {study.best_value:.4f}")
        print(f"[Optuna] Best searched params: {study.best_params}")
        best_params = {
            "model_type": config["model_type"],
            "pooling": config["pooling"],
            "gat_heads": config["gat_heads"],
            "scheduler_type": config["scheduler_type"],
            "hidden_channels": config["hidden_channels"],
            "output_channels": config["output_channels"],
            "batch_size": config["batch_size"],
            **study.best_params,
        }
        with open(study_path, "wb") as f:
            pickle.dump(study, f)
        with open(best_params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"[Optuna] Study saved: {study_path}")
        importances = optuna.importance.get_param_importances(study)
        print("[Optuna] Parameter importances:")
        for param, imp in importances.items():
            print(f"  {param:20s}: {imp:.4f}")
        if tune_only:
            return None
    else:
        if not os.path.exists(best_params_path):
            raise FileNotFoundError(f"Best params not found at {best_params_path}. Run without --eval-only first.")
        with open(best_params_path) as f:
            best_params = json.load(f)
        for key in ("model_type", "pooling", "gat_heads", "scheduler_type"):
            best_params.setdefault(key, config[key])
        print(f"[Loaded] Best params: {best_params}")

    # --- Outer CV evaluation (on all data) ---
    print(f"\n[Final CV] Evaluating with best params ({config['n_splits']}-fold outer CV)...")
    summary, fold_results = run_final_cv(
        all_graphs, all_labels_encoded, label_encoder, num_classes, best_params, config
    )

    print(f"\n[{arch}] Final CV Summary ({config['n_splits']}-fold):")
    for m in ["accuracy", "precision", "recall", "f1_micro", "f1_macro", "auc"]:
        print(f"  {m:12s}: {summary[f'avg_{m}']:.4f} ± {summary[f'std_{m}']:.4f}")

    results_dict = {
        "mode": "Classification (family)",
        "arch_mode": "單架構",
        "arch": arch,
        "embedding": w2v_model,
        "source_cpus": config["source_cpus"],
        "n_splits": config["n_splits"],
        "best_params": best_params,
        **summary,
        "all_results": [
            {
                "fold": i,
                "accuracy": r["accuracy"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1_micro": r["f1_micro"],
                "f1_macro": r["f1_macro"],
                "auc": r["auc"],
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
    parser = argparse.ArgumentParser(description="Single-arch family classification (Word2Vec + CV + Optuna)")
    parser.add_argument("--arch", type=str, default=None,
                        help=f"Target arch. If omitted, runs all: {ALL_ARCHS}")
    parser.add_argument("--w2v-model", type=str, default="cbow", choices=W2V_MODELS,
                        help="Word2Vec variant: cbow, skipgram, fast_text")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Override number of Optuna trials")
    parser.add_argument("--tune-only", action="store_true",
                        help="Only run Optuna search, skip final CV evaluation")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip Optuna, load existing best_params and run final CV")
    args = parser.parse_args()

    archs = [args.arch] if args.arch else ALL_ARCHS
    summaries = {}
    for arch in archs:
        summary = run_arch(
            arch, args.w2v_model,
            tune_only=args.tune_only,
            eval_only=args.eval_only,
            n_trials=args.n_trials,
        )
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
