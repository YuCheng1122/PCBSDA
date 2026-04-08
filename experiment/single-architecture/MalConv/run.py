"""
Single-Architecture Family Classification — MalConv (raw bytes)
- Optuna hyperparameter search (inner CV)
- K-Fold cross-validation for final evaluation

Run from PCBSDA root:
    python experiment/single-architecture/MalConv/run.py
    python experiment/single-architecture/MalConv/run.py --arch x86_64
    python experiment/single-architecture/MalConv/run.py --arch x86_64 --tune-only
    python experiment/single-architecture/MalConv/run.py --arch x86_64 --eval-only
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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from config import get_malconv_single_config, ALL_ARCHS
from model import MalConv

import pandas as pd


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RawByteDataset(Dataset):
    def __init__(self, file_names, labels, raw_byte_dir, max_len):
        self.file_names = file_names
        self.labels = labels
        self.raw_byte_dir = raw_byte_dir
        self.max_len = max_len

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        path = os.path.join(self.raw_byte_dir, self.file_names[idx] + ".npy")
        arr = np.load(path).astype(np.int64)
        # truncate or pad to max_len
        if len(arr) >= self.max_len:
            arr = arr[:self.max_len]
        else:
            arr = np.pad(arr, (0, self.max_len - len(arr)), constant_values=0)
        return torch.tensor(arr, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


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


def build_model(params, num_classes, config, device):
    # Architecture is fixed per paper; only dropout comes from Optuna
    model = MalConv(
        num_classes=num_classes,
        embed_dim=config["embed_dim"],
        num_filters=config["num_filters"],
        filter_size=config["filter_size"],
        stride=config["stride"],
        dropout=params["dropout"],
    )
    return model.to(device)


def build_scheduler(optimizer, params):
    stype = params["scheduler_type"]
    if stype == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params.get("step_size", 20),
            gamma=params.get("gamma", 0.5),
        )
    elif stype == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=params.get("plateau_patience", 5),
            factor=params.get("plateau_factor", 0.5),
        )
    else:  # cosine
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=params.get("cosine_T_max", 50)
        )


def make_class_weights(labels, num_classes, device):
    counts = torch.zeros(num_classes)
    for l in labels:
        counts[int(l)] += 1
    weights = len(labels) / (num_classes * counts)
    return weights.to(device)


def load_data(config):
    """Load CSV, filter by arch, intersect with available npy files."""
    df = pd.read_csv(config["csv_path"])
    df = df[df["CPU"].isin(config["source_cpus"])].reset_index(drop=True)

    npy_dir = config["raw_byte_dir"]
    available = set(f.replace(".npy", "") for f in os.listdir(npy_dir))
    df = df[df["file_name"].isin(available)].reset_index(drop=True)

    print(f"Samples after filtering: {len(df)}")

    le = LabelEncoder()
    labels = le.fit_transform(df["family"].tolist())
    return df["file_name"].tolist(), labels, le


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    return acc, avg_loss


def test_model(model, loader, device, label_encoder):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    num_classes = len(label_encoder.classes_)
    avg = "macro"
    results = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average=avg, zero_division=0),
        "recall": recall_score(all_labels, all_preds, average=avg, zero_division=0),
        "f1_micro": f1_score(all_labels, all_preds, average="micro", zero_division=0),
        "f1_macro": f1_score(all_labels, all_preds, average=avg, zero_division=0),
    }
    try:
        if num_classes == 2:
            probs_arr = np.array(all_probs)[:, 1]
            results["auc"] = roc_auc_score(all_labels, probs_arr)
        else:
            results["auc"] = roc_auc_score(
                all_labels, np.array(all_probs),
                multi_class="ovr", average=avg,
            )
    except Exception:
        results["auc"] = float("nan")
    return results


# ---------------------------------------------------------------------------
# Single fold
# ---------------------------------------------------------------------------

def train_fold(train_names, train_labels, val_names, val_labels,
               params, num_classes, config, seed=42):
    set_seed(seed)
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    train_ds = RawByteDataset(train_names, train_labels, config["raw_byte_dir"], config["max_len"])
    val_ds = RawByteDataset(val_names, val_labels, config["raw_byte_dir"], config["max_len"])

    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True,
                              num_workers=config["num_workers"], pin_memory=config["pin_memory"])
    val_loader = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False,
                            num_workers=config["num_workers"], pin_memory=config["pin_memory"])

    model = build_model(params, num_classes, config, device)
    # Paper uses SGD + Nesterov momentum=0.9
    optimizer = torch.optim.SGD(model.parameters(), lr=params["learning_rate"],
                                momentum=0.9, nesterov=True)
    scheduler = build_scheduler(optimizer, {**config, **params})
    class_weights = make_class_weights(train_labels, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

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

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"    Epoch {epoch:3d}/{config['epochs']} | val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  patience={patience_counter}/{patience}{'  *' if improved else ''}")

        if patience_counter >= patience:
            print(f"    Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_model_state)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.arange(num_classes)
    val_results = test_model(model, val_loader, device, label_encoder)
    return val_results["f1_macro"], best_model_state


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(file_names, labels, num_classes, config):
    skf = StratifiedKFold(n_splits=config["n_splits"], shuffle=True,
                          random_state=config["random_state"])
    ss = config["search_space"]

    def objective(trial):
        # Architecture fixed per paper; only tune training hyperparameters
        params = {
            "dropout": trial.suggest_float("dropout", ss["dropout"][0], ss["dropout"][1]),
            "batch_size": trial.suggest_categorical("batch_size", ss["batch_size"]),
            "learning_rate": trial.suggest_float("learning_rate", ss["learning_rate"][0],
                                                  ss["learning_rate"][1], log=True),
            "scheduler_type": trial.suggest_categorical("scheduler_type", ss["scheduler_type"]),
        }

        fold_f1s = []
        names_arr = np.array(file_names)
        try:
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(file_names, labels)):
                f1, _ = train_fold(
                    names_arr[train_idx].tolist(), labels[train_idx],
                    names_arr[val_idx].tolist(), labels[val_idx],
                    params, num_classes, config,
                    seed=config["random_state"] + fold_idx,
                )
                fold_f1s.append(f1)
                trial.report(np.mean(fold_f1s), fold_idx)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()

        return np.mean(fold_f1s)

    return objective


# ---------------------------------------------------------------------------
# Final CV evaluation
# ---------------------------------------------------------------------------

def run_final_cv(file_names, labels, label_encoder, num_classes, best_params, config):
    skf = StratifiedKFold(n_splits=config["n_splits"], shuffle=True,
                          random_state=config["random_state"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    names_arr = np.array(file_names)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(file_names, labels)):
        print(f"\n  [Fold {fold_idx+1}/{config['n_splits']}]")
        _, best_state = train_fold(
            names_arr[train_idx].tolist(), labels[train_idx],
            names_arr[val_idx].tolist(), labels[val_idx],
            best_params, num_classes, config,
            seed=config["random_state"] + fold_idx,
        )

        model = build_model(best_params, num_classes, config, device)
        model.load_state_dict(best_state)

        val_ds = RawByteDataset(
            names_arr[val_idx].tolist(), labels[val_idx],
            config["raw_byte_dir"], config["max_len"],
        )
        val_loader = DataLoader(val_ds, batch_size=best_params["batch_size"], shuffle=False)
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

def save_results(results_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"results_{ts}.json")
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"[Saved] {path}")


def run_arch(arch, tune_only=False, eval_only=False, n_trials=None):
    config = get_malconv_single_config(arch)
    if n_trials is not None:
        config["n_trials"] = n_trials

    os.makedirs(config["result_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Arch: {arch} | Model: MalConv (raw bytes)")
    print(f"{'='*60}")

    file_names, labels, label_encoder = load_data(config)
    num_classes = len(label_encoder.classes_)
    print(f"Num classes: {num_classes}")

    # --- Paper default params (no Optuna) ---
    # paper: SGD lr=0.01, momentum=0.9; we use Adam with similar lr
    best_params = {
        "dropout": 0.5,
        "batch_size": 32,
        "learning_rate": 0.01,
        "scheduler_type": "step",
    }
    print(f"[Paper params] {best_params}")

    # --- Optuna phase (commented out for now) ---
    # os.makedirs(config["optuna_dir"], exist_ok=True)
    # best_params_path = os.path.join(config["optuna_dir"], "best_params.json")
    # study_path = os.path.join(config["optuna_dir"], "study.pkl")
    # if not eval_only:
    #     print(f"\n[Optuna] Starting search: {config['n_trials']} trials, {config['n_splits']}-fold inner CV")
    #     pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
    #     study = optuna.create_study(direction="maximize", pruner=pruner,
    #                                 study_name=f"malconv_{arch}")
    #     objective = make_objective(file_names, labels, num_classes, config)
    #     study.optimize(objective, n_trials=config["n_trials"],
    #                    timeout=config["optuna_timeout"], show_progress_bar=True)
    #     best_params = study.best_params
    #     print(f"\n[Optuna] Best F1-macro: {study.best_value:.4f}")
    #     print(f"[Optuna] Best params: {best_params}")
    #     with open(study_path, "wb") as f:
    #         pickle.dump(study, f)
    #     with open(best_params_path, "w") as f:
    #         json.dump(best_params, f, indent=2)
    #     if tune_only:
    #         return None
    # else:
    #     with open(best_params_path) as f:
    #         best_params = json.load(f)
    #     print(f"[Loaded] Best params: {best_params}")

    # --- Final CV evaluation ---
    print(f"\n[Final CV] Evaluating with best params ({config['n_splits']}-fold)...")
    summary, fold_results = run_final_cv(
        file_names, labels, label_encoder, num_classes, best_params, config
    )

    print(f"\n[{arch}] Final CV Summary ({config['n_splits']}-fold):")
    for m in ["accuracy", "precision", "recall", "f1_micro", "f1_macro", "auc"]:
        print(f"  {m:12s}: {summary[f'avg_{m}']:.4f} ± {summary[f'std_{m}']:.4f}")

    results_dict = {
        "mode": "Classification (family)",
        "arch_mode": "單架構",
        "arch": arch,
        "embedding": "raw_byte",
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
    save_results(results_dict, save_dir=config["result_dir"])
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Single-arch family classification (MalConv raw bytes + CV + Optuna)"
    )
    parser.add_argument("--arch", type=str, default=None,
                        help=f"Target arch. If omitted, runs all: {ALL_ARCHS}")
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
            arch,
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
