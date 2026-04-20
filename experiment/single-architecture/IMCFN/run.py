"""
Single-Architecture Family Classification — IMCFN
(Image-based Malware Classification using Fine-tuned CNN, Vasan et al. 2020)

Pipeline: 224×224 RGB PNG → ImageNet normalize → fine-tuned VGG16 → softmax

Run from PCBSDA root:
    python experiment/single-architecture/IMCFN/run.py
    python experiment/single-architecture/IMCFN/run.py --arch x86_64
    python experiment/single-architecture/IMCFN/run.py --arch x86_64 --tune-only
    python experiment/single-architecture/IMCFN/run.py --arch x86_64 --eval-only
"""

import sys
import os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import copy
import json
import pickle
import random
import datetime

import numpy as np
from tqdm import tqdm
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
import pandas as pd

from config import get_imcfn_single_config, ALL_ARCHS
from model import IMCFN
from PIL import Image
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# ImageNet normalisation — same as used during VGG16 pretraining
_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


class MalwareImageDataset(Dataset):
    """
    Loads pre-generated 224×224 RGB PNG malware images.
    """

    def __init__(self, file_names, labels, image_dir):
        self.file_names = file_names
        self.labels = labels
        self.image_dir = image_dir

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        path = os.path.join(self.image_dir, self.file_names[idx] + ".png")
        img = Image.open(path).convert("RGB")
        return _TRANSFORM(img), torch.tensor(self.labels[idx], dtype=torch.long)


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



def load_data(config):
    """Load CSV, filter by arch, intersect with available PNG files."""
    df = pd.read_csv(config["csv_path"])
    df = df[df["CPU"].isin(config["source_cpus"])].reset_index(drop=True)

    image_dir = config["image_dir"]
    available = set(f.replace(".png", "") for f in os.listdir(image_dir))
    df = df[df["file_name"].isin(available)].reset_index(drop=True)

    print(f"Samples after filtering: {len(df)}")

    le = LabelEncoder()
    labels = le.fit_transform(df["family"].tolist())
    return df["file_name"].tolist(), labels, le


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{total_epochs}", leave=False, unit="batch")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")


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
        "accuracy":  accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average=avg, zero_division=0),
        "recall":    recall_score(all_labels, all_preds, average=avg, zero_division=0),
        "f1_micro":  f1_score(all_labels, all_preds, average="micro", zero_division=0),
        "f1_macro":  f1_score(all_labels, all_preds, average=avg, zero_division=0),
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

    train_ds = MalwareImageDataset(train_names, train_labels, config["image_dir"])
    val_ds   = MalwareImageDataset(val_names, val_labels, config["image_dir"])

    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True,
                              num_workers=config["num_workers"], pin_memory=config["pin_memory"])
    val_loader   = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False,
                              num_workers=config["num_workers"], pin_memory=config["pin_memory"])

    model = IMCFN(num_classes=num_classes, dropout=params["dropout"]).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=params["learning_rate"],
        weight_decay=params.get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.get("cosine_T_max", 25)
    )
    counts = torch.zeros(num_classes)
    for l in train_labels:
        counts[int(l)] += 1
    criterion = nn.CrossEntropyLoss(
        weight=(len(train_labels) / (num_classes * counts)).to(device)
    )

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    patience = config["patience"]

    for epoch in range(1, config["epochs"] + 1):
        train_epoch(model, train_loader, optimizer, criterion, device, epoch, config["epochs"])
        val_acc, val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if improved or patience_counter >= patience:
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
    skf = StratifiedKFold(n_splits=config["optuna_n_splits"], shuffle=True,
                          random_state=config["random_state"])
    ss = config["search_space"]

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", ss["learning_rate"][0],
                                                  ss["learning_rate"][1], log=True),
            "dropout":       trial.suggest_float("dropout", ss["dropout"][0], ss["dropout"][1]),
            "weight_decay":  trial.suggest_float("weight_decay", ss["weight_decay"][0],
                                                  ss["weight_decay"][1], log=True),
            "batch_size":    trial.suggest_categorical("batch_size", ss["batch_size"]),
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
# Nested CV: Optuna per outer fold, evaluate on outer test set
# ---------------------------------------------------------------------------

def run_nested_cv(file_names, labels, label_encoder, num_classes, config):
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
    names_arr = np.array(file_names)

    fold_results = []
    fold_best_params = []

    for outer_idx, (train_idx, test_idx) in enumerate(
            outer_skf.split(file_names, labels)):
        print(f"\n{'─'*60}")
        print(f"[Outer Fold {outer_idx+1}/{config['n_splits']}]  "
              f"train={len(train_idx)}  test={len(test_idx)}")

        outer_train_names = names_arr[train_idx].tolist()
        outer_test_names  = names_arr[test_idx].tolist()
        outer_train_y     = labels[train_idx]
        outer_test_y      = labels[test_idx]

        # ── Inner CV: Optuna on outer train set only ──────────────────────
        print(f"  [Optuna] {config['n_trials']} trials, "
              f"{config['optuna_n_splits']}-fold inner CV")
        pruner  = optuna.pruners.MedianPruner(n_warmup_steps=2)
        sampler = optuna.samplers.TPESampler(seed=config["random_state"])
        study   = optuna.create_study(
            direction="maximize", pruner=pruner, sampler=sampler,
            study_name=f"imcfn_{config['source_cpus'][0]}_outer{outer_idx}",
        )
        objective = make_objective(
            outer_train_names, outer_train_y, num_classes, config
        )
        study.optimize(objective, n_trials=config["n_trials"],
                       timeout=config["optuna_timeout"], show_progress_bar=True)

        best_params = {**study.best_params}
        print(f"  [Optuna] Best inner F1-macro={study.best_value:.4f}  "
              f"params={best_params}")
        fold_best_params.append(best_params)

        # ── Retrain on full outer train set with best params ──────────────
        print(f"  [Retrain] Training on full outer train set …")
        _, best_state = train_fold(
            outer_train_names, outer_train_y,
            outer_test_names,  outer_test_y,
            best_params, num_classes, config,
            seed=config["random_state"] + outer_idx,
        )

        # ── Evaluate on outer test set ────────────────────────────────────
        model = IMCFN(num_classes=num_classes, dropout=best_params["dropout"]).to(device)
        model.load_state_dict(best_state)
        test_ds = MalwareImageDataset(outer_test_names, outer_test_y, config["image_dir"])
        test_loader = DataLoader(test_ds, batch_size=best_params["batch_size"], shuffle=False)
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
# Save results
# ---------------------------------------------------------------------------

def save_results(results_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"results_{ts}.json")
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"[Saved] {path}")


# ---------------------------------------------------------------------------
# Main per-arch runner
# ---------------------------------------------------------------------------

def run_arch(arch, n_trials=None):
    config = get_imcfn_single_config(arch)
    if n_trials is not None:
        config["n_trials"] = n_trials

    os.makedirs(config["optuna_dir"], exist_ok=True)
    os.makedirs(config["result_dir"], exist_ok=True)
    os.makedirs(config["log_dir"],    exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Arch: {arch} | Model: IMCFN (binary → color image → VGG16)")
    print(f"{'='*60}")

    file_names, labels, label_encoder = load_data(config)
    num_classes = len(label_encoder.classes_)
    print(f"Num classes: {num_classes}  Total: {len(file_names)}")

    # --- Nested CV (Optuna inside each outer fold) ---
    print(f"\n[Nested CV] {config['n_splits']}-fold outer × "
          f"{config['optuna_n_splits']}-fold inner, "
          f"{config['n_trials']} Optuna trials/fold")
    summary, fold_results, fold_best_params = run_nested_cv(
        file_names, labels, label_encoder, num_classes, config
    )

    print(f"\n[{arch}] Nested CV Summary ({config['n_splits']}-fold):")
    for m in ["accuracy", "precision", "recall", "f1_micro", "f1_macro", "auc"]:
        print(f"  {m:12s}: {summary[f'avg_{m}']:.4f} ± {summary[f'std_{m}']:.4f}")

    results_dict = {
        "mode": "Classification (family)",
        "arch_mode": "單架構",
        "arch": arch,
        "model": "IMCFN",
        "embedding": "binary_color_image",
        "source_cpus": config["source_cpus"],
        "n_splits": config["n_splits"],
        "fold_best_params": fold_best_params,
        **summary,
        "all_results": [
            {
                "fold": i,
                "best_params": fold_best_params[i],
                "accuracy":  r["accuracy"],
                "precision": r["precision"],
                "recall":    r["recall"],
                "f1_micro":  r["f1_micro"],
                "f1_macro":  r["f1_macro"],
                "auc":       r["auc"],
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
        description="Single-arch family classification (IMCFN: binary → color image → VGG16, Nested CV)"
    )
    parser.add_argument("--arch", type=str, default=None,
                        help=f"Target arch. If omitted, runs all: {ALL_ARCHS}")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Override number of Optuna trials")
    args = parser.parse_args()

    archs = [args.arch] if args.arch else ALL_ARCHS
    summaries = {}
    for arch in archs:
        summary = run_arch(arch, n_trials=args.n_trials)
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
