import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score
)
from sklearn.preprocessing import LabelEncoder

from src.gnn.utils import load_graphs_from_df


def load_domain_data(csv_path, graph_dir, cpus, cache_file, force_reload, classification):
    """載入某個 domain (source 或 target) 的所有圖資料"""
    if force_reload and os.path.exists(cache_file):
        os.remove(cache_file)

    if os.path.exists(cache_file):
        print(f"載入快取: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    df = pd.read_csv(csv_path)
    domain_df = df[df['CPU'].isin(cpus)]
    print(f"載入 {cpus} 資料: {len(domain_df)} 個樣本")

    graphs, labels = load_graphs_from_df(domain_df, graph_dir, classification)

    data = {'graphs': graphs, 'labels': labels}
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"已快取到: {cache_file}")

    return data


def prepare_dann_data(config, random_state):
    """
    準備 DANN 訓練所需的資料:
    - Source domain: train/val split (fully labeled)
    - Target domain: few-shot labeled train + test set (rest)

    DANN 原版 target train 不需要 label，但保留 label 可以選擇性加 CE_target loss。

    Returns:
        source_train_graphs, source_val_graphs,
        target_train_graphs, target_test_graphs,
        label_encoder, num_classes
    """
    # Load source domain
    source_data = load_domain_data(
        config["csv_path"], config["graph_dir"], config["source_cpus"],
        config["source_cache_file"], config["force_reload"], config["classification"]
    )
    source_graphs = source_data['graphs']
    source_labels = source_data['labels']

    # Load target domain
    target_data = load_domain_data(
        config["csv_path"], config["graph_dir"], config["target_cpus"],
        config["target_cache_file"], config["force_reload"], config["classification"]
    )
    target_graphs = target_data['graphs']
    target_labels = target_data['labels']

    # Fit label encoder on all labels
    label_encoder = LabelEncoder()
    label_encoder.fit(source_labels + target_labels)
    num_classes = len(label_encoder.classes_)

    # Encode labels
    source_encoded = label_encoder.transform(source_labels)
    for i, g in enumerate(source_graphs):
        g.y = torch.tensor(source_encoded[i], dtype=torch.long)

    target_encoded = label_encoder.transform(target_labels)
    for i, g in enumerate(target_graphs):
        g.y = torch.tensor(target_encoded[i], dtype=torch.long)

    # Split source: train/val
    src_train, src_val = train_test_split(
        source_graphs, test_size=config["source_val_size"],
        stratify=[int(g.y) for g in source_graphs], random_state=random_state
    )

    # Split target: few-shot train / test
    target_labels_int = [int(g.y) for g in target_graphs]

    if "num_target_samples_per_class" in config:
        n_per_class = config["num_target_samples_per_class"]
        rng = np.random.RandomState(random_state)
        tgt_train_idx = []
        for c in sorted(set(target_labels_int)):
            c_indices = [i for i, y in enumerate(target_labels_int) if y == c]
            n_sample = min(n_per_class, len(c_indices))
            tgt_train_idx.extend(rng.choice(c_indices, n_sample, replace=False).tolist())
        tgt_test_idx = [i for i in range(len(target_graphs)) if i not in set(tgt_train_idx)]
        tgt_train = [target_graphs[i] for i in tgt_train_idx]
        tgt_test = [target_graphs[i] for i in tgt_test_idx]
    else:
        num_target = config["num_target_samples"]
        if num_target >= len(target_graphs):
            raise ValueError(
                f"num_target_samples ({num_target}) >= total target graphs ({len(target_graphs)}). "
                f"Reduce num_target_samples."
            )
        tgt_train, tgt_test = train_test_split(
            target_graphs, train_size=num_target,
            stratify=target_labels_int, random_state=random_state
        )

    # DANN domain alignment 用全量 target（train + test），不看 class label
    tgt_all = tgt_train + tgt_test

    print(f"\nDANN Data Summary:")
    print(f"  Source train: {len(src_train)}, Source val: {len(src_val)}")
    print(f"  Target domain align (unlabeled): {len(tgt_all)}  |  Target test: {len(tgt_test)}")
    print(f"  Num classes: {num_classes}")
    print(f"  Source train distribution: {dict(Counter(int(g.y) for g in src_train))}")

    return src_train, src_val, tgt_train, tgt_test, tgt_all, label_encoder, num_classes


def compute_alpha(epoch, total_epochs):
    """DANN 原版 alpha schedule: 從 0 漸增到 1，用 sigmoid-like 曲線.
    p = epoch / total_epochs, alpha = 2 / (1 + exp(-10*p)) - 1
    """
    p = epoch / total_epochs
    return 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0


def train_dann_epoch(model, src_loader, tgt_domain_loader, optimizer,
                     ce_criterion, domain_criterion, device,
                     alpha, lambda_domain):
    """
    DANN 訓練一個 epoch（論文原版 unsupervised DA）:
    - Source: CE_class loss（分類）+ domain loss（label=0）
    - Target all: domain loss（label=1）—target class label 完全不使用

    Args:
        model: GCN_DANN or GAT_DANN
        src_loader: source DataLoader（有 class label）
        tgt_domain_loader: 全量 target DataLoader（domain alignment 用，不看 class label）
        optimizer: optimizer
        ce_criterion: CrossEntropyLoss for class
        domain_criterion: CrossEntropyLoss for domain
        device: torch device
        alpha: gradient reversal scale（隨 epoch 增加）
        lambda_domain: domain loss weight

    Returns:
        avg_total_loss, avg_ce_src, avg_domain_src, avg_domain_tgt
    """
    model.train()

    total_loss = 0.0
    total_ce_src = 0.0
    total_domain_src = 0.0
    total_domain_tgt = 0.0
    n_samples = 0

    # target domain loader cycle iterator
    tgt_domain_iter = iter(tgt_domain_loader)

    for src_batch in src_loader:
        # 從 target domain loader cycle 取一個 batch（不用 class label）
        try:
            tgt_batch = next(tgt_domain_iter)
        except StopIteration:
            tgt_domain_iter = iter(tgt_domain_loader)
            tgt_batch = next(tgt_domain_iter)

        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)

        # Source forward
        src_class_out, src_domain_out, _ = model(
            src_batch.x, src_batch.edge_index, src_batch.batch, alpha=alpha
        )

        # Target forward（只用 domain loss，class label 完全不看）
        _, tgt_domain_out, _ = model(
            tgt_batch.x, tgt_batch.edge_index, tgt_batch.batch, alpha=alpha
        )

        # Domain labels: source=0, target=1
        src_domain_label = torch.zeros(src_batch.num_graphs, dtype=torch.long, device=device)
        tgt_domain_label = torch.ones(tgt_batch.num_graphs, dtype=torch.long, device=device)

        # Loss = CE_src + lambda * (domain_src + domain_tgt)
        ce_src = ce_criterion(src_class_out, src_batch.y)
        domain_src = domain_criterion(src_domain_out, src_domain_label)
        domain_tgt = domain_criterion(tgt_domain_out, tgt_domain_label)

        loss = ce_src + lambda_domain * (domain_src + domain_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = src_batch.num_graphs
        total_loss += loss.item() * bs
        total_ce_src += ce_src.item() * bs
        total_domain_src += domain_src.item() * bs
        total_domain_tgt += domain_tgt.item() * bs
        n_samples += bs

    n = max(n_samples, 1)
    return (
        total_loss / n,
        total_ce_src / n,
        total_domain_src / n,
        total_domain_tgt / n,
    )


def evaluate(model, data_loader, device):
    """在 source val 或 target test 上評估準確率"""
    model.eval()
    correct = 0
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            class_out, _, _ = model(batch.x, batch.edge_index, batch.batch, alpha=0.0)
            loss = criterion(class_out, batch.y)
            pred = class_out.argmax(dim=1)
            correct += int((pred == batch.y).sum())
            total_loss += loss.item() * batch.num_graphs

    accuracy = correct / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader.dataset)
    return accuracy, avg_loss


def test_model(model, test_loader, device, label_encoder):
    """完整的測試評估，回傳各項 metrics"""
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            class_out, _, _ = model(batch.x, batch.edge_index, batch.batch, alpha=0.0)
            pred = class_out.argmax(dim=1)
            prob = torch.softmax(class_out, dim=1)
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_prob.extend(prob.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    y_prob_array = np.array(y_prob)
    if len(label_encoder.classes_) == 2:
        auc_score = roc_auc_score(y_true, y_prob_array[:, 1])
    else:
        auc_score = roc_auc_score(y_true, y_prob_array, multi_class='ovr', average='macro')

    report = classification_report(y_true, y_pred,
                                   target_names=[str(c) for c in label_encoder.classes_])

    original_labels = label_encoder.inverse_transform(sorted(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=label_encoder.transform(original_labels))

    print("Report:\n", report)

    return {
        'accuracy': accuracy,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'auc': auc_score,
        'precision': precision,
        'recall': recall,
        'classification_report': report,
        'confusion_matrix': cm,
        'original_labels': original_labels,
        'y_true': y_true,
        'y_pred': y_pred,
    }


# ============================================================
# Plotting & Saving
# ============================================================

def plot_training_curves(train_losses, val_losses, val_accuracies, save_dir, random_state=None):
    os.makedirs(save_dir, exist_ok=True)

    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Source Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()

    axes[1].plot(val_accuracies, label='Source Val Acc', color='blue')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Source Validation Accuracy')
    axes[1].legend()

    plt.tight_layout()
    suffix = f"_rs{random_state}" if random_state is not None else ""
    plt.savefig(os.path.join(save_dir, f'dann_curves{suffix}.png'))
    plt.close()


def save_experiment_results(results_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    summary_data = {k: v for k, v in results_dict.items()}
    with open(os.path.join(save_dir, f'dann_summary_{timestamp}.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)

    return timestamp
