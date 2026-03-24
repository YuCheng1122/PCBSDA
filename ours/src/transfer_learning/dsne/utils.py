import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score
)
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Batch

from src.gnn.utils import load_graphs_from_df


def load_domain_data(csv_path, graph_dir, cpus, cache_file, force_reload, classification):
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


def prepare_dsne_data(config, random_state):
    """
    準備 d-SNE 訓練所需的資料:
    - Source domain: train/val split (fully labeled)
    - Target domain: few-shot labeled + test set (rest)
    """
    source_data = load_domain_data(
        config["csv_path"], config["graph_dir"], config["source_cpus"],
        config["source_cache_file"], config["force_reload"], config["classification"]
    )
    source_graphs = source_data['graphs']
    source_labels = source_data['labels']

    target_data = load_domain_data(
        config["csv_path"], config["graph_dir"], config["target_cpus"],
        config["target_cache_file"], config["force_reload"], config["classification"]
    )
    target_graphs = target_data['graphs']
    target_labels = target_data['labels']

    label_encoder = LabelEncoder()
    label_encoder.fit(source_labels + target_labels)
    num_classes = len(label_encoder.classes_)

    source_encoded = label_encoder.transform(source_labels)
    for i, g in enumerate(source_graphs):
        g.y = torch.tensor(source_encoded[i], dtype=torch.long)

    target_encoded = label_encoder.transform(target_labels)
    for i, g in enumerate(target_graphs):
        g.y = torch.tensor(target_encoded[i], dtype=torch.long)

    src_train, src_val = train_test_split(
        source_graphs, test_size=config["source_val_size"],
        stratify=[int(g.y) for g in source_graphs], random_state=random_state
    )

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

    print(f"\nd-SNE Data Summary:")
    print(f"  Source train: {len(src_train)}, Source val: {len(src_val)}")
    print(f"  Target train (few-shot): {len(tgt_train)}, Target test: {len(tgt_test)}")
    print(f"  Num classes: {num_classes}")
    print(f"  Source train distribution: {dict(Counter(int(g.y) for g in src_train))}")
    print(f"  Target train distribution: {dict(Counter(int(g.y) for g in tgt_train))}")
    print(f"  Target test distribution: {dict(Counter(int(g.y) for g in tgt_test))}")

    return src_train, src_val, tgt_train, tgt_test, label_encoder, num_classes


def dsne_loss(src_features, tgt_features, src_labels, tgt_labels, margin=1.0, normalize=False):
    """
    d-SNE loss: modified-Hausdorff distance for domain adaptation (paper Eq. 6).

    對每個 target sample x_j (label=k):
      max_intra = max distance to same-class source
      min_inter = min distance to different-class source
      loss_j = clamp(margin - (min_inter - max_intra), min=0)
    """
    if normalize:
        src_features = F.normalize(src_features, dim=1)
        tgt_features = F.normalize(tgt_features, dim=1)

    # Pairwise L2-squared distance: (N_s, N_t)
    src_sq = src_features.pow(2).sum(dim=1, keepdim=True)      # (N_s, 1)
    tgt_sq = tgt_features.pow(2).sum(dim=1, keepdim=True)      # (N_t, 1)
    cross = src_features @ tgt_features.t()                     # (N_s, N_t)
    dists = src_sq + tgt_sq.t() - 2 * cross                    # (N_s, N_t)
    dists = dists.clamp(min=0)

    # Label match mask: (N_s, N_t)
    same_mask = src_labels.unsqueeze(1) == tgt_labels.unsqueeze(0)
    diff_mask = ~same_mask

    losses = []
    n_tgt = tgt_features.size(0)

    for j in range(n_tgt):
        col = dists[:, j]
        same_j = same_mask[:, j]
        diff_j = diff_mask[:, j]

        if not same_j.any() or not diff_j.any():
            continue

        max_intra = col[same_j].max()
        min_inter = col[diff_j].min()

        loss_j = (margin - (min_inter - max_intra)).clamp(min=0)
        losses.append(loss_j)

    if len(losses) == 0:
        return torch.tensor(0.0, device=src_features.device, requires_grad=True)

    return torch.stack(losses).mean()


def train_dsne_epoch(model, source_loader, target_batch, optimizer, ce_criterion,
                     device, alpha, beta, dsne_margin, bidirectional=True):
    """
    d-SNE 訓練一個 epoch (paper Eq. 7):
    loss = d-SNE + alpha * CE_source + beta * CE_target

    bidirectional=True: 每個 step 做兩次 backward（官方設計）
      - 第一次：src 為 training side，tgt 為 comparison
      - 第二次：tgt 為 training side，src 為 comparison
    """
    model.train()
    total_loss = 0
    total_ce_src = 0
    total_ce_tgt = 0
    total_dsne = 0
    num_samples = 0

    for src_batch in source_loader:
        src_batch = src_batch.to(device)

        # --- Forward both domains ---
        src_pred, src_feat = model(src_batch.x, src_batch.edge_index, src_batch.batch)
        tgt_pred, tgt_feat = model(target_batch.x, target_batch.edge_index, target_batch.batch)

        ce_src = ce_criterion(src_pred, src_batch.y)
        ce_tgt = ce_criterion(tgt_pred, target_batch.y)

        # --- Direction 1: src as training side (tgt as comparison) ---
        dsne_fwd = dsne_loss(src_feat, tgt_feat, src_batch.y, target_batch.y,
                             margin=dsne_margin)
        loss_fwd = dsne_fwd + alpha * ce_src + beta * ce_tgt

        optimizer.zero_grad()
        loss_fwd.backward()
        optimizer.step()

        if bidirectional:
            # --- Re-forward after weight update ---
            src_pred2, src_feat2 = model(src_batch.x, src_batch.edge_index, src_batch.batch)
            tgt_pred2, tgt_feat2 = model(target_batch.x, target_batch.edge_index, target_batch.batch)

            ce_src2 = ce_criterion(src_pred2, src_batch.y)
            ce_tgt2 = ce_criterion(tgt_pred2, target_batch.y)

            # --- Direction 2: tgt as training side (src as comparison) ---
            dsne_bwd = dsne_loss(tgt_feat2, src_feat2, target_batch.y, src_batch.y,
                                 margin=dsne_margin)
            loss_bwd = dsne_bwd + alpha * ce_tgt2 + beta * ce_src2

            optimizer.zero_grad()
            loss_bwd.backward()
            optimizer.step()

            # Track average of both directions
            batch_dsne = (dsne_fwd.item() + dsne_bwd.item()) / 2
            batch_loss = (loss_fwd.item() + loss_bwd.item()) / 2
            batch_ce_src = (ce_src.item() + ce_src2.item()) / 2
            batch_ce_tgt = (ce_tgt.item() + ce_tgt2.item()) / 2
        else:
            batch_dsne = dsne_fwd.item()
            batch_loss = loss_fwd.item()
            batch_ce_src = ce_src.item()
            batch_ce_tgt = ce_tgt.item()

        batch_size = src_batch.num_graphs
        total_loss += batch_loss * batch_size
        total_ce_src += batch_ce_src * batch_size
        total_ce_tgt += batch_ce_tgt * batch_size
        total_dsne += batch_dsne * batch_size
        num_samples += batch_size

    return (total_loss / num_samples, total_ce_src / num_samples,
            total_ce_tgt / num_samples, total_dsne / num_samples)


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            out, _ = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            pred = out.argmax(dim=1)
            correct += int((pred == batch.y).sum())
            total_loss += loss.item() * batch.num_graphs

    accuracy = correct / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader.dataset)
    return accuracy, avg_loss


def test_model(model, test_loader, device, label_encoder):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out, _ = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            prob = torch.softmax(out, dim=1)
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
        'y_pred': y_pred
    }


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
    plt.savefig(os.path.join(save_dir, f'dsne_curves{suffix}.png'))
    plt.close()


def save_experiment_results(results_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    summary_data = {k: v for k, v in results_dict.items()}
    with open(os.path.join(save_dir, f'dsne_summary_{timestamp}.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)

    return timestamp
