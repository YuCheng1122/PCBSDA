import json
import os
import pickle
import random

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
from torch.utils.data import Dataset
from torch_geometric.data import Batch

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


def prepare_ccsa_data(config, random_state):
    """
    準備 CCSA 訓練所需的資料:
    - Source domain: train/val split (fully labeled)
    - Target domain: few-shot labeled (num_target_samples) + test set (rest)

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
    # 支援兩種模式: per-class sampling (新) 或 total sampling (舊)
    target_labels_int = [int(g.y) for g in target_graphs]

    if "num_target_samples_per_class" in config:
        # 新模式: 每個 class 抽 n 個，確保所有 class 都被覆蓋
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
        # 舊模式: 總共抽 num_target_samples 個
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

    print(f"\nCCSA Data Summary:")
    print(f"  Source train: {len(src_train)}, Source val: {len(src_val)}")
    print(f"  Target train (few-shot): {len(tgt_train)}, Target test: {len(tgt_test)}")
    print(f"  Num classes: {num_classes}")
    print(f"  Source train distribution: {dict(Counter(int(g.y) for g in src_train))}")
    print(f"  Target train distribution: {dict(Counter(int(g.y) for g in tgt_train))}")
    print(f"  Target test distribution: {dict(Counter(int(g.y) for g in tgt_test))}")

    return src_train, src_val, tgt_train, tgt_test, label_encoder, num_classes

class CCSAPairDataset(Dataset):
    """
    生成 source-target pair 用於 CCSA 訓練。
    每個 pair 包含一個 source graph 和一個 target graph。
    同類 pair (positive) 和不同類 pair (negative) 按比例混合。
    """

    def __init__(self, source_graphs, target_graphs, neg_pair_ratio=3):
        self.source_graphs = source_graphs
        self.target_graphs = target_graphs

        positive_pairs = []
        negative_pairs = []

        for si in range(len(source_graphs)):
            src_label = int(source_graphs[si].y)
            for ti in range(len(target_graphs)):
                tgt_label = int(target_graphs[ti].y)
                if src_label == tgt_label:
                    positive_pairs.append((si, ti, 1))
                else:
                    negative_pairs.append((si, ti, 0))

        print(f"  Positive pairs: {len(positive_pairs)}, Negative pairs: {len(negative_pairs)}")

        random.shuffle(negative_pairs)
        max_neg = neg_pair_ratio * len(positive_pairs)
        self.pairs = positive_pairs + negative_pairs[:max_neg]
        random.shuffle(self.pairs)

        print(f"  Total training pairs: {len(self.pairs)} "
              f"(pos={len(positive_pairs)}, neg={min(len(negative_pairs), max_neg)})")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_idx, tgt_idx, pair_label = self.pairs[idx]
        return src_idx, tgt_idx, pair_label


def ccsa_pair_collate_fn(batch, source_graphs, target_graphs):
    """
    Custom collate: 把 pair indices 轉換成兩個 PyG batch (source batch, target batch)
    以及 pair label (class_eq)。
    """
    from torch_geometric.data import Batch

    src_data_list = []
    tgt_data_list = []
    pair_labels = []

    for src_idx, tgt_idx, pair_label in batch:
        src_data_list.append(source_graphs[src_idx])
        tgt_data_list.append(target_graphs[tgt_idx])
        pair_labels.append(pair_label)

    src_batch = Batch.from_data_list(src_data_list)
    tgt_batch = Batch.from_data_list(tgt_data_list)
    class_eq = torch.tensor(pair_labels, dtype=torch.float)

    return src_batch, tgt_batch, class_eq


def csa_loss(src_feature, tgt_feature, class_eq, margin=1.0):
    """
    Contrastive Semantic Alignment Loss (原版 L2 距離).
    - class_eq=1 (同類): 最小化 feature 距離
    - class_eq=0 (不同類): 確保距離 >= margin
    問題: 模型可能透過操縱 feature norm 來 shortcut，而非學到真正的語義對齊
    """
    dist = F.pairwise_distance(src_feature, tgt_feature)
    loss = class_eq * dist.pow(2)
    loss += (1 - class_eq) * (margin - dist).clamp(min=0).pow(2)
    return loss.mean()


# def csa_loss(src_feature, tgt_feature, class_eq, margin=0.3):
#     """
#     Contrastive Semantic Alignment Loss (Cosine 版本).
#     - 使用 cosine similarity 取代 L2 distance，只看方向不看大小
#     - 避免模型透過放大/縮小 norm 來操縱距離
#     - class_eq=1 (同類): cosine similarity → 1
#     - class_eq=0 (不同類): cosine similarity < margin
#     - margin 建議 0.2~0.5 (cosine range [-1, 1])
#     """
#     # class_eq: 1=同類, 0=不同類 → 轉換成 1/-1
#     target = class_eq * 2 - 1
#     return F.cosine_embedding_loss(src_feature, tgt_feature, target, margin=margin)


def train_ccsa_epoch(model, pair_loader, optimizer, ce_criterion, device, alpha, margin):
    """
    CCSA 訓練一個 epoch:
    - Source: CE loss (分類)
    - Source + Target: CSA loss (contrastive semantic alignment)
    """
    model.train()
    total_loss = 0
    total_ce = 0
    total_csa = 0
    num_pairs = 0

    for src_batch, tgt_batch, class_eq in pair_loader:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)

        src_pred, src_feat = model(src_batch.x, src_batch.edge_index, src_batch.batch)
        _, tgt_feat = model(tgt_batch.x, tgt_batch.edge_index, tgt_batch.batch)

        ce = ce_criterion(src_pred, src_batch.y)

        csa = csa_loss(src_feat, tgt_feat, class_eq.to(device), margin=margin)

        loss = (1 - alpha) * ce + alpha * csa

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = src_batch.num_graphs
        total_loss += loss.item() * batch_size
        total_ce += ce.item() * batch_size
        total_csa += csa.item() * batch_size
        num_pairs += batch_size

    return total_loss / num_pairs, total_ce / num_pairs, total_csa / num_pairs


def evaluate(model, data_loader, device):
    """在 source val 或 target test 上評估準確率"""
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
    """完整的測試評估, 回傳各項 metrics"""
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
    plt.savefig(os.path.join(save_dir, f'ccsa_curves{suffix}.png'))
    plt.close()


def save_experiment_results(results_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    summary_data = {k: v for k, v in results_dict.items()}
    with open(os.path.join(save_dir, f'ccsa_summary_{timestamp}.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)

    return timestamp


# ============================================================
# GraphInflu Phase 1: Supportive Source Graph Selector
# ============================================================

def compute_per_graph_gradients(model, graphs, device, classifier_params=None):
    """
    對每張 graph 單獨算 CE loss 對 classifier head 的 gradient。

    Args:
        model: GNN model (eval mode ok, 但需要 gradient)
        graphs: list of PyG Data objects (each has .y label)
        device: torch device
        classifier_params: list of parameters to compute gradients for.
                          If None, uses model.classifier.parameters()

    Returns:
        grad_matrix: tensor [n_graphs, grad_dim], 每張 graph 的 gradient vector
    """
    if classifier_params is None:
        classifier_params = list(model.classifier.parameters())

    ce_criterion = torch.nn.CrossEntropyLoss()
    grad_list = []

    model.eval()
    for g in graphs:
        g_dev = g.to(device)
        # batch tensor: 單張 graph 所有 node 都屬於 batch 0
        batch_vec = torch.zeros(g_dev.x.size(0), dtype=torch.long, device=device)

        pred, _ = model(g_dev.x, g_dev.edge_index, batch_vec)
        loss = ce_criterion(pred, g_dev.y.unsqueeze(0))

        model.zero_grad()
        loss.backward()

        # 收集 classifier head 的 gradient，flatten 成一個 vector
        grads = []
        for p in classifier_params:
            if p.grad is not None:
                grads.append(p.grad.detach().flatten())
            else:
                grads.append(torch.zeros(p.numel(), device=device))
        grad_list.append(torch.cat(grads))

    return torch.stack(grad_list)  # [n_graphs, grad_dim]


def compute_score_matrix_update(source_grads, target_grads, lr):
    """
    算一次 gradient matching 的 score matrix update。

    score_update[i, j] = lr * ∇L(t_j)^T · ∇L(s_i)

    Args:
        source_grads: [n_source, grad_dim]
        target_grads: [n_target, grad_dim]
        lr: current learning rate

    Returns:
        score_update: [n_source, n_target]
    """
    # target_grads @ source_grads.T → [n_target, n_source]
    # transpose → [n_source, n_target]
    return lr * (target_grads @ source_grads.T).T


def class_balanced_selection(score_matrix, source_graphs, target_graphs,
                             num_select):
    """
    GraphInflu 的 class-balanced selection strategy (論文 Eq.5)。

    做法：每個 class 分配 Q/C 的配額，各自從該 class 的 source graphs 中
    按 class-wise score 選 top-k。確保選出的 subset class 分布均衡。

    Args:
        score_matrix: [n_source, n_target] contribution score matrix
        source_graphs: list of source PyG Data (each has .y)
        target_graphs: list of target PyG Data (each has .y)
        num_select: Q, 要選幾個 source graphs

    Returns:
        selected_indices: list of selected source graph indices
    """
    n_source = len(source_graphs)
    target_labels = [int(g.y) for g in target_graphs]
    source_labels = [int(g.y) for g in source_graphs]
    classes = sorted(set(target_labels))
    num_classes = len(classes)

    # Step 1: 算 class-wise score matrix Γ_hat[i, c]
    # 對每個 class c, 取 target 中 label=c 的 columns 的平均
    class_scores = torch.zeros(n_source, num_classes)
    for ci, c in enumerate(classes):
        target_mask = [j for j, lab in enumerate(target_labels) if lab == c]
        if len(target_mask) > 0:
            class_scores[:, ci] = score_matrix[:, target_mask].mean(dim=1).cpu()

    # Step 2: 每個 source class 分配 Q/C 的配額
    # 從該 class 的 source graphs 中，按對應的 class-wise score 選 top-k
    per_class_quota = num_select // num_classes
    remainder = num_select - per_class_quota * num_classes

    # 按 source 的 label 分組
    class_to_src_indices = {}
    for i, lab in enumerate(source_labels):
        if lab not in class_to_src_indices:
            class_to_src_indices[lab] = []
        class_to_src_indices[lab].append(i)

    selected_indices = []
    for ci, c in enumerate(classes):
        src_indices_for_class = class_to_src_indices.get(c, [])
        if len(src_indices_for_class) == 0:
            continue

        # 這個 class 的配額
        quota = per_class_quota + (1 if ci < remainder else 0)
        quota = min(quota, len(src_indices_for_class))

        # 取這些 source graphs 在 class c 上的 score
        src_idx_tensor = torch.tensor(src_indices_for_class)
        scores = class_scores[src_idx_tensor, ci]

        _, top_k = scores.topk(quota)
        for k in top_k:
            selected_indices.append(src_indices_for_class[k.item()])

    selected_indices = sorted(set(selected_indices))

    # 印出 selection 統計
    sel_labels = [source_labels[i] for i in selected_indices]
    print(f"\n  GraphInflu Selection Summary:")
    print(f"    Selected {len(selected_indices)}/{n_source} source graphs "
          f"({len(selected_indices)/n_source*100:.1f}%)")
    print(f"    Selected distribution: {dict(Counter(sel_labels))}")
    print(f"    Original distribution: {dict(Counter(source_labels))}")

    return selected_indices


def pretrain_and_select(model, source_train, target_train, source_val,
                        config, device):
    """
    GraphInflu Phase 1 完整流程:
    Step 1: Source-only CE pre-training
    Step 2: 每隔 T epoch 收集 gradient，累積 score matrix
    Step 3: Class-balanced selection

    Args:
        model: 已初始化的 GNN model
        source_train: list of source train graphs
        target_train: list of target few-shot graphs
        source_val: list of source val graphs
        config: config dict (需要 graphinflu 相關欄位)
        device: torch device

    Returns:
        selected_source: list of selected source graphs
        model: pre-trained model (可以選擇用或不用)
    """
    from torch_geometric.loader import DataLoader as PyGDataLoader

    pretrain_epochs = config.get("selection_pretrain_epochs", 50)
    gradient_interval = config.get("gradient_interval", 5)
    selection_ratio = config.get("source_selection_ratio", 0.3)
    lr = config.get("learning_rate", 0.001)

    num_select = max(1, int(len(source_train) * selection_ratio))

    # DataLoader for source-only CE training
    train_loader = PyGDataLoader(
        source_train, batch_size=config["batch_size"], shuffle=True,
        num_workers=0, pin_memory=False
    )
    val_loader = PyGDataLoader(
        source_val, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=config["pin_memory"]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce_criterion = torch.nn.CrossEntropyLoss()
    classifier_params = list(model.classifier.parameters())

    # Score matrix: [n_source, n_target]
    score_matrix = torch.zeros(len(source_train), len(target_train), device=device)

    print(f"\n{'='*60}")
    print(f"GraphInflu Phase 1: Source-only Pre-training")
    print(f"  Epochs: {pretrain_epochs}, Gradient interval: {gradient_interval}")
    print(f"  Selection ratio: {selection_ratio} ({num_select}/{len(source_train)} graphs)")
    print(f"{'='*60}")

    for epoch in range(1, pretrain_epochs + 1):
        # --- Train one epoch (source-only CE) ---
        model.train()
        total_loss = 0
        total_samples = 0

        for batch in train_loader:
            batch = batch.to(device)
            pred, _ = model(batch.x, batch.edge_index, batch.batch)
            loss = ce_criterion(pred, batch.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs

        avg_loss = total_loss / total_samples
        val_acc, val_loss = evaluate(model, val_loader, device)

        # --- 每隔 T epoch 收集 gradient 並更新 score matrix ---
        if epoch % gradient_interval == 0:
            current_lr = lr  # 如果用 scheduler 可以改這裡

            source_grads = compute_per_graph_gradients(
                model, source_train, device, classifier_params
            )
            target_grads = compute_per_graph_gradients(
                model, target_train, device, classifier_params
            )

            score_update = compute_score_matrix_update(source_grads, target_grads, current_lr)
            score_matrix += score_update

            print(f"  [Epoch {epoch}] Loss={avg_loss:.4f} | Val Acc={val_acc:.4f} | "
                  f"Val Loss={val_loss:.4f} | * Gradient collected")
        else:
            if epoch <= 5 or epoch % 10 == 0:
                print(f"  [Epoch {epoch}] Loss={avg_loss:.4f} | Val Acc={val_acc:.4f} | "
                      f"Val Loss={val_loss:.4f}")

    # --- Step 3: Class-balanced selection ---
    selected_indices = class_balanced_selection(
        score_matrix, source_train, target_train,
        num_select=num_select
    )

    selected_source = [source_train[i] for i in selected_indices]

    return selected_source, model
