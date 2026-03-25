import json
import numpy as np
import os
import pickle
import torch
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score
)
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torch_geometric.utils import from_networkx


def load_single_arch_data(csv_path, graph_dir, source_cpus, cache_file,
                          val_size, test_size, random_state, force_reload, classification):
    """
    單架構模式：載入資料並切分為 train/val/test

    Returns:
        (train_graphs, val_graphs, test_graphs, label_encoder, num_classes)
    """
    # cache 路徑帶 random_state，確保不同 seed 的 split 不互相覆蓋
    cache_file = cache_file.replace(".pkl", f"_seed{random_state}.pkl")

    if force_reload and os.path.exists(cache_file):
        os.remove(cache_file)

    if os.path.exists(cache_file):
        print(f"載入快取資料: {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        return (cached_data['train_graphs'], cached_data['val_graphs'],
                cached_data['test_graphs'], cached_data['label_encoder'],
                cached_data['num_classes'])

    print("載入 CSV 資料...")
    df = pd.read_csv(csv_path)
    all_df = df[df['CPU'].isin(source_cpus)]
    print(f"單架構模式: {len(all_df)} 個樣本 (架構: {source_cpus})")

    all_graphs, all_labels = load_graphs_from_df(all_df, graph_dir, classification)

    train_val_graphs, test_graphs, train_val_labels, test_labels = train_test_split(
        all_graphs, all_labels, test_size=test_size,
        stratify=all_labels, random_state=random_state
    )
    train_graphs, val_graphs, train_labels, val_labels = train_test_split(
        train_val_graphs, train_val_labels, test_size=val_size / (1 - test_size),
        stratify=train_val_labels, random_state=random_state
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels + val_labels + test_labels)

    for split_graphs, split_labels in [(train_graphs, train_labels),
                                       (val_graphs, val_labels),
                                       (test_graphs, test_labels)]:
        encoded = label_encoder.transform(split_labels)
        for i, data in enumerate(split_graphs):
            data.y = torch.tensor(encoded[i], dtype=torch.long)

    num_classes = len(label_encoder.classes_)

    # 快取
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'train_graphs': train_graphs, 'val_graphs': val_graphs,
            'test_graphs': test_graphs, 'label_encoder': label_encoder,
            'num_classes': num_classes
        }, f)
    print(f"資料已快取到: {cache_file}")

    return train_graphs, val_graphs, test_graphs, label_encoder, num_classes


def load_cross_arch_data(csv_path, graph_dir, source_cpus, target_cpus, cache_file,
                         val_size, random_state, force_reload, classification):
    """
    跨架構模式：用 source 架構 train/val，target 架構在評估時另外載入

    Returns:
        (train_graphs, val_graphs, label_encoder, num_classes)
    """
    if force_reload and os.path.exists(cache_file):
        os.remove(cache_file)

    if os.path.exists(cache_file):
        print(f"載入快取資料: {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        return (cached_data['train_graphs'], cached_data['val_graphs'],
                cached_data['label_encoder'], cached_data['num_classes'])

    print("載入 CSV 資料...")
    df = pd.read_csv(csv_path)
    train_df = df[df['CPU'].isin(source_cpus)]
    print(f"跨架構模式:")
    print(f"訓練資料: {len(train_df)} 個樣本 (架構: {source_cpus})")
    print(f"測試架構: {target_cpus} (將在訓練結束後分別載入)")

    train_all_graphs, train_all_labels = load_graphs_from_df(train_df, graph_dir, classification)
    train_graphs, val_graphs, train_labels, val_labels = train_test_split(
        train_all_graphs, train_all_labels, test_size=val_size,
        stratify=train_all_labels, random_state=random_state
    )

    label_encoder = LabelEncoder()
    target_df = df[df['CPU'].isin(target_cpus)]
    all_labels_for_fit = train_labels + val_labels
    if classification:
        all_labels_for_fit += target_df['family'].tolist()
    else:
        all_labels_for_fit += target_df['label'].tolist()
    label_encoder.fit(all_labels_for_fit)

    for split_graphs, split_labels in [(train_graphs, train_labels),
                                       (val_graphs, val_labels)]:
        encoded = label_encoder.transform(split_labels)
        for i, data in enumerate(split_graphs):
            data.y = torch.tensor(encoded[i], dtype=torch.long)

    num_classes = len(label_encoder.classes_)

    # 快取
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'train_graphs': train_graphs, 'val_graphs': val_graphs,
            'label_encoder': label_encoder, 'num_classes': num_classes
        }, f)
    print(f"資料已快取到: {cache_file}")

    return train_graphs, val_graphs, label_encoder, num_classes


def load_graphs_from_df(df, graph_dir, classification=False, log_dir=None):
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "logs")
    os.makedirs(log_dir, exist_ok=True)
    method_tag = os.path.basename(graph_dir.rstrip('/'))
    log_path = os.path.join(log_dir, f"skipped_files_{method_tag}.log")

    graphs = []
    labels = []
    skipped = []  # (file_name, cpu, reason)

    for _, row in df.iterrows():
        file_name = row['file_name']
        prefix = file_name[:2]
        cpu = row.get('CPU', 'unknown')
        label = row['family'] if classification else row['label']
        graph_path = Path(graph_dir) / prefix / f"{file_name}.gpickle"

        if not graph_path.exists():
            skipped.append((file_name, cpu, "file_not_found"))
            continue

        with open(graph_path, 'rb') as f:
            fcg = pickle.load(f)

        if fcg.number_of_nodes() == 0:
            skipped.append((file_name, cpu, "empty_nodes"))
            continue

        if fcg.number_of_edges() == 0:
            skipped.append((file_name, cpu, "no_edges"))
            continue

        node_features = np.array([fcg.nodes[n]['x'] for n in fcg.nodes()])
        for n in fcg.nodes():
            fcg.nodes[n].pop('x', None)
        torch_data = from_networkx(fcg)
        torch_data.x = torch.tensor(node_features, dtype=torch.float)
        graphs.append(torch_data)
        labels.append(label)

    # Write skip log
    if skipped:
        with open(log_path, 'a') as f:
            skip_by_arch = Counter()
            skip_by_reason = Counter()
            f.write(f"\n{'='*60}\n")
            f.write(f"Skipped files: {len(skipped)} / {len(df)} total\n")
            f.write(f"{'='*60}\n")
            for fname, cpu, reason in skipped:
                f.write(f"  {fname}  CPU={cpu}  reason={reason}\n")
                skip_by_arch[cpu] += 1
                skip_by_reason[reason] += 1
            f.write(f"\nSkipped per architecture:\n")
            for arch, cnt in sorted(skip_by_arch.items()):
                f.write(f"  {arch}: {cnt}\n")
            f.write(f"\nSkipped per reason:\n")
            for reason, cnt in sorted(skip_by_reason.items()):
                f.write(f"  {reason}: {cnt}\n")
        print(f"Skipped {len(skipped)}/{len(df)} files (log: {log_path})")
        for arch, cnt in sorted(Counter(c for _, c, _ in skipped).items()):
            print(f"  {arch}: -{cnt}")

    return graphs, labels


def load_test_data_by_arch(csv_path, graph_dir, target_cpus, label_encoder, classification=False,
                           cache_file=None, force_reload=False):
    if cache_file and force_reload and os.path.exists(cache_file):
        os.remove(cache_file)

    if cache_file and os.path.exists(cache_file):
        print(f"載入測試資料快取: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    df = pd.read_csv(csv_path)
    test_graphs_by_arch = {}

    for cpu in target_cpus:
        cpu_df = df[df['CPU'] == cpu]
        print(f"載入 {cpu} 測試資料: {len(cpu_df)} 個樣本")

        graphs, labels = load_graphs_from_df(cpu_df, graph_dir, classification)

        encoded_labels = label_encoder.transform(labels)
        for i, data in enumerate(graphs):
            data.y = torch.tensor(encoded_labels[i], dtype=torch.long)

        test_graphs_by_arch[cpu] = graphs

    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(test_graphs_by_arch, f)
        print(f"測試資料已快取到: {cache_file}")

    return test_graphs_by_arch


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(train_loader.dataset)


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
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
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            prob = torch.softmax(out, dim=1)
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_prob.extend(prob.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
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


def create_gnn_scheduler(optimizer, scheduler_type, **kwargs):
    if scheduler_type == "step":
        step_size = kwargs.get("step_size", 30)
        gamma = kwargs.get("gamma", 0.5)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "plateau":
        patience = kwargs.get("patience", 10)
        factor = kwargs.get("factor", 0.5)
        min_lr = kwargs.get("min_lr", 1e-6)
        return ReduceLROnPlateau(optimizer, mode='min', patience=patience,
                                 factor=factor, min_lr=min_lr)
    elif scheduler_type == "cosine":
        T_max = kwargs.get("T_max", 100)
        eta_min = kwargs.get("eta_min", 1e-6)
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def plot_training_curves(train_losses, val_losses, val_accuracies, seed, save_dir="outputs/plots"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train / Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'loss_curves_{seed}.png'))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(val_accuracies, label='Val Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'val_accuracy_curve_{seed}.png'))
    plt.close()


def save_experiment_results(results_dict, save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    if 'all_results' in results_dict:
        results_df = pd.DataFrame(results_dict['all_results'])
        results_df.to_csv(os.path.join(save_dir, f'results_{timestamp}.csv'), index=False)

    summary_data = {k: v for k, v in results_dict.items() if k != 'all_results'}
    with open(os.path.join(save_dir, f'summary_{timestamp}.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)

    return timestamp
