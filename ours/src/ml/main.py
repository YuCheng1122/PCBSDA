import json
import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from configs.ml.baseline import get_ml_config
from src.gnn.utils import load_graphs_from_df


def load_target_data(config):
    cache_file = config["target_cache_file"]

    if not config["force_reload"] and os.path.exists(cache_file):
        print(f"載入快取: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    df = pd.read_csv(config["csv_path"])
    target_df = df[df["CPU"].isin(config["target_cpus"])]
    print(f"載入 {config['target_cpus']} 資料: {len(target_df)} 個樣本")

    graphs, labels = load_graphs_from_df(
        target_df, config["graph_dir"], config["classification"]
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    encoded = label_encoder.transform(labels)

    data = {"graphs": graphs, "labels": encoded, "label_encoder": label_encoder}
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)
    print(f"已快取到: {cache_file}")

    return data


def graphs_to_vectors(graphs, pooling="mean"):
    """把 PyG graph list 轉成 numpy feature matrix，對 node embeddings 做 pooling"""
    vectors = []
    for g in graphs:
        x = g.x.numpy()   # (num_nodes, feat_dim)
        if pooling == "mean":
            vec = x.mean(axis=0)
        else:
            vec = x.sum(axis=0)
        vectors.append(vec)
    return np.array(vectors)


def sample_fewshot(graphs, labels, n_per_class, rng):
    """每個 class 抽 n_per_class 個，其餘當 test"""
    train_idx, test_idx = [], []
    for c in sorted(set(labels)):
        c_idx = [i for i, y in enumerate(labels) if y == c]
        n = min(n_per_class, len(c_idx))
        chosen = rng.choice(c_idx, n, replace=False).tolist()
        train_idx.extend(chosen)
        test_idx.extend([i for i in c_idx if i not in set(chosen)])
    return train_idx, test_idx


def evaluate(y_true, y_prob, y_pred, label_encoder):
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    if len(label_encoder.classes_) == 2:
        auc = roc_auc_score(y_true, y_prob[:, 1])
    else:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "precision": precision,
        "recall": recall,
        "auc": auc,
    }


def run_single(clf, X_train, y_train, X_test, y_test, label_encoder):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    return evaluate(y_test, y_prob, y_pred, label_encoder)


def build_classifiers(config):
    """依 config 建立所有要跑的 (label, clf) pairs"""
    clfs = []

    for c in config["rf_configs"]:
        clf = RandomForestClassifier(
            n_estimators=c["n_estimators"],
            max_depth=c["max_depth"],
            min_samples_split=c["min_samples_split"],
            max_features=c.get("max_features", "sqrt"),
            random_state=0,
            n_jobs=-1,
        )
        clfs.append((c["label"], clf))

    for c in config["xgb_configs"]:
        clf = XGBClassifier(
            n_estimators=c["n_estimators"],
            max_depth=c["max_depth"],
            learning_rate=c["learning_rate"],
            subsample=c["subsample"],
            random_state=0,
            eval_metric="mlogloss",
            verbosity=0,
        )
        clfs.append((c["label"], clf))

    for c in config["svm_configs"]:
        clf = SVC(
            kernel=c["kernel"],
            C=c["C"],
            gamma=c["gamma"],
            probability=True,
        )
        clfs.append((c["label"], clf))

    return clfs


def main():
    config = get_ml_config()
    os.makedirs(config["result_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)

    print("=== Few-shot ML Baseline ===")
    print(f"Target: {config['target_cpus']}")
    print(f"Few-shot per class: {config['num_target_samples_per_class']}")
    print(f"Runs: {len(config['random_states'])} random seeds")
    print(f"Pooling: {config['pooling']}")
    print()

    # Load data
    data = load_target_data(config)
    graphs = data["graphs"]
    labels = data["labels"]
    label_encoder = data["label_encoder"]

    print(f"Total target samples: {len(graphs)}")
    print(f"Class distribution: {dict(Counter(labels))}")
    print(f"Num classes: {len(label_encoder.classes_)}")
    print()

    # Graph → vector
    X_all = graphs_to_vectors(graphs, pooling=config["pooling"])

    # Build classifiers
    clfs = build_classifiers(config)
    print(f"Classifiers to run: {[label for label, _ in clfs]}")
    print()

    # Accumulate results per classifier label
    all_results = {label: [] for label, _ in clfs}

    for rs in config["random_states"]:
        print(f"--- random_state={rs} ---")
        rng = np.random.RandomState(rs)
        train_idx, test_idx = sample_fewshot(
            graphs, labels, config["num_target_samples_per_class"], rng
        )

        X_train = X_all[train_idx]
        y_train = labels[train_idx]
        X_test  = X_all[test_idx]
        y_test  = labels[test_idx]

        print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
        print(f"  Train dist: {dict(Counter(y_train))}")

        for label, clf in clfs:
            metrics = run_single(clf, X_train, y_train, X_test, y_test, label_encoder)
            all_results[label].append(metrics)
            print(f"  [{label}] acc={metrics['accuracy']:.4f}  f1={metrics['f1_macro']:.4f}  auc={metrics['auc']:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary (mean ± std over random seeds)")
    print(f"Target: {config['target_cpus']}  |  {config['num_target_samples_per_class']} samples/class")
    print(f"{'='*60}")

    summary = {}
    for label in all_results:
        accs  = [r["accuracy"]  for r in all_results[label]]
        f1s   = [r["f1_macro"]  for r in all_results[label]]
        aucs  = [r["auc"]       for r in all_results[label]]
        precs = [r["precision"] for r in all_results[label]]
        recs  = [r["recall"]    for r in all_results[label]]

        print(f"\n[{label}]")
        print(f"  Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  F1-macro : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"  AUC      : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"  Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
        print(f"  Recall   : {np.mean(recs):.4f} ± {np.std(recs):.4f}")

        summary[label] = {
            "mean_accuracy":  float(np.mean(accs)),
            "std_accuracy":   float(np.std(accs)),
            "mean_f1_macro":  float(np.mean(f1s)),
            "std_f1_macro":   float(np.std(f1s)),
            "mean_auc":       float(np.mean(aucs)),
            "std_auc":        float(np.std(aucs)),
            "mean_precision": float(np.mean(precs)),
            "std_precision":  float(np.std(precs)),
            "mean_recall":    float(np.mean(recs)),
            "std_recall":     float(np.std(recs)),
            "per_seed": [
                {"random_state": rs, **r}
                for rs, r in zip(config["random_states"], all_results[label])
            ],
        }

    # Save
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(config["result_dir"], f"ml_summary_{timestamp}.json")
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    main()
