import json
import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter

from configs.ml.run_all_config import get_ml_run_all_config
from src.ml.main import (
    graphs_to_vectors, sample_fewshot, evaluate,
    run_single, build_classifiers,
)
from src.gnn.utils import load_graphs_from_df


def load_target_data(config, target_cpus):
    cpu_tag = "+".join(target_cpus).replace("/", "_")
    cache_file = os.path.join(config["cache_dir"], f"ml_target_{cpu_tag}.pkl")

    if not config["force_reload"] and os.path.exists(cache_file):
        print(f"載入快取: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    from sklearn.preprocessing import LabelEncoder
    df = pd.read_csv(config["csv_path"])
    target_df = df[df["CPU"].isin(target_cpus)]
    print(f"載入 {target_cpus} 資料: {len(target_df)} 個樣本")

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


def run_target(config, target_cpus):
    cpu_tag = "+".join(target_cpus)
    result_dir = os.path.join(config["result_dir"], cpu_tag.lower())
    os.makedirs(result_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Target: {target_cpus}")
    print(f"Few-shot per class: {config['num_target_samples_per_class']}")
    print(f"Runs: {len(config['random_states'])} random seeds")
    print(f"{'='*60}")

    data = load_target_data(config, target_cpus)
    graphs = data["graphs"]
    labels = data["labels"]
    label_encoder = data["label_encoder"]

    print(f"Total samples: {len(graphs)}")
    print(f"Class distribution: {dict(Counter(labels))}")
    print(f"Num classes: {len(label_encoder.classes_)}")

    X_all = graphs_to_vectors(graphs, pooling=config["pooling"])
    clfs = build_classifiers(config)

    all_results = {label: [] for label, _ in clfs}

    for rs in config["random_states"]:
        print(f"\n--- seed={rs} ---")
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
            all_results[label].append({"random_state": rs, **metrics})
            print(f"  [{label}] acc={metrics['accuracy']:.4f}  f1={metrics['f1_macro']:.4f}  auc={metrics['auc']:.4f}")

    # Summary
    print(f"\n--- Summary: {target_cpus} ---")
    summary = {}
    for label in all_results:
        accs  = [r["accuracy"]  for r in all_results[label]]
        f1s   = [r["f1_macro"]  for r in all_results[label]]
        aucs  = [r["auc"]       for r in all_results[label]]
        precs = [r["precision"] for r in all_results[label]]
        recs  = [r["recall"]    for r in all_results[label]]

        print(f"  [{label}] acc={np.mean(accs):.4f}±{np.std(accs):.4f}  "
              f"f1={np.mean(f1s):.4f}±{np.std(f1s):.4f}  "
              f"auc={np.mean(aucs):.4f}±{np.std(aucs):.4f}")

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
            "per_seed": all_results[label],
        }

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(result_dir, f"ml_summary_{timestamp}.json")
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {save_path}")

    return summary


def main():
    config = get_ml_run_all_config()

    print(f"\n{'#'*60}")
    print(f"# ML Run All — RF_200_sqrt")
    print(f"# Targets: {config['target_cpus_list']}")
    print(f"# Seeds: {config['random_states']}")
    print(f"{'#'*60}")

    all_summaries = {}
    for target_cpus in config["target_cpus_list"]:
        summary = run_target(config, target_cpus)
        all_summaries["+".join(target_cpus)] = summary

    # Global summary table
    print(f"\n{'#'*60}")
    print(f"# Final Summary — RF_200_sqrt")
    print(f"{'#'*60}")
    print(f"  {'Target':<12} {'Acc':>8} {'±':>6} {'F1':>8} {'±':>6} {'AUC':>8} {'±':>6}")
    print(f"  {'-'*54}")
    for target, summary in all_summaries.items():
        s = summary["RF_200_sqrt"]
        print(f"  {target:<12} "
              f"{s['mean_accuracy']:>8.4f} {s['std_accuracy']:>6.4f} "
              f"{s['mean_f1_macro']:>8.4f} {s['std_f1_macro']:>6.4f} "
              f"{s['mean_auc']:>8.4f} {s['std_auc']:>6.4f}")


if __name__ == "__main__":
    main()
