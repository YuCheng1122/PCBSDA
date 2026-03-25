"""
Single-architecture family classification.
Loops over all 4 architectures (or a specified subset) using gnn_single config.
"""

import argparse
import numpy as np
from src.gnn.main import run_experiment, set_random_seed
from src.gnn.utils import save_experiment_results
from configs.gnn.gnn_single import get_gnn_single_config, ALL_ARCHS


def run_single_arch(arch, verbose=True):
    config = get_gnn_single_config(arch)
    seeds = config["seeds"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Arch: {arch}")
        print(f"Model: {config['model_type']} | Pooling: {config['pooling']} | Layers: {config['num_layers']}")
        print(f"Scheduler: {config['scheduler_type']}")
        print(f"{'='*60}")

    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n[{arch}] Experiment {i+1}/{len(seeds)}, Seed={seed}")
        results = run_experiment(seed, config)
        all_results.append(results)

    # Aggregate
    overall_accs       = [r['overall']['accuracy']  for r in all_results]
    overall_precisions = [r['overall']['precision'] for r in all_results]
    overall_recalls    = [r['overall']['recall']    for r in all_results]
    overall_f1_micros  = [r['overall']['f1_micro']  for r in all_results]
    overall_f1_macros  = [r['overall']['f1_macro']  for r in all_results]
    overall_aucs       = [r['overall']['auc']       for r in all_results]

    print(f"\n[{arch}] {len(seeds)} Experiments Summary:")
    print(f"  Accuracy  : {np.mean(overall_accs):.4f} ± {np.std(overall_accs):.4f}")
    print(f"  Precision : {np.mean(overall_precisions):.4f} ± {np.std(overall_precisions):.4f}")
    print(f"  Recall    : {np.mean(overall_recalls):.4f} ± {np.std(overall_recalls):.4f}")
    print(f"  F1-micro  : {np.mean(overall_f1_micros):.4f} ± {np.std(overall_f1_micros):.4f}")
    print(f"  F1-macro  : {np.mean(overall_f1_macros):.4f} ± {np.std(overall_f1_macros):.4f}")
    print(f"  AUC       : {np.mean(overall_aucs):.4f} ± {np.std(overall_aucs):.4f}")

    results_summary = {
        'mode': 'Classification (family)',
        'arch_mode': '單架構',
        'arch': arch,
        'source_cpus': config['source_cpus'],
        'target_cpus': [],
        'model_type': config['model_type'],
        'pooling': config['pooling'],
        'num_layers': config['num_layers'],
        'hidden_channels': config['hidden_channels'],
        'output_channels': config['output_channels'],
        'scheduler': config['scheduler_type'],
        'avg_accuracy': np.mean(overall_accs),
        'std_accuracy': np.std(overall_accs),
        'avg_precision': np.mean(overall_precisions),
        'std_precision': np.std(overall_precisions),
        'avg_recall': np.mean(overall_recalls),
        'std_recall': np.std(overall_recalls),
        'avg_f1_micro': np.mean(overall_f1_micros),
        'std_f1_micro': np.std(overall_f1_micros),
        'avg_f1_macro': np.mean(overall_f1_macros),
        'std_f1_macro': np.std(overall_f1_macros),
        'avg_auc': np.mean(overall_aucs),
        'std_auc': np.std(overall_aucs),
        'seeds': seeds,
        'all_results': [
            {'seed': seed, 'accuracy': r['overall']['accuracy'],
             'precision': r['overall']['precision'],
             'recall': r['overall']['recall'],
             'f1_micro': r['overall']['f1_micro'],
             'f1_macro': r['overall']['f1_macro'],
             'auc': r['overall']['auc']}
            for seed, r in zip(seeds, all_results)
        ]
    }

    timestamp = save_experiment_results(results_summary, save_dir=config["result_dir"])
    print(f"[{arch}] Results saved: {timestamp}")
    return results_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default=None,
                        help=f"Single arch to run. If omitted, runs all: {ALL_ARCHS}")
    args = parser.parse_args()

    archs = [args.arch] if args.arch else ALL_ARCHS

    all_arch_summaries = {}
    for arch in archs:
        summary = run_single_arch(arch)
        all_arch_summaries[arch] = summary

    print(f"\n{'='*60}")
    print("All Architectures Final Summary:")
    print(f"{'='*60}")
    for arch, s in all_arch_summaries.items():
        print(f"  {arch:8s}  Acc: {s['avg_accuracy']:.4f}±{s['std_accuracy']:.4f}  "
              f"F1-macro: {s['avg_f1_macro']:.4f}±{s['std_f1_macro']:.4f}  "
              f"AUC: {s['avg_auc']:.4f}±{s['std_auc']:.4f}")


if __name__ == "__main__":
    main()
