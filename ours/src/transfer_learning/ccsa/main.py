import copy
import os
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

from configs.ccsa.baseline import get_ccsa_config
from src.gnn.models import GCN
from src.transfer_learning.ccsa.utils import (
    prepare_ccsa_data, CCSAPairDataset, ccsa_pair_collate_fn,
    train_ccsa_epoch, evaluate, test_model,
    plot_training_curves, save_experiment_results
)


def build_model(config, num_classes, device):
    model = GCN(
        num_node_features=config["num_node_features"],
        hidden_channels=config["hidden_channels"],
        output_channels=config["output_channels"],
        num_classes=num_classes,
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        pooling=config["pooling"],
    )
    return model.to(device)


def run_experiment(config, random_state, device):
    """執行一次 CCSA 實驗，使用指定的 random_state 抽 target samples"""

    # --- Load data ---
    src_train, src_val, tgt_train, tgt_test, label_encoder, num_classes = prepare_ccsa_data(config, random_state)

    # --- Create CCSA pair dataset & loader ---
    pair_dataset = CCSAPairDataset(
        src_train, tgt_train, neg_pair_ratio=config["neg_pair_ratio"]
    )

    collate_fn = partial(ccsa_pair_collate_fn,
                         source_graphs=src_train, target_graphs=tgt_train)

    pair_loader = DataLoader(
        pair_dataset, batch_size=config["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=config["pin_memory"]
    )

    val_loader = PyGDataLoader(
        src_val, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=config["pin_memory"]
    )

    target_test_loader = PyGDataLoader(
        tgt_test, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=config["pin_memory"]
    )

    target_train_loader = PyGDataLoader(
        tgt_train, batch_size=config["batch_size"], shuffle=False,
        num_workers=0, pin_memory=config["pin_memory"]
    )

    # --- Model / Optimizer ---
    model = build_model(config, num_classes, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    ce_criterion = torch.nn.CrossEntropyLoss()

    # --- Training loop ---
    train_losses = []
    val_losses_list = []
    val_accuracies = []

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    alpha = config["alpha"]
    margin = config["csa_margin"]
    epochs = config["epochs"]
    patience = config["patience"]

    for epoch in range(1, epochs + 1):
        train_loss, train_ce, train_csa = train_ccsa_epoch(
            model, pair_loader, optimizer, ce_criterion, device, alpha, margin
        )
        val_accuracy, val_loss = evaluate(model, val_loader, device)
        tgt_accuracy, _ = evaluate(model, target_test_loader, device)
        tgt_train_accuracy, _ = evaluate(model, target_train_loader, device)

        train_losses.append(train_loss)
        val_losses_list.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Early stopping 只看 source val loss（避免 target test leakage）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        print(f"[Epoch {epoch}] "
              f"Loss={train_loss:.4f} (CE={train_ce:.4f}, CSA={train_csa:.4f}) | "
              f"Src Val Acc={val_accuracy:.4f} | Tgt Few-shot Acc={tgt_train_accuracy:.4f} | Tgt Test Acc={tgt_accuracy:.4f}")

    # --- Load best model ---
    model.load_state_dict(best_model_state)

    # --- Save best model ---
    os.makedirs(config["model_output_dir"], exist_ok=True)
    save_path = os.path.join(config["model_output_dir"], f"ccsa_best_rs{random_state}.pt")
    torch.save(best_model_state, save_path)
    print(f"Model saved to: {save_path}")

    # --- Plot ---
    plot_training_curves(train_losses, val_losses_list, val_accuracies,
                         save_dir=config["plot_dir"], random_state=random_state)

    # --- Final evaluation ---
    print(f"\n--- Final Evaluation (random_state={random_state}) ---")
    target_results = test_model(model, target_test_loader, device, label_encoder)

    return target_results, best_model_state, label_encoder, num_classes


def main():
    config = get_ccsa_config()
    random_states = config["random_states"]

    mode = "Classification (family)" if config["classification"] else "Detection (label)"
    if "num_target_samples_per_class" in config:
        fewshot_desc = f"{config['num_target_samples_per_class']} per class"
    else:
        fewshot_desc = f"{config.get('num_target_samples', '?')} total"
    print(f"=== CCSA Transfer Learning ===")
    print(f"Mode: {mode}")
    print(f"Source: {config['source_cpus']} -> Target: {config['target_cpus']}")
    print(f"Target few-shot samples: {fewshot_desc}")
    print(f"Model: GCN | Pooling: {config['pooling']} | Layers: {config['num_layers']}")
    print(f"Dropout: {config['dropout']} | LR: {config['learning_rate']}")
    print(f"Alpha: {config['alpha']} | Margin: {config['csa_margin']}")
    print(f"Runs: {len(random_states)} (random_states: {random_states})")
    print()

    device = torch.device(config["device"] if torch.cuda.is_available() else 'cpu')
    os.makedirs(config["model_output_dir"], exist_ok=True)

    all_results = []

    for i, rs in enumerate(random_states):
        print(f"\n{'='*60}")
        print(f"Run {i + 1}/{len(random_states)}, random_state={rs}")
        print(f"{'='*60}")
        results, _, _, _ = \
            run_experiment(config, rs, device)
        all_results.append({'random_state': rs, **results})

    # --- Summary ---
    accs = [r['accuracy'] for r in all_results]
    f1_macros = [r['f1_macro'] for r in all_results]
    aucs = [r['auc'] for r in all_results]
    precisions = [r['precision'] for r in all_results]
    recalls = [r['recall'] for r in all_results]

    print(f"\n{'='*60}")
    print(f"Summary ({len(random_states)} runs)")
    print(f"Source: {config['source_cpus']} -> Target: {config['target_cpus']}")
    print(f"Target few-shot samples: {fewshot_desc}")
    print(f"{'='*60}")
    print(f"  Accuracy  : {np.mean(accs):.4f} +/- {np.std(accs):.4f}  (min={np.min(accs):.4f}, max={np.max(accs):.4f})")
    print(f"  Precision : {np.mean(precisions):.4f} +/- {np.std(precisions):.4f}")
    print(f"  Recall    : {np.mean(recalls):.4f} +/- {np.std(recalls):.4f}")
    print(f"  F1-macro  : {np.mean(f1_macros):.4f} +/- {np.std(f1_macros):.4f}")
    print(f"  AUC       : {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    for r in all_results:
        print(f"  rs={r['random_state']}: acc={r['accuracy']:.4f}, f1={r['f1_macro']:.4f}, auc={r['auc']:.4f}")

    # --- Save results ---
    results_summary = {
        'mode': mode,
        'source_cpus': config['source_cpus'],
        'target_cpus': config['target_cpus'],
        'target_fewshot': fewshot_desc,
        'alpha': config['alpha'],
        'csa_margin': config['csa_margin'],
        'model_type': 'GCN',
        'mean_accuracy': float(np.mean(accs)),
        'std_accuracy': float(np.std(accs)),
        'mean_f1_macro': float(np.mean(f1_macros)),
        'std_f1_macro': float(np.std(f1_macros)),
        'mean_auc': float(np.mean(aucs)),
        'std_auc': float(np.std(aucs)),
        'per_run': [
            {'random_state': r['random_state'],
             'accuracy': r['accuracy'],
             'f1_macro': r['f1_macro'],
             'auc': r['auc']}
            for r in all_results
        ],
    }

    timestamp = save_experiment_results(results_summary, save_dir=config["result_dir"])
    print(f"\nResults saved with timestamp: {timestamp}")


if __name__ == "__main__":
    main()
