import copy
import logging
import os

import numpy as np
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from configs.dann.baseline import get_dann_config
from src.transfer_learning.dann.models import GCN_DANN, GAT_DANN
from src.transfer_learning.dann.utils import (
    prepare_dann_data, compute_alpha,
    train_dann_epoch, evaluate, test_model,
    plot_training_curves, save_experiment_results,
)


def setup_logger(log_dir, name="dann"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def build_model(config, num_classes, device):
    model_type = config.get("model_type", "GCN")
    common_kwargs = dict(
        num_node_features=config["num_node_features"],
        hidden_channels=config["hidden_channels"],
        output_channels=config["output_channels"],
        num_classes=num_classes,
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        pooling=config["pooling"],
    )
    if model_type == "GAT":
        model = GAT_DANN(**common_kwargs, heads=config.get("gat_heads", 4))
    else:
        model = GCN_DANN(**common_kwargs)
    return model.to(device)


def run_experiment(config, random_state, dirs, logger):
    """執行一次 DANN 實驗，使用指定的 random_state 抽 target samples"""

    # --- Load data ---
    src_train, src_val, _, tgt_test, tgt_all, label_encoder, num_classes = \
        prepare_dann_data(config, random_state)

    # --- DataLoaders ---
    src_loader = PyGDataLoader(
        src_train, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"], pin_memory=config["pin_memory"]
    )
    # DANN domain alignment: 全量 target（unlabeled），class label 不使用
    tgt_domain_loader = PyGDataLoader(
        tgt_all, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"], pin_memory=config["pin_memory"]
    )
    val_loader = PyGDataLoader(
        src_val, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=config["pin_memory"]
    )
    target_test_loader = PyGDataLoader(
        tgt_test, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=config["pin_memory"]
    )

    # --- Model / Optimizer ---
    device = config["device"]
    model = build_model(config, num_classes, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    ce_criterion = torch.nn.CrossEntropyLoss()
    domain_criterion = torch.nn.CrossEntropyLoss()

    # --- Hyperparams ---
    lambda_domain = config["lambda_domain"]
    epochs = config["epochs"]
    patience = config["patience"]

    train_losses = []
    val_losses_list = []
    val_accuracies = []

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    logger.info(f"\n--- Training (seed={random_state}) ---")
    header = (f"{'Epoch':>6}  {'Loss':>8}  {'CE_src':>8}  {'Dom_src':>8}  "
              f"{'Dom_tgt':>8}  {'Alpha':>7}  {'SrcValAcc':>10}")
    logger.info(header)
    logger.info("-" * len(header))

    for epoch in range(1, epochs + 1):
        alpha = compute_alpha(epoch, epochs)

        train_loss, ce_src, dom_src, dom_tgt = train_dann_epoch(
            model, src_loader, tgt_domain_loader, optimizer,
            ce_criterion, domain_criterion, device,
            alpha=alpha, lambda_domain=lambda_domain,
        )

        val_accuracy, val_loss = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses_list.append(val_loss)
        val_accuracies.append(val_accuracy)

        logger.info(
            f"{epoch:>6}  {train_loss:>8.4f}  {ce_src:>8.4f}  {dom_src:>8.4f}  "
            f"{dom_tgt:>8.4f}  {alpha:>7.4f}  {val_accuracy:>10.4f}"
        )

        # Early stopping on source val loss（避免 target test leakage）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # --- Load best model ---
    model.load_state_dict(best_model_state)

    # --- Save best model ---
    os.makedirs(dirs["model_output_dir"], exist_ok=True)
    save_path = os.path.join(dirs["model_output_dir"], f"dann_best_rs{random_state}.pt")
    torch.save(best_model_state, save_path)
    logger.info(f"Model saved to: {save_path}")

    # --- Plot ---
    plot_training_curves(train_losses, val_losses_list, val_accuracies,
                         save_dir=dirs["plot_dir"], random_state=random_state)

    # --- Final evaluation ---
    logger.info(f"\n--- Final Evaluation (seed={random_state}) ---")
    target_results = test_model(model, target_test_loader, device, label_encoder)
    logger.info(f"Report:\n{target_results['classification_report']}")

    return target_results, best_model_state, label_encoder, num_classes


def main():
    config = get_dann_config()
    random_states = config["random_states"]

    model_type = config.get("model_type", "GCN")
    dirs = {
        "model_output_dir": config["model_output_dir"],
        "plot_dir":         config["plot_dir"],
        "result_dir":       config["result_dir"],
        "log_dir":          config["log_dir"],
    }

    device = torch.device(config["device"] if torch.cuda.is_available() else 'cpu')
    config["device"] = device

    logger = setup_logger(dirs["log_dir"], name="dann")

    mode = "Classification (family)" if config["classification"] else "Detection (label)"
    if "num_target_samples_per_class" in config:
        fewshot_desc = f"{config['num_target_samples_per_class']} per class"
    else:
        fewshot_desc = f"{config.get('num_target_samples', '?')} total"

    logger.info(f"=== DANN Transfer Learning ===")
    logger.info(f"Mode: {mode}")
    logger.info(f"Source: {config['source_cpus']} -> Target: {config['target_cpus']}")
    logger.info(f"Target few-shot samples: {fewshot_desc}")
    logger.info(f"Model: {model_type} | Pooling: {config['pooling']} | Layers: {config['num_layers']}")
    logger.info(f"Dropout: {config['dropout']} | LR: {config['learning_rate']}")
    logger.info(f"Lambda_domain: {config['lambda_domain']}")
    logger.info(f"Runs: {len(random_states)} (seeds: {random_states})")
    logger.info("")

    os.makedirs(dirs["model_output_dir"], exist_ok=True)

    all_results = []

    for i, rs in enumerate(random_states):
        logger.info(f"\n{'='*60}")
        logger.info(f"Run {i + 1}/{len(random_states)}, seed={rs}")
        logger.info(f"{'='*60}")
        results, _, _, _ = run_experiment(config, rs, dirs, logger)
        all_results.append({'random_state': rs, **results})

    # --- Summary ---
    accs       = [r['accuracy']  for r in all_results]
    f1_macros  = [r['f1_macro']  for r in all_results]
    aucs       = [r['auc']       for r in all_results]
    precisions = [r['precision'] for r in all_results]
    recalls    = [r['recall']    for r in all_results]

    logger.info(f"\n{'='*60}")
    logger.info(f"Summary ({len(random_states)} runs)")
    logger.info(f"Source: {config['source_cpus']} -> Target: {config['target_cpus']}")
    logger.info(f"Target few-shot samples: {fewshot_desc}")
    logger.info(f"{'='*60}")
    logger.info(f"  Accuracy  : {np.mean(accs):.4f} +/- {np.std(accs):.4f}  (min={np.min(accs):.4f}, max={np.max(accs):.4f})")
    logger.info(f"  Precision : {np.mean(precisions):.4f} +/- {np.std(precisions):.4f}")
    logger.info(f"  Recall    : {np.mean(recalls):.4f} +/- {np.std(recalls):.4f}")
    logger.info(f"  F1-macro  : {np.mean(f1_macros):.4f} +/- {np.std(f1_macros):.4f}")
    logger.info(f"  AUC       : {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    logger.info("")
    for r in all_results:
        logger.info(f"  seed={r['random_state']}: acc={r['accuracy']:.4f}, f1={r['f1_macro']:.4f}, auc={r['auc']:.4f}")

    results_summary = {
        'mode': mode,
        'source_cpus': config['source_cpus'],
        'target_cpus': config['target_cpus'],
        'target_fewshot': fewshot_desc,
        'lambda_domain': config['lambda_domain'],
        'model_type': model_type,
        'pooling': config['pooling'],
        'mean_accuracy': float(np.mean(accs)),
        'std_accuracy':  float(np.std(accs)),
        'mean_f1_macro': float(np.mean(f1_macros)),
        'std_f1_macro':  float(np.std(f1_macros)),
        'mean_auc':      float(np.mean(aucs)),
        'std_auc':       float(np.std(aucs)),
        'per_run': [
            {'random_state': r['random_state'],
             'accuracy': r['accuracy'],
             'f1_macro': r['f1_macro'],
             'auc': r['auc']}
            for r in all_results
        ],
    }

    timestamp = save_experiment_results(results_summary, save_dir=dirs["result_dir"])
    logger.info(f"\nResults saved with timestamp: {timestamp}")


if __name__ == "__main__":
    main()
