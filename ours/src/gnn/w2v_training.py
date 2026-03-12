import copy
import torch
import random
import numpy as np
import os
import argparse
from collections import Counter
from torch_geometric.loader import DataLoader

from src.gnn.utils import (
    load_single_arch_data, load_cross_arch_data, train_epoch, evaluate,
    create_gnn_scheduler, test_model, plot_training_curves,
    save_experiment_results, load_test_data_by_arch
)
from src.gnn.models import GCN, GAT
from configs.gnn.w2v import get_w2v_gnn_config


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(config, num_classes, device):
    common_args = dict(
        num_node_features=config["num_node_features"],
        hidden_channels=config["hidden_channels"],
        output_channels=config["output_channels"],
        num_classes=num_classes,
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        pooling=config["pooling"],
    )

    if config["model_type"] == "GAT":
        model = GAT(**common_args, heads=config["gat_heads"])
    else:
        model = GCN(**common_args)

    return model.to(device)


def build_scheduler(optimizer, config):
    stype = config["scheduler_type"]
    if stype == "step":
        return create_gnn_scheduler(optimizer, stype,
                                    step_size=config["step_size"],
                                    gamma=config["gamma"])
    elif stype == "plateau":
        return create_gnn_scheduler(optimizer, stype,
                                    patience=config["plateau_patience"],
                                    factor=config["plateau_factor"])
    elif stype == "cosine":
        return create_gnn_scheduler(optimizer, stype,
                                    T_max=config["cosine_T_max"])
    else:
        raise ValueError(f"Unknown scheduler type: {stype}")


def run_experiment(seed, config, model_name):
    set_random_seed(seed)

    train_losses = []
    val_losses_list = []
    val_accuracies = []

    device = torch.device(config["device"] if torch.cuda.is_available() else 'cpu')
    model_save_dir = config["model_output_dir"]
    os.makedirs(model_save_dir, exist_ok=True)

    batch_size = config["batch_size"]
    lr = config["learning_rate"]
    epochs = config["epochs"]
    patience = config["patience"]

    # --- Load data ---
    if config["target_cpus"]:
        train_graphs, val_graphs, label_encoder, num_classes = load_cross_arch_data(
            csv_path=config["csv_path"],
            graph_dir=config["graph_dir"],
            source_cpus=config["source_cpus"],
            target_cpus=config["target_cpus"],
            cache_file=config["cache_file"],
            val_size=config["cross_arch_val_size"],
            random_state=config["random_state"],
            force_reload=config["force_reload"],
            classification=config["classification"]
        )
    else:
        train_graphs, val_graphs, test_graphs, label_encoder, num_classes = load_single_arch_data(
            csv_path=config["csv_path"],
            graph_dir=config["graph_dir"],
            source_cpus=config["source_cpus"],
            cache_file=config["cache_file"],
            val_size=config["single_arch_val_size"],
            test_size=config["single_arch_test_size"],
            random_state=config["random_state"],
            force_reload=config["force_reload"],
            classification=config["classification"]
        )

    # --- Sample distribution ---
    train_label_counts = Counter(int(g.y) for g in train_graphs)
    val_label_counts = Counter(int(g.y) for g in val_graphs)
    print(f"Train samples: {len(train_graphs)}, distribution: {dict(sorted(train_label_counts.items()))}")
    print(f"Val samples: {len(val_graphs)}, distribution: {dict(sorted(val_label_counts.items()))}")

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True,
                              num_workers=config["num_workers"], pin_memory=config["pin_memory"])
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False,
                            num_workers=config["num_workers"], pin_memory=config["pin_memory"])

    # --- Model / Optimizer / Scheduler ---
    model = build_model(config, num_classes, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = build_scheduler(optimizer, config)

    # --- Training loop ---
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_accuracy, val_loss = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses_list.append(val_loss)
        val_accuracies.append(val_accuracy)

        if config["scheduler_type"] == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] Val Accuracy = {val_accuracy:.4f}, Val Loss = {val_loss:.4f}")

    # --- Load best model ---
    model.load_state_dict(best_model_state)

    # --- Plot ---
    plot_training_curves(train_losses, val_losses_list, val_accuracies, seed,
                         save_dir=config["plot_dir"])

    # --- Save model ---
    mode_str = "classification" if config["classification"] else "detection"
    arch_str = "_".join(config["source_cpus"]) if config["source_cpus"] else "default"
    model_filename = f"gnn_{mode_str}_{arch_str}_{model_name}_seed_{seed}.pt"
    model_path = os.path.join(model_save_dir, model_filename)

    torch.save({
        'seed': seed,
        'model_name': model_name,
        'model_state_dict': best_model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'num_classes': num_classes,
        'label_encoder': label_encoder,
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses_list,
        'val_accuracies': val_accuracies
    }, model_path)
    print(f"Model saved to: {model_path} (best val loss: {best_val_loss:.4f})")

    # --- Evaluate ---
    if config["target_cpus"]:
        test_results_by_arch = {}
        test_graphs_by_arch = load_test_data_by_arch(
            config["csv_path"], config["graph_dir"],
            config["target_cpus"], label_encoder, config["classification"],
            cache_file=config.get("test_cache_file"),
            force_reload=config["force_reload"]
        )

        for cpu, graphs in test_graphs_by_arch.items():
            cpu_loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
            cpu_results = test_model(model, cpu_loader, device, label_encoder)
            test_results_by_arch[cpu] = cpu_results
            print(f"\n{cpu} Results:")
            print(f"  Accuracy:  {cpu_results['accuracy']:.4f}")
            print(f"  Precision: {cpu_results['precision']:.4f}")
            print(f"  Recall:    {cpu_results['recall']:.4f}")
            print(f"  F1-micro:  {cpu_results['f1_micro']:.4f}")
            print(f"  F1-macro:  {cpu_results['f1_macro']:.4f}")
            print(f"  AUC:       {cpu_results['auc']:.4f}")

        return test_results_by_arch
    else:
        test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
        test_results = test_model(model, test_loader, device, label_encoder)
        return {'overall': test_results}


def main():
    parser = argparse.ArgumentParser(description='GNN training with W2V embeddings')
    parser.add_argument('--model', type=str, required=True,
                        choices=['fasttext', 'cbow', 'skipgram'],
                        help='Word2Vec model type')
    args = parser.parse_args()

    config = get_w2v_gnn_config(args.model)
    seed = config["seed"]

    mode = "Classification (family)" if config["classification"] else "Detection (label)"
    arch_mode = "single-arch" if not config["target_cpus"] else "cross-arch"

    print(f"W2V Model: {args.model}")
    print(f"Mode: {mode}")
    print(f"Arch Mode: {arch_mode}")
    print(f"Model: {config['model_type']} | Pooling: {config['pooling']} | Layers: {config['num_layers']}")
    print(f"Training Architecture: {config['source_cpus']}")
    if config['target_cpus']:
        print(f"Testing Architecture: {config['target_cpus']}")
    print(f"Graph Data Directory: {config['graph_dir']}")

    print(f"\n{'='*60}")
    print(f"Seed = {seed}")
    print(f"{'='*60}")
    results = run_experiment(seed, config, args.model)

    # --- Save results ---
    results_summary = {
        'model_name': args.model,
        'mode': mode,
        'arch_mode': arch_mode,
        'source_cpus': config['source_cpus'],
        'target_cpus': config['target_cpus'],
        'model_type': config['model_type'],
        'seed': seed,
    }

    if config["target_cpus"]:
        for cpu, metrics in results.items():
            print(f"\n{cpu}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  F1-macro:  {metrics['f1_macro']:.4f}")
            print(f"  AUC:       {metrics['auc']:.4f}")

        results_summary['results_by_arch'] = {
            cpu: {k: v for k, v in m.items()
                  if k in ('accuracy', 'f1_micro', 'f1_macro', 'auc', 'precision', 'recall')}
            for cpu, m in results.items()
        }
    else:
        overall = results['overall']
        print(f"\nOverall:")
        print(f"  Accuracy:  {overall['accuracy']:.4f}")
        print(f"  F1-macro:  {overall['f1_macro']:.4f}")
        print(f"  AUC:       {overall['auc']:.4f}")

        results_summary.update({
            k: v for k, v in overall.items()
            if k in ('accuracy', 'f1_micro', 'f1_macro', 'auc', 'precision', 'recall')
        })

    timestamp = save_experiment_results(results_summary, save_dir=config["result_dir"])
    print(f"\nResults saved with timestamp: {timestamp}")


if __name__ == "__main__":
    main()
