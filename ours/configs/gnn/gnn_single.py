def get_gnn_single_config(arch="x86_64"):
    """
    Single-architecture family classification config.
    arch: one of ["Intel", "ARM-32", "x86_64", "MIPS"]
    """
    BASE_PATH = "/home/tommy/Project/PCBSDA"
    EMBEDDING = "roberta_20"  # "cbow", "skipgram", "fast_text", "roberta"

    arch_tag = arch.replace("-", "_").lower()

    config = {
        # Task mode
        "classification": True,
        "source_cpus": [arch],
        "target_cpus": [],      # 空 = 單架構模式，使用 train/val/test split

        # Data paths
        "csv_path": f"{BASE_PATH}/datasets/csv/single_arch_dataset.csv",
        "graph_dir": f"{BASE_PATH}/ours/outputs/embedded_graphs/{EMBEDDING}",
        "cache_file": f"{BASE_PATH}/ours/outputs/cache/gnn_single/{EMBEDDING}/{arch_tag}.pkl",
        "test_cache_file": f"{BASE_PATH}/ours/outputs/cache/gnn_single/{EMBEDDING}/{arch_tag}_test.pkl",

        # Data split (單架構: train/val/test = 7/2/1)
        "single_arch_val_size": 0.2,
        "single_arch_test_size": 0.2,
        "cross_arch_val_size": 0.2,
        "random_state": 42,
        "force_reload": True,

        # Model architecture
        "model_type": "GAT",            # "GCN" or "GAT"
        "num_node_features": 256,
        "hidden_channels": 256,
        "output_channels": 256,
        "num_layers": 2,
        "dropout": 0.2,
        "pooling": "attention",               # "add" or "attention"

        # GAT specific
        "gat_heads": 4,
        "gat_dropout": 0.2,

        # Training
        "num_workers": 4,
        "pin_memory": True,
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 200,
        "patience": 15,

        # Scheduler
        "scheduler_type": "step",       # "step", "plateau", "cosine"
        "step_size": 30,
        "gamma": 0.5,
        "plateau_patience": 10,
        "plateau_factor": 0.5,
        "cosine_T_max": 100,

        # Seeds
        "seeds": [42, 123, 57],
        "device": "cuda",

        # Output paths
        "model_output_dir": f"{BASE_PATH}/ours/outputs/models/gnn_single/{EMBEDDING}/{arch_tag}",
        "plot_dir": f"{BASE_PATH}/ours/outputs/plots/gnn_single/{EMBEDDING}/{arch_tag}",
        "result_dir": f"{BASE_PATH}/ours/outputs/results/gnn_single/{EMBEDDING}/{arch_tag}",
        "log_dir": f"{BASE_PATH}/ours/outputs/logs/gnn_single/{EMBEDDING}/{arch_tag}",
    }

    return config


# 所有目標架構，main_single.py 依序執行
ALL_ARCHS = ["Intel", "ARM-32", "x86_64", "MIPS"]
