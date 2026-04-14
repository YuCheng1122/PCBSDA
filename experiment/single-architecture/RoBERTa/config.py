def get_roberta_single_config(arch="x86_64", roberta_tag="roberta_20"):
    """
    Single-architecture family classification config for RoBERTa-based embeddings.
    arch: one of ["Intel", "ARM-32", "x86_64", "MIPS"]
    roberta_tag: embedding folder name, e.g. "roberta_20"
    """
    BASE_PATH = "/home/tommy/Project/PCBSDA"

    arch_tag = arch.replace("-", "_").lower()

    config = {
        # Task mode
        "classification": True,
        "source_cpus": [arch],

        # Data paths
        "csv_path": f"{BASE_PATH}/datasets/csv/single_arch_dataset.csv",
        "graph_dir": f"{BASE_PATH}/ours/outputs/embedded_graphs/{roberta_tag}",

        # Cross-Validation
        "n_splits": 5,
        "random_state": 42,

        # Optuna
        "n_trials": 20,
        "optuna_timeout": None,
        "test_size": 0.2,
        "optuna_n_splits": 5,

        # Model fixed settings (from gnn_single.py)
        "model_type": "GAT",
        "pooling": "attention",
        "gat_heads": 4,
        "scheduler_type": "step",
        "num_node_features": 256,   # roberta_20 output dim
        "num_classes": None,
        "hidden_channels": 256,
        "output_channels": 256,
        "num_layers": 2,
        "dropout": 0.2,
        "batch_size": 32,
        "learning_rate": 0.001,

        # Optuna search space (only the impactful parameters)
        "search_space": {
            "learning_rate": [1e-4, 1e-2],  # log-uniform
            "num_layers": [1, 2, 3],
            "dropout": [0.1, 0.5],          # uniform float
        },

        # Training (fixed)
        "num_workers": 4,
        "pin_memory": True,
        "epochs": 200,
        "patience": 15,

        # Scheduler fixed params
        "step_size": 30,
        "gamma": 0.5,
        "plateau_patience": 10,
        "plateau_factor": 0.5,
        "cosine_T_max": 100,

        "device": "cuda",

        # Output paths
        "optuna_dir": f"{BASE_PATH}/experiment/outputs/optuna/roberta/{roberta_tag}/{arch_tag}",
        "result_dir": f"{BASE_PATH}/experiment/outputs/results/roberta/{roberta_tag}/{arch_tag}",
        "log_dir":    f"{BASE_PATH}/experiment/outputs/logs/roberta/{roberta_tag}/{arch_tag}",
        "plot_dir":   f"{BASE_PATH}/experiment/outputs/plots/roberta/{roberta_tag}/{arch_tag}",
    }

    return config


ALL_ARCHS = ["Intel", "ARM-32", "x86_64", "MIPS"]
ROBERTA_TAGS = ["roberta_20"]
