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
        "n_trials": 50,
        "optuna_timeout": None,

        # Default model (RoBERTa → GAT is the default choice)
        "model_type": "GAT",
        "num_node_features": 256,   # roberta_20 output dim
        "num_classes": None,

        # Optuna search space
        "search_space": {
            "hidden_channels": [64, 128, 256],
            "output_channels": [64, 128, 256],
            "num_layers": [1, 2, 3],
            "dropout": [0.0, 0.5],
            "batch_size": [16, 32, 64],
            "learning_rate": [1e-4, 1e-2],
            "pooling": ["add", "attention"],
            "model_type": ["GCN", "GAT"],
            "gat_heads": [2, 4, 8],
            "scheduler_type": ["step", "plateau", "cosine"],
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
        "optuna_dir": f"{BASE_PATH}/ours/outputs/optuna/{roberta_tag}/{arch_tag}",
        "result_dir": f"{BASE_PATH}/ours/outputs/results/single_arch_cv/{roberta_tag}/{arch_tag}",
        "log_dir": f"{BASE_PATH}/ours/outputs/logs/single_arch_cv/{roberta_tag}/{arch_tag}",
        "plot_dir": f"{BASE_PATH}/ours/outputs/plots/single_arch_cv/{roberta_tag}/{arch_tag}",
    }

    return config


ALL_ARCHS = ["Intel", "ARM-32", "x86_64", "MIPS"]
ROBERTA_TAGS = ["roberta_20"]
