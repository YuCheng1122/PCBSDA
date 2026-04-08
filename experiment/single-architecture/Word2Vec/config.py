def get_w2v_single_config(arch="x86_64", w2v_model="cbow"):
    """
    Single-architecture family classification config for Word2Vec-based embeddings.
    arch: one of ["Intel", "ARM-32", "x86_64", "MIPS"]
    w2v_model: "cbow", "skipgram", or "fast_text"
    """
    BASE_PATH = "/home/tommy/Project/PCBSDA"

    arch_tag = arch.replace("-", "_").lower()

    config = {
        # Task mode
        "classification": True,
        "source_cpus": [arch],

        # Data paths
        "csv_path": f"{BASE_PATH}/datasets/csv/single_arch_dataset.csv",
        "graph_dir": f"{BASE_PATH}/ours/outputs/embedded_graphs/{w2v_model}",

        # Cross-Validation
        "n_splits": 5,          # K-Fold CV
        "random_state": 42,

        # Optuna
        "n_trials": 50,
        "optuna_timeout": None,  # seconds, None = no limit

        # Default model architecture (starting point / search space center)
        "model_type": "GCN",
        "num_node_features": 256,
        "num_classes": None,     # auto-detected from data

        # Optuna search space bounds
        "search_space": {
            "hidden_channels": [64, 128, 256],
            "output_channels": [64, 128, 256],
            "num_layers": [1, 2, 3],
            "dropout": [0.0, 0.5],          # uniform float
            "batch_size": [16, 32, 64],
            "learning_rate": [1e-4, 1e-2],  # log-uniform
            "pooling": ["add", "attention"],
            "model_type": ["GCN", "GAT"],
            "gat_heads": [2, 4, 8],         # only used when model_type=GAT
            "scheduler_type": ["step", "plateau", "cosine"],
        },

        # Training (fixed)
        "num_workers": 4,
        "pin_memory": True,
        "epochs": 200,
        "patience": 15,

        # Scheduler fixed params (non-searched)
        "step_size": 30,
        "gamma": 0.5,
        "plateau_patience": 10,
        "plateau_factor": 0.5,
        "cosine_T_max": 100,

        "device": "cuda",

        # Output paths
        "optuna_dir": f"{BASE_PATH}/ours/outputs/optuna/{w2v_model}/{arch_tag}",
        "result_dir": f"{BASE_PATH}/ours/outputs/results/single_arch_cv/{w2v_model}/{arch_tag}",
        "log_dir": f"{BASE_PATH}/ours/outputs/logs/single_arch_cv/{w2v_model}/{arch_tag}",
        "plot_dir": f"{BASE_PATH}/ours/outputs/plots/single_arch_cv/{w2v_model}/{arch_tag}",
    }

    return config


ALL_ARCHS = ["Intel", "ARM-32", "x86_64", "MIPS"]
W2V_MODELS = ["cbow", "skipgram", "fast_text"]
