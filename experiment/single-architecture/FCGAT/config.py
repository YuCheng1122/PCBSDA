def get_fcgat_single_config(arch="x86_64"):
    """
    Single-architecture family classification config for FCGAT.
    arch: one of ["Intel", "ARM-32", "x86_64", "MIPS"]

    Word2Vec (CBOW) settings follow the paper:
      vector_size=100, window=2, epochs=100

    GAT + Set2Set architecture follows Table I:
      GATConv(100→192, K=3), Set2Set(192→384, steps=4), FC(384→N)
    """
    BASE_PATH = "/home/tommy/Project/PCBSDA"

    config = {
        # Task
        "source_cpus": [arch],

        # Data paths
        "csv_path":  f"{BASE_PATH}/datasets/csv/single_arch_dataset.csv",
        "graph_dir": f"{BASE_PATH}/experiment/outputs/embedded_graphs/single-architecture/FCGAT",

        # Cross-validation
        "n_splits":     5,
        "random_state": 42,
        "test_size":    0.2,

        # Optuna
        "n_trials":        20,
        "optuna_timeout":  None,
        "optuna_n_splits": 5,
        "optuna_epochs":   150,
        "search_space": {
            "learning_rate": [1e-4, 5e-3],
            "weight_decay":  [1e-5, 1e-2],
            "batch_size":    [32, 64],
        },

        # Model (Table I)
        "gat_heads":       3,      # K=3 multi-head
        "hidden_channels": 192,    # GAT output dim (64 per head × 3)
        "set2set_steps":   4,      # Set2Set LSTM steps
        "dropout":         0.5,    # paper: Dropout 0.5

        # Training
        "epochs":       700,       # paper: epochs 700
        "patience":     30,
        "batch_size":   32,
        "learning_rate": 0.001,    # paper: lr 0.001 (AdamW)
        "weight_decay":  1e-2,

        # Scheduler
        "scheduler_type":   "step",
        "step_size":        30,
        "gamma":            0.5,
        "plateau_patience": 10,
        "plateau_factor":   0.5,
        "cosine_T_max":     100,

        "device": "cuda",

        # DataLoader
        "num_workers": 4,
        "pin_memory":  True,

        # Output paths
        "optuna_dir": f"{BASE_PATH}/experiment/outputs/optuna/fcgat/{arch}",
        "result_dir": f"{BASE_PATH}/experiment/outputs/results/fcgat/{arch}",
        "log_dir":    f"{BASE_PATH}/experiment/outputs/logs/fcgat/{arch}",
    }

    return config


ALL_ARCHS = ["Intel", "ARM-32", "x86_64", "MIPS"]
