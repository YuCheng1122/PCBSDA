def get_gemal_single_config(arch: str = "x86_64") -> dict:
    """
    Single-architecture family classification config for GEMAL.
    arch: one of ["Intel", "ARM-32", "x86_64", "MIPS"]

    Model: GCN + attention readout, graph embedding → 300-d → classifier
    """
    BASE_PATH = "/home/tommy/Project/PCBSDA"

    config = {
        # Task
        "source_cpus": [arch],

        # Data paths — reuse FCGAT CBOW embedded graphs (node dim = 100)
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
        "optuna_epochs":   70,
        "search_space": {
            "learning_rate": [1e-4, 5e-3],
            "weight_decay":  [1e-5, 1e-2],
            "batch_size":    [16, 32],
        },

        # Model — GCN + attention readout
        "hidden_channels": 64,
        "embed_dim":       300,
        "dropout":         0.5,

        # Training
        "epochs":        70,
        "patience":      10,
        "batch_size":    32,
        "learning_rate": 0.001,
        "weight_decay":  0.0,

        # Scheduler
        "scheduler_type":   "plateau",
        "plateau_patience": 5,
        "plateau_factor":   0.5,

        "device": "cuda",

        # DataLoader
        "num_workers": 4,
        "pin_memory":  True,

        # Output paths
        "optuna_dir": f"{BASE_PATH}/experiment/outputs/optuna/gemal/{arch}",
        "result_dir": f"{BASE_PATH}/experiment/outputs/results/gemal/{arch}",
        "log_dir":    f"{BASE_PATH}/experiment/outputs/logs/gemal/{arch}",
    }

    return config


ALL_ARCHS = ["Intel", "ARM-32", "x86_64", "MIPS"]
