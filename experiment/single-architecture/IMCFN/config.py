BASE_PATH = "/home/tommy/Projects/PCBSDA"

ALL_ARCHS = ["Intel", "ARM-32", "x86_64", "MIPS"]


def get_imcfn_single_config(arch="x86_64"):
    """
    Single-architecture family classification config for IMCFN
    (binary → color image → fine-tuned VGG16).
    arch: one of ["Intel", "ARM-32", "x86_64", "MIPS"]
    """
    arch_tag = arch.replace("-", "_").lower()

    config = {
        # Task
        "classification": True,
        "source_cpus": [arch],

        # Data paths
        "csv_path": f"{BASE_PATH}/datasets/csv/single_arch_dataset.csv",
        # Each sample is a 224×224 RGB PNG already converted from binary
        "image_dir": f"{BASE_PATH}/experiment/outputs/raw_data/single_architecture/IMCFN/results_image",

        # Cross-Validation
        "n_splits": 5,
        "random_state": 42,
        "test_size": 0.2,
        "optuna_n_splits": 5,

        # Optuna
        "n_trials": 20,
        "optuna_timeout": None,

        "batch_size": 32,  # default fallback; overridden by search_space

        "search_space": {
            "learning_rate": [1e-5, 1e-3],   # log-uniform; paper uses 5e-6
            "dropout":       [0.3, 0.7],
            "weight_decay":  [1e-5, 1e-3],   # log-uniform
            "batch_size":    [16, 32],   # categorical
        },

        # Training (fixed)
        "scheduler_type": "cosine",
        "num_workers": 12,
        "pin_memory": True,
        "epochs": 50,
        "patience": 10,
        "cosine_T_max": 25,

        "device": "cuda",

        # Output paths — aligned to experiment/outputs/{type}/single_architecture/IMCFN/
        "optuna_dir": f"{BASE_PATH}/experiment/outputs/optuna/single_architecture/IMCFN/{arch_tag}",
        "result_dir": f"{BASE_PATH}/experiment/outputs/results/single_architecture/IMCFN/{arch_tag}",
        "log_dir":    f"{BASE_PATH}/experiment/outputs/logs/single_architecture/IMCFN/{arch_tag}",
    }

    return config
