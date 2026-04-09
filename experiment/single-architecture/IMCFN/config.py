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
        # Each sample is stored as a .npy file of raw uint8 bytes (shape: (N,))
        "raw_byte_dir": f"{BASE_PATH}/experiment/outputs/raw_data/single_architecture/MalConv/results_raw_byte",

        # Image settings (paper: 224×224, jet colormap)
        "image_size": 224,

        # Cross-Validation
        "n_splits": 5,
        "random_state": 42,
        "test_size": 0.2,
        "optuna_n_splits": 3,

        # Optuna — only tune dropout and learning rate; architecture is fixed by paper
        "n_trials": 10,
        "optuna_timeout": None,

        "search_space": {
            "learning_rate": [1e-5, 1e-3],   # log-uniform; paper uses 5e-6
            "dropout":       [0.3, 0.7],
            "batch_size":    [16, 32],
            "weight_decay":  [1e-5, 1e-3],   # log-uniform
        },

        # Training (fixed)
        "scheduler_type": "cosine",
        "num_workers": 4,
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
