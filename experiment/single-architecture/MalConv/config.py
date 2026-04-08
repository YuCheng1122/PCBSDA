BASE_PATH = "/home/tommy/Projects/PCBSDA"

ALL_ARCHS = ["Intel", "ARM-32", "x86_64", "MIPS"]


def get_malconv_single_config(arch="x86_64"):
    """
    Single-architecture family classification config for MalConv (raw bytes).
    arch: one of ["Intel", "ARM-32", "x86_64", "MIPS"]
    """
    arch_tag = arch.replace("-", "_").lower()

    config = {
        # Task
        "classification": True,
        "source_cpus": [arch],

        # Data paths
        "csv_path": f"{BASE_PATH}/datasets/csv/single_arch_dataset.csv",
        "raw_byte_dir": f"{BASE_PATH}/experiment/outputs/malConv/raw_byte/results_raw_byte",

        # Sequence truncation: same as MalConv paper (2MB).
        # Memory is managed by chunk-wise conv in model.forward(), not by truncation.
        "max_len": 1_048_576,

        # Cross-Validation
        "n_splits": 5,
        "random_state": 42,

        # Architecture fixed per paper (filter_size=500, stride=500, 128 filters, embed_dim=8)
        # Paper used batch=256 on 8xGPU DGX-1; single RTX 3080 → batch 16-32
        # Paper: more filters / larger embed → overfitting, so keep these fixed
        "embed_dim": 8,
        "num_filters": 128,
        "filter_size": 500,
        "stride": 500,

        # Optuna — only tune the parts not fixed by the paper
        "n_trials": 10,
        "optuna_timeout": None,

        "search_space": {
            "dropout": [0.0, 0.5],
            "batch_size": [16, 32],
            "learning_rate": [1e-3, 1e-1],   # paper used SGD lr=0.01; we use Adam
            "scheduler_type": ["step", "plateau", "cosine"],
        },

        # Training (fixed)
        "num_workers": 4,
        "pin_memory": True,
        "epochs": 100,
        "patience": 10,

        # Scheduler fixed params
        "step_size": 20,
        "gamma": 0.5,
        "plateau_patience": 5,
        "plateau_factor": 0.5,
        "cosine_T_max": 50,

        "device": "cuda",

        # Output paths
        "optuna_dir": f"{BASE_PATH}/experiment/outputs/malConv/optuna/{arch_tag}",
        "result_dir": f"{BASE_PATH}/experiment/outputs/malConv/results/{arch_tag}",
        "log_dir": f"{BASE_PATH}/experiment/outputs/malConv/logs/{arch_tag}",
    }

    return config
