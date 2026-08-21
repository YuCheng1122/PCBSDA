from pathlib import Path


def get_sensitivity_config():
    BASE_PATH = Path(__file__).resolve().parents[3]
    EMBEDDING = "roberta"
    MODEL = "gat"

    config = {
        # Pairs to sweep — each gets its own output subdir
        "experiment_pairs": [
            (["x86_64"], ["MIPS"]),
            (["ARM-32"], ["MIPS"]),
            (["Intel"],  ["MIPS"]),
            (["x86_64"], ["ARM-32"]),
            (["x86_64"], ["Intel"]),
            (["ARM-32"], ["x86_64"]),
            (["ARM-32"], ["Intel"]),
            (["Intel"],  ["x86_64"]),
            (["Intel"],  ["ARM-32"]),
        ],

        # kept for backward compat (used as fallback in run_setting)
        "source_cpus": ["x86_64"],
        "target_cpus": ["MIPS"],

        "random_states": [42, 123, 7],

        # Task mode
        "classification": True,

        # Data paths
        "csv_path": f"{BASE_PATH}/datasets/csv/cross_architecture_dataset_family8.csv",
        "graph_dir": f"{BASE_PATH}/ours/outputs/embedded_graphs/{EMBEDDING}",

        # Cache dir
        "cache_dir": f"{BASE_PATH}/ours/outputs/cache/ccsa/{EMBEDDING}",
        "force_reload": False,

        # Data split
        "source_val_size": 0.2,

        # Few-shot
        "num_target_samples_per_class": 5,

        # Pair generation
        "neg_pair_ratio": 3,

        # Model architecture (GAT + attention pooling)
        "model_type": "GAT",
        "num_node_features": 256,
        "hidden_channels": 256,
        "output_channels": 256,
        "num_layers": 2,
        "dropout": 0.2,
        "pooling": "attention",
        "gat_heads": 4,

        # Base CCSA hyperparams (used as defaults when sweeping the other)
        "alpha": 0.3,
        "csa_margin": 0.5,
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 200,
        "patience": 10,

        # Device
        "device": "cuda",

        # Workers
        "num_workers": 24,
        "pin_memory": True,

        # Sensitivity sweep values
        "alpha_sweep": [0.1, 0.2, 0.3, 0.4, 0.5, 0.7],
        "margin_sweep": [0.1, 0.3, 0.5, 0.7, 1.0],

        # Output dirs for sensitivity analysis
        "model_output_dir": f"{BASE_PATH}/ours/outputs/models/ccsa/{EMBEDDING}/{MODEL}/sensitivity",
        "plot_dir":         f"{BASE_PATH}/ours/outputs/plots/ccsa/{EMBEDDING}/{MODEL}/sensitivity",
        "result_dir":       f"{BASE_PATH}/ours/outputs/results/ccsa/{EMBEDDING}/{MODEL}/sensitivity",
        "log_dir":          f"{BASE_PATH}/ours/outputs/logs/ccsa/{EMBEDDING}/{MODEL}/sensitivity",
    }

    return config
