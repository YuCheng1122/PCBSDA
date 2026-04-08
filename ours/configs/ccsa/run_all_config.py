def get_run_all_config():
    BASE_PATH = "/home/tommy/Project/PCBSDA"
    EMBEDDING = "roberta_20"
    MODEL = "gat"

    config = {
        # Experiment pairs and seeds
        # MIPS is test-only, never used as source
        "experiment_pairs": [
            # (["x86_64"], ["ARM-32"]),
            # (["x86_64"], ["MIPS"]),
            (["x86_64"], ["Intel"]),
            (["ARM-32"], ["x86_64"]),
            (["ARM-32"], ["MIPS"]),
            (["ARM-32"], ["Intel"]),
            (["Intel"],  ["x86_64"]),
            (["Intel"],  ["MIPS"]),
            (["Intel"],  ["ARM-32"]),
        ],
        "random_states": [42, 123, 7],

        # Task mode
        "classification": True,

        # Data paths
        "csv_path": f"{BASE_PATH}/datasets/csv/cross_architecture_dataset_family8.csv",
        "graph_dir": f"{BASE_PATH}/ours/outputs/embedded_graphs/{EMBEDDING}",

        # Cache dir; per-pair filenames are generated automatically in run_all.py
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

        # CCSA Training
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

        # Output base dirs (run_all.py appends /<pair_tag>/ per pair)
        "model_output_dir": f"{BASE_PATH}/ours/outputs/models/ccsa/{EMBEDDING}/{MODEL}",
        "plot_dir":         f"{BASE_PATH}/ours/outputs/plots/ccsa/{EMBEDDING}/{MODEL}",
        "result_dir":       f"{BASE_PATH}/ours/outputs/results/ccsa/{EMBEDDING}/{MODEL}",
        "log_dir":          f"{BASE_PATH}/ours/outputs/logs/ccsa/{EMBEDDING}/{MODEL}",
    }

    return config
