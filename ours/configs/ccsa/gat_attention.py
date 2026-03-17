def get_ccsa_config():
    BASE_PATH = "/home/tommy/Project/PCBSDA"
    EMBEDDING = "cbow"
    MODEL = "gat"

    config = {
        # Task mode
        "classification": True,

        # Domain: source = fully labeled, target = few-shot labeled
        "source_cpus": ["x86_64"],
        "target_cpus": ["ARM-32"],

        # Data paths
        "csv_path": f"{BASE_PATH}/datasets/csv/cross_architecture_dataset_family8.csv",
        "graph_dir": f"{BASE_PATH}/ours/outputs/embedded_graphs/{EMBEDDING}",

        # Cache (reuse same cache as baseline)
        "source_cache_file": f"{BASE_PATH}/ours/outputs/cache/ccsa/{EMBEDDING}/ccsa_source.pkl",
        "target_cache_file": f"{BASE_PATH}/ours/outputs/cache/ccsa/{EMBEDDING}/ccsa_target.pkl",
        "force_reload": False,

        # Data split
        "source_val_size": 0.2,
        "random_states": [42, 123, 7, 21, 99],  # 5 seeds

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
        "pooling": "attention",         # "add", "mean", or "attention"
        "gat_heads": 4,

        # CCSA Training
        "alpha": 0.9,
        "csa_margin": 0.5,
        "batch_size": 128,
        "learning_rate": 0.001,
        "epochs": 200,
        "patience": 10,

        # Device
        "device": "cuda",

        # Workers
        "num_workers": 24,
        "pin_memory": True,

        # Output paths
        "model_output_dir": f"{BASE_PATH}/ours/outputs/models/ccsa/{EMBEDDING}/{MODEL}",
        "plot_dir":         f"{BASE_PATH}/ours/outputs/plots/ccsa/{EMBEDDING}/{MODEL}",
        "result_dir":       f"{BASE_PATH}/ours/outputs/results/ccsa/{EMBEDDING}/{MODEL}",
        "log_dir":          f"{BASE_PATH}/ours/outputs/logs/ccsa/{EMBEDDING}/{MODEL}",
    }

    return config
