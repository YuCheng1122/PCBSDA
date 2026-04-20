def get_dann_config():
    BASE_PATH = "/home/tommy/Project/PCBSDA"
    EMBEDDING = "roberta_20"
    MODEL = "gat"

    config = {
        # Task mode
        "classification": True,

        # Domain: source = fully labeled, target = few-shot labeled
        "source_cpus": ["x86_64"],
        "target_cpus": ["Intel"],

        # Data paths
        "csv_path": f"{BASE_PATH}/datasets/csv/cross_architecture_dataset_family8.csv",
        "graph_dir": f"{BASE_PATH}/ours/outputs/embedded_graphs/{EMBEDDING}",

        # Cache
        "source_cache_file": f"{BASE_PATH}/ours/outputs/cache/dann/{EMBEDDING}/dann_source.pkl",
        "target_cache_file": f"{BASE_PATH}/ours/outputs/cache/dann/{EMBEDDING}/dann_target.pkl",
        "force_reload": False,

        # Data split
        "source_val_size": 0.2,
        "random_states": [42, 123, 7, 21, 99],  # 5 seeds

        # Few-shot (target train)
        "num_target_samples_per_class": 5,

        # Model architecture (GAT + attention pooling)
        "model_type": "GAT",
        "num_node_features": 256,
        "hidden_channels": 256,
        "output_channels": 256,
        "num_layers": 2,
        "dropout": 0.2,
        "pooling": "attention",
        "gat_heads": 4,

        # DANN Training
        # Loss = CE_src + lambda_domain * (domain_src + domain_tgt)
        "lambda_domain": 0.1,       # domain adversarial loss weight
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 200,
        "patience": 10,

        # Device
        "device": "cuda",

        # Workers
        "num_workers": 24,
        "pin_memory": True,

        # Output paths
        "model_output_dir": f"{BASE_PATH}/ours/outputs/models/dann/{EMBEDDING}/{MODEL}",
        "plot_dir":         f"{BASE_PATH}/ours/outputs/plots/dann/{EMBEDDING}/{MODEL}",
        "result_dir":       f"{BASE_PATH}/ours/outputs/results/dann/{EMBEDDING}/{MODEL}",
        "log_dir":          f"{BASE_PATH}/ours/outputs/logs/dann/{EMBEDDING}/{MODEL}",
    }

    return config
