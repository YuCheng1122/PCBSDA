def get_dsne_config():
    BASE_PATH = "/home/tommy/Project/PCBSDA"
    EMBEDDING = "cbow"
    MODEL = "gcn"

    config = {
        # Task mode
        "classification": True,

        # Domain: source = fully labeled, target = few-shot labeled
        "source_cpus": ["x86_64"],
        "target_cpus": ["ARM-32"],

        # Data paths
        "csv_path": f"{BASE_PATH}/datasets/csv/cross_architecture_dataset_family8.csv",
        "graph_dir": f"{BASE_PATH}/ours/outputs/embedded_graphs/{EMBEDDING}",

        # Cache
        "source_cache_file": f"{BASE_PATH}/ours/outputs/cache/dsne/{EMBEDDING}/dsne_source.pkl",
        "target_cache_file": f"{BASE_PATH}/ours/outputs/cache/dsne/{EMBEDDING}/dsne_target.pkl",
        "force_reload": False,

        # Data split
        "source_val_size": 0.2,
        "random_states": [42, 123],

        # Few-shot
        "num_target_samples_per_class": 5,

        # Model architecture (GCN baseline)
        "model_type": "GCN",
        "num_node_features": 256,
        "hidden_channels": 256,
        "output_channels": 256,
        "num_layers": 2,
        "dropout": 0.2,
        "pooling": "add",

        # d-SNE Training (paper Eq. 7: loss = d-SNE + alpha*CE_src + beta*CE_tgt)
        "alpha": 0.5,               # CE_source weight
        "beta": 0.5,                # CE_target weight
        "dsne_margin": 1.0,         # margin for modified-Hausdorff loss
        "bidirectional": True,      # 雙向訓練（官方設計）
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
        "model_output_dir": f"{BASE_PATH}/ours/outputs/models/dsne/{EMBEDDING}/{MODEL}",
        "plot_dir":         f"{BASE_PATH}/ours/outputs/plots/dsne/{EMBEDDING}/{MODEL}",
        "result_dir":       f"{BASE_PATH}/ours/outputs/results/dsne/{EMBEDDING}/{MODEL}",
        "log_dir":          f"{BASE_PATH}/ours/outputs/logs/dsne/{EMBEDDING}/{MODEL}",
    }

    return config
