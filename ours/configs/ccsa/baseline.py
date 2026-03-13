def get_ccsa_config():
    BASE_PATH = "/home/tommy/Project/PCBSDA"
    EMBEDDING = "cbow"  # "cbow", "skipgram", "fast_text", "roberta"

    config = {
        # Task mode
        "classification": True,

        # Domain: source = fully labeled, target = few-shot labeled
        "source_cpus": ["x86_64"],
        "target_cpus": ["ARM-32"],

        # Data paths
        "csv_path": f"{BASE_PATH}/datasets/csv/cross_architecture_dataset_clean.csv",
        "graph_dir": f"{BASE_PATH}/ours/outputs/embedded_graphs/{EMBEDDING}",

        # Cache
        "source_cache_file": f"{BASE_PATH}/ours/outputs/cache/ccsa/{EMBEDDING}/ccsa_source.pkl",
        "target_cache_file": f"{BASE_PATH}/ours/outputs/cache/ccsa/{EMBEDDING}/ccsa_target.pkl",
        "force_reload": True,

        # Data split
        "source_val_size": 0.2,         # source: train/val = 80/20
        "random_states": [42, 123],  # 多次實驗用不同 target sampling

        # Few-shot: target domain 使用的 labeled 樣本數量
        # "num_target_samples": 10,      # 舊: 總共 10 個 (binary)
        "num_target_samples_per_class": 5,  # 每個 class 5 個 (4-class × 5 = 20 total)

        # Pair generation
        "neg_pair_ratio": 3,            # negative pairs = neg_pair_ratio * positive pairs

        # Model architecture (GCN baseline)
        "num_node_features": 128,
        "hidden_channels": 128,
        "output_channels": 128,         # feature dimension (for CSA loss)
        "num_layers": 2,
        "dropout": 0.2,
        "pooling": "add",               # "add" or "mean"

        # CCSA Training (best from sweep: alpha=0.9, margin=0.5)
        "alpha": 0.9,                  # loss = (1-alpha)*CE + alpha*CSA
        "csa_margin": 0.5,             # CSA cosine margin
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
        "model_output_dir": f"{BASE_PATH}/ours/outputs/models/ccsa/{EMBEDDING}",
        "plot_dir": f"{BASE_PATH}/ours/outputs/plots/ccsa/{EMBEDDING}",
        "result_dir": f"{BASE_PATH}/ours/outputs/results/ccsa/{EMBEDDING}",
        "log_dir": f"{BASE_PATH}/ours/outputs/logs/ccsa/{EMBEDDING}",
    }

    return config

