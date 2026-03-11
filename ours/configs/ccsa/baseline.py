def get_ccsa_config():
    BASE_PATH = "/home/tommy/Project/PcodeBERT"
    TEST_CCSA_PATH = f"{BASE_PATH}/experiment/temp/test_ccsa"

    config = {
        # Task mode
        # "classification": False,  # False = label detection (binary), True = family classification
        "classification": True,   # 5-class family classification

        # Domain: source = fully labeled, target = few-shot labeled
        "source_cpus": ["x86_64"],
        "target_cpus": ["MIPS"],

        # Data paths
        # "csv_path": f"{BASE_PATH}/dataset/csv/merged_adjusted_filtered.csv",
        # "csv_path": f"{TEST_CCSA_PATH}/dataset/csv/family_balanced.csv",
        "csv_path": f"{TEST_CCSA_PATH}/dataset/csv/malware_only_multi_arch.csv",
        "graph_dir": f"{BASE_PATH}/experiment/temp/gnn/outputs/data/GNN/gpickle_cbow",

        # Cache
        # "source_cache_file": f"{TEST_CCSA_PATH}/outputs/cache/ccsa_source.pkl",
        # "target_cache_file": f"{TEST_CCSA_PATH}/outputs/cache/ccsa_target.pkl",
        # "source_cache_file": f"{TEST_CCSA_PATH}/outputs/cache/ccsa_family_source.pkl",
        # "target_cache_file": f"{TEST_CCSA_PATH}/outputs/cache/ccsa_family_target.pkl",
        "source_cache_file": f"{TEST_CCSA_PATH}/outputs/cache/ccsa_malware_source.pkl",
        "target_cache_file": f"{TEST_CCSA_PATH}/outputs/cache/ccsa_malware_target.pkl",
        "force_reload": True,   # 新 CSV 要 reload

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
        "model_output_dir": f"{TEST_CCSA_PATH}/outputs/models",
        "plot_dir": f"{TEST_CCSA_PATH}/outputs/plots",
        "result_dir": f"{TEST_CCSA_PATH}/outputs/results",
    }

    return config

