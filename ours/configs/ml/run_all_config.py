def get_ml_run_all_config():
    BASE_PATH = "/home/tommy/Project/PCBSDA"
    EMBEDDING = "cbow"

    config = {
        # Task mode
        "classification": True,

        # Architectures to evaluate (each runs independently as target)
        "target_cpus_list": [
            ["Intel"],
            ["MIPS"],
            ["x86_64"],
        ],

        # Data paths
        "csv_path": f"{BASE_PATH}/datasets/csv/cross_architecture_dataset_family8.csv",
        "graph_dir": f"{BASE_PATH}/ours/outputs/embedded_graphs/{EMBEDDING}",

        # Cache dir; per-target filenames are generated automatically
        "cache_dir": f"{BASE_PATH}/ours/outputs/cache/ml/{EMBEDDING}",
        "force_reload": False,

        # Few-shot: 每個 class 5 個 labeled samples 當 train，其餘當 test
        # 每個 seed 獨立抽樣，train/test split 皆不同
        "num_target_samples_per_class": 5,

        # 10 random seeds
        "random_states": [42, 123, 7, 21, 99, 314, 512, 1024, 2048, 4096],

        # Graph → vector pooling
        "pooling": "mean",

        # Only RF_200_sqrt
        "rf_configs": [
            {"n_estimators": 200, "max_depth": None, "min_samples_split": 2, "max_features": "sqrt", "label": "RF_200_sqrt"},
        ],
        "xgb_configs": [],
        "svm_configs": [],

        # Output paths (run_all.py appends /<target>/)
        "result_dir": f"{BASE_PATH}/ours/outputs/results/ml/{EMBEDDING}",
        "log_dir":    f"{BASE_PATH}/ours/outputs/logs/ml/{EMBEDDING}",
    }

    return config
