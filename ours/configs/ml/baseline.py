def get_ml_config():
    BASE_PATH = "/home/tommy/Project/PCBSDA"
    EMBEDDING = "cbow"

    config = {
        # Task mode
        "classification": True,

        # Target domain only (no source used)
        "target_cpus": ["ARM-32"],

        # Data paths
        "csv_path": f"{BASE_PATH}/datasets/csv/cross_architecture_dataset_family8.csv",
        "graph_dir": f"{BASE_PATH}/ours/outputs/embedded_graphs/{EMBEDDING}",

        # Cache
        "target_cache_file": f"{BASE_PATH}/ours/outputs/cache/ml/{EMBEDDING}/ml_target.pkl",
        "force_reload": False,

        # Few-shot: 每個 class 5 個 labeled samples 當 train，其餘當 test
        "num_target_samples_per_class": 5,

        # 10 個 random seed，控制每次抽到的 5 個樣本不一樣
        "random_states": [42, 123, 7, 21, 99, 314, 512, 1024, 2048, 4096],

        # Graph → vector: 對所有 node embedding 做 pooling
        "pooling": "mean",  # "mean" or "sum"

        # ---------- Model configs ----------

        # Random Forest
        "rf_configs": [
            {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "max_features": "sqrt",  "label": "RF_100_sqrt"},
            {"n_estimators": 200, "max_depth": None, "min_samples_split": 2, "max_features": "sqrt",  "label": "RF_200_sqrt"},
            {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "max_features": "log2",  "label": "RF_100_log2"},
        ],

        # XGBoost
        "xgb_configs": [
            {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1,  "subsample": 0.8, "label": "XGB_d6"},
            {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1,  "subsample": 0.8, "label": "XGB_d3"},
            {"n_estimators":  50, "max_depth": 3, "learning_rate": 0.3,  "subsample": 0.8, "label": "XGB_d3_lr03"},
        ],

        # SVM
        "svm_configs": [
            {"kernel": "rbf",    "C": 1.0,  "gamma": "scale", "label": "SVM_rbf_C1"},
            {"kernel": "rbf",    "C": 1.0,  "gamma": 0.01,    "label": "SVM_rbf_g001"},
            {"kernel": "rbf",    "C": 0.1,  "gamma": 0.01,    "label": "SVM_rbf_C01_g001"},
            {"kernel": "linear", "C": 0.1,  "gamma": "scale", "label": "SVM_linear_C01"},
            {"kernel": "linear", "C": 1.0,  "gamma": "scale", "label": "SVM_linear_C1"},
            {"kernel": "linear", "C": 10.0, "gamma": "scale", "label": "SVM_linear_C10"},
        ],

        # Output paths
        "result_dir": f"{BASE_PATH}/ours/outputs/results/ml/{EMBEDDING}",
        "log_dir":    f"{BASE_PATH}/ours/outputs/logs/ml/{EMBEDDING}",
    }

    return config
