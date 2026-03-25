def get_gnn_cross_config():
    BASE_PATH = "/home/tommy/Project/PCBSDA"
    EMBEDDING = "roberta_20"  # "cbow", "skipgram", "fast_text", "roberta"
    MODEL = "gat"             # "gcn" or "gat"

    config = {
        # Task mode
        "classification": True,

        # Source architectures and their targets
        # MIPS 只作為 target，不作為 source
        "arch_experiments": {
            "x86_64": {"targets": ["ARM-32", "Intel", "MIPS"]},
            "ARM-32": {"targets": ["Intel",  "MIPS",  "x86_64"]},
            "Intel":  {"targets": ["ARM-32", "MIPS",  "x86_64"]},
        },

        # Data paths (all experiments share the same CSV)
        "csv_path":  f"{BASE_PATH}/datasets/csv/cross_architecture_dataset_family8_x86.csv",
        "graph_dir": f"{BASE_PATH}/ours/outputs/embedded_graphs/{EMBEDDING}",

        # Data split
        "cross_arch_val_size": 0.2,
        "random_state": 42,
        "force_reload": True,

        # Model architecture
        "model_type": "GAT",            # "GCN" or "GAT"
        "num_node_features": 256,
        "hidden_channels": 256,
        "output_channels": 256,
        "num_layers": 2,
        "dropout": 0.2,
        "pooling": "attention",

        # GAT specific
        "gat_heads": 4,
        "gat_dropout": 0.2,

        # Training
        "num_workers": 4,
        "pin_memory": True,
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 200,
        "patience": 15,

        # Scheduler
        "scheduler_type": "step",
        "step_size": 30,
        "gamma": 0.5,
        "plateau_patience": 10,
        "plateau_factor": 0.5,
        "cosine_T_max": 100,

        # Seeds
        "seeds": [42],
        "device": "cuda",

        # Output paths
        "model_output_dir": f"{BASE_PATH}/ours/outputs/models/gnn/{EMBEDDING}/{MODEL}",
        "plot_dir":         f"{BASE_PATH}/ours/outputs/plots/gnn/{EMBEDDING}/{MODEL}",
        "result_dir":       f"{BASE_PATH}/ours/outputs/results/gnn_cross/{EMBEDDING}/{MODEL}",
        "log_dir":          f"{BASE_PATH}/ours/outputs/logs/gnn/{EMBEDDING}/{MODEL}",
    }

    return config
