def get_gnn_config():
    BASE_PATH = "/home/tommy/Project/PCBSDA"
    EMBEDDING = "roberta_20"  # "cbow", "skipgram", "fast_text", "roberta"

    config = {
        # Task mode
        "classification": True,  # False = label detection, True = family classification
        "source_cpus": ["x86_64"],  # 4-class classification: ARM, MIPS, Intel, PPC; binary classification: ARM vs non-ARM (MIPS/Intel/PPC)
        "target_cpus": ["MIPS", "Intel", "ARM-32"],  # 4-class classification: ARM, MIPS, Intel, PPC; binary classification: ARM vs non-ARM (MIPS/Intel/PPC)
        # "MIPS", "Intel", "PPC"

        # Data paths
        "csv_path": f"{BASE_PATH}/datasets/csv/cross_architecture_dataset_family8_x86.csv",
        "graph_dir": f"{BASE_PATH}/ours/outputs/embedded_graphs/{EMBEDDING}",
        "cache_file": f"{BASE_PATH}/ours/outputs/cache/gnn/{EMBEDDING}/gnn.pkl",
        "test_cache_file": f"{BASE_PATH}/ours/outputs/cache/gnn/{EMBEDDING}/gnn_test.pkl",

        # Data split (單架構: train/val/test = 7/2/1, 跨架構: train/val = 8/2)
        "single_arch_val_size": 0.2,
        "single_arch_test_size": 0.1,
        "cross_arch_val_size": 0.2,
        "random_state": 42,
        "force_reload": True,

        # Model architecture
        "model_type": "GAT",            # "GCN" or "GAT"
        "num_node_features": 256,       # input node feature dimension
        "hidden_channels": 256,          # hidden layer dimension
        "output_channels": 256,         # output dimension before classifier
        "num_layers": 2,                # number of GNN conv layers
        "dropout": 0.2,
        "pooling": "attention",               # "add" or "attention"

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
        "scheduler_type": "step",       # "step", "plateau", "cosine"
        "step_size": 30,                # for StepLR
        "gamma": 0.5,                   # for StepLR
        "plateau_patience": 10,         # for ReduceLROnPlateau
        "plateau_factor": 0.5,          # for ReduceLROnPlateau
        "cosine_T_max": 100,            # for CosineAnnealingLR

        # Seeds
        "seeds": [42],
        "device": "cuda",

        # Output paths
        "model_output_dir": f"{BASE_PATH}/ours/outputs/models/gnn/{EMBEDDING}",
        "plot_dir": f"{BASE_PATH}/ours/outputs/plots/gnn/{EMBEDDING}",
        "result_dir": f"{BASE_PATH}/ours/outputs/results/gnn/{EMBEDDING}",
        "log_dir": f"{BASE_PATH}/ours/outputs/logs/gnn/{EMBEDDING}",
    }

    return config
