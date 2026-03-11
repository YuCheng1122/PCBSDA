def get_gnn_config():
    BASE_PATH = "/home/tommy/Project/PcodeBERT"
    TEST_GNN_PATH = f"{BASE_PATH}/experiment/temp/gnn"

    config = {
        # Task mode
        "classification": True,  # False = label detection, True = family classification
        "source_cpus": ["x86_64"],  # 4-class classification: ARM, MIPS, Intel, PPC; binary classification: ARM vs non-ARM (MIPS/Intel/PPC)
        "target_cpus": ["MIPS"],  # 4-class classification: ARM, MIPS, Intel, PPC; binary classification: ARM vs non-ARM (MIPS/Intel/PPC)
        # "MIPS", "Intel", "PPC"

        # Data paths
        "csv_path": "/home/tommy/Project/PcodeBERT/experiment/temp/test_ccsa/dataset/csv/malware_only_multi_arch.csv",
        "graph_dir": f"{TEST_GNN_PATH}/outputs/data/GNN/gpickle_roberta",  # directory containing gpickle files with Skip-gram embeddings
        "cache_file": f"{TEST_GNN_PATH}/outputs/cache/gnn_roberta.pkl",
        "test_cache_file": f"{TEST_GNN_PATH}/outputs/cache/gnn_roberta_test.pkl",

        # Data split (單架構: train/val/test = 7/2/1, 跨架構: train/val = 8/2)
        "single_arch_val_size": 0.2,
        "single_arch_test_size": 0.1,
        "cross_arch_val_size": 0.2,
        "random_state": 42,
        "force_reload": False,

        # Model architecture
        "model_type": "GCN",            # "GCN" or "GAT"
        "num_node_features": 128,       # input node feature dimension
        "hidden_channels": 128,          # hidden layer dimension
        "output_channels": 128,         # output dimension before classifier
        "num_layers": 2,                # number of GNN conv layers
        "dropout": 0.2,
        "pooling": "add",               # "add" or "attention"

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
        "seeds": [42, 2021],
        "device": "cuda",

        # Output paths
        "model_output_dir": f"{TEST_GNN_PATH}/outputs/models/gnn",
        "plot_dir": f"{TEST_GNN_PATH}/outputs/plots",
        "result_dir": f"{TEST_GNN_PATH}/outputs/results",
    }

    return config
