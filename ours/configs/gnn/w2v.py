def get_w2v_gnn_config(model_name):
    """
    GNN config for w2v-based embeddings (cbow, skipgram, fasttext).
    model_name: 'cbow', 'skipgram', or 'fasttext'
    """
    BASE_PATH = "/home/tommy/Project/PCBSDA"

    config = {
        # Task mode
        "classification": False,  # False = label detection, True = family classification
        "source_cpus": ["x86_64"],
        "target_cpus": ["ARM", "PPC", "MIPS", "Intel"],

        # Data paths
        "csv_path": f"{BASE_PATH}/datasets/csv/merged_adjusted_filtered.csv",
        "graph_dir": f"{BASE_PATH}/ours/outputs/embedded_graphs/{model_name}",
        "cache_file": f"{BASE_PATH}/ours/outputs/cache/gnn_data_{model_name}.pkl",
        "test_cache_file": f"{BASE_PATH}/ours/outputs/cache/gnn_data_{model_name}_test.pkl",

        # Data split
        "single_arch_val_size": 0.2,
        "single_arch_test_size": 0.1,
        "cross_arch_val_size": 0.2,
        "random_state": 42,
        "force_reload": False,

        # Model architecture
        "model_type": "GCN",
        "num_node_features": 256,   # matches w2v vector_size
        "hidden_channels": 128,
        "output_channels": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "pooling": "add",

        # GAT specific
        "gat_heads": 4,
        "gat_dropout": 0.2,

        # Training
        "num_workers": 4,
        "pin_memory": True,
        "batch_size": 32,
        "learning_rate": 0.01,
        "epochs": 200,
        "patience": 20,

        # Scheduler
        "scheduler_type": "plateau",
        "step_size": 30,
        "gamma": 0.5,
        "plateau_patience": 10,
        "plateau_factor": 0.5,
        "cosine_T_max": 100,

        # Seed (single run)
        "seed": 42,
        "device": "cuda",

        # Output paths (per embedding model)
        "model_output_dir": f"{BASE_PATH}/ours/outputs/models/gnn/{model_name}",
        "plot_dir": f"{BASE_PATH}/ours/outputs/plots/gnn/{model_name}",
        "result_dir": f"{BASE_PATH}/ours/outputs/results/gnn/{model_name}",
    }

    return config
