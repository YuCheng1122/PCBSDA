def get_cbow_config():
    BASE_PATH = "/home/tommy/Project/PCBSDA"

    config = {
        # Data paths
        "corpus_path": f"{BASE_PATH}/ours/outputs/raw_data/embedding/corpus_Advanced Micro Devices x86-64.pkl",
        "output_path": f"{BASE_PATH}/ours/outputs/models/embedding/cbow/",

        # Word2Vec parameters
        "vector_size": 256,
        "window": 5,
        "min_count": 3,
        "workers": 48,
        "sg": 0,  
        "epochs": 5,
        "seed": 42,

        # Output filenames
        "model_filename": "cbow_model.model",
        "vectors_filename": "cbow_vectors.kv",
    }
    return config
