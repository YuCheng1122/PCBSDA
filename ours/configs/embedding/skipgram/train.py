def get_skipgram_config():
    BASE_PATH = "/home/tommy/Project/PCBSDA"

    config = {
        # Data paths
        "corpus_path": f"{BASE_PATH}/datasets/preprocessed/pcode_corpus_x86_64_new_data.pkl",
        "output_path": f"{BASE_PATH}/ours/outputs/models/embedding/skipgram/",

        # Word2Vec parameters
        "vector_size": 256,
        "window": 5,
        "min_count": 3,
        "workers": 4,
        "sg": 1,  # 1 = Skip-gram
        "epochs": 5,
        "seed": 42,

        # Output filenames
        "model_filename": "skipgram_model.model",
        "vectors_filename": "skipgram_vectors.kv",
    }
    return config
