from pathlib import Path
def get_fasttext_config():
    BASE_PATH = str(Path(__file__).resolve().parents[4])

    config = {
        # Data paths
        "corpus_path": f"{BASE_PATH}/ours/outputs/raw_data/embedding/corpus_Advanced Micro Devices x86-64.pkl",
        "output_path": f"{BASE_PATH}/ours/outputs/models/embedding/fasttext/",

        # FastText parameters
        "vector_size": 256,
        "window": 5,
        "min_count": 3,
        "workers": 4,
        "sg": 1,  # 1 = Skip-gram variant
        "epochs": 5,
        "seed": 42,

        # Output filenames
        "model_filename": "fasttext_model.model",
        "vectors_filename": "fasttext_vectors.kv",
    }
    return config
