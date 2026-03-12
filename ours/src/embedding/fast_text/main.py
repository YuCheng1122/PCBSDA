import pickle
from gensim.models import FastText
from pathlib import Path
from typing import Union
from datasets import Dataset

from configs.embedding.fast_text.train import get_fasttext_config

def load_corpus_dataset(corpus_path: Union[str, Path]) -> Dataset:
    corpus_path = Path(corpus_path)
    processed_path = corpus_path.parent / f"{corpus_path.stem}_processed"
    if processed_path.exists():
        print(f"Loading processed dataset from cache: {processed_path}")
        from datasets import load_from_disk
        dataset = load_from_disk(str(processed_path))
        print(f"Loaded processed dataset: {len(dataset)} samples (memory-mapped)")
        return dataset

    print(f"Processing dataset from: {corpus_path}")
    print("Using generator to avoid loading all data into RAM at once...")

    def data_generator():
        with open(corpus_path, 'rb') as f:
            while True:
                try:
                    corpus_batch = pickle.load(f)
                    if isinstance(corpus_batch, list):
                        for sentence_tokens in corpus_batch:
                            if isinstance(sentence_tokens, list):
                                yield {"text": " ".join(sentence_tokens)}
                except EOFError:
                    break
                except Exception as e:
                    print(f"Error reading batch: {e}")
                    break

    dataset = Dataset.from_generator(data_generator)

    print(f"Saving processed dataset to cache: {processed_path}")
    dataset.save_to_disk(str(processed_path))
    print(f"Dataset cached: {len(dataset)} samples (memory-mapped format)")

    return dataset

if __name__ == "__main__":
    cfg = get_fasttext_config()

    dataset = load_corpus_dataset(cfg["corpus_path"])

    sentences = []
    for item in dataset:
        tokens = item['text'].split()
        sentences.append(tokens)

    model = FastText(
        sentences=sentences,
        vector_size=cfg["vector_size"],
        window=cfg["window"],
        min_count=cfg["min_count"],
        workers=cfg["workers"],
        sg=cfg["sg"],
        epochs=cfg["epochs"],
        seed=cfg["seed"]
    )

    output_path = cfg["output_path"]
    model.save(output_path + cfg["model_filename"])
    model.wv.save(output_path + cfg["vectors_filename"])

    print(f"FastText model trained and saved to {output_path}")
    print(f"Vocabulary size: {len(model.wv)}")
