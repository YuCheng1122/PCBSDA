import pickle
from gensim.models import FastText
from pathlib import Path
from typing import Union
from datasets import Dataset

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

corpus_path = '/home/tommy/Project/PcodeBERT/outputs/preprocessed/pcode_corpus_x86_64_new_data.pkl'
output_path = '/home/tommy/Project/PcodeBERT/outputs/models/fasttext/'

dataset = load_corpus_dataset(corpus_path)

sentences = []
for item in dataset:
    tokens = item['text'].split()
    sentences.append(tokens)

model = FastText(
    sentences=sentences,
    vector_size=256,
    window=5,
    min_count=3,
    workers=4,
    sg=1,
    epochs=5,
    seed=42
)

model.save(output_path + 'fasttext_model.model')
model.wv.save(output_path + 'fasttext_vectors.kv')

print(f"FastText model trained and saved to {output_path}")
print(f"Vocabulary size: {len(model.wv)}")
