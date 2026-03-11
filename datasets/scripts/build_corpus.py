import pickle
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


from utils import iterate_json_files, extract_sentences_from_file_BERT, read_filenames_from_csv


CSV_PATH = Path("/home/tommy/Projects/PcodeBERT/dataset/csv/merged_adjusted_filtered.csv")
RAW_DATA_PATH = Path("/home/tommy/Projects/PcodeBERT/reverse/new/results")
TEMP_DIR = Path("/home/tommy/Projects/PcodeBERT/outputs/preprocessed/temp_files")
FINAL_CORPUS_PATH = Path("/home/tommy/Projects/PcodeBERT/outputs/preprocessed/pcode_corpus_x86_64_new_data.pkl")
ERROR_LOG_PATH = Path("/home/tommy/Projects/PcodeBERT/outputs/preprocessed/error_log.txt")
TARGET_CPU = "x86_64"


def process_and_save_worker(args):
    file_data, output_path = args
    sentences = extract_sentences_from_file_BERT(file_data)
    if sentences:
        with open(output_path, "wb") as f:
            pickle.dump(sentences, f)
    return output_path

def build_batches(csv_path: Path, root_dir: Path, temp_output_dir: Path, error_log_path: Path, cpu_to_process: str):
    if temp_output_dir.exists():
        shutil.rmtree(temp_output_dir)
    temp_output_dir.mkdir(parents=True)

    if error_log_path.exists():
        error_log_path.unlink()

    print("--- Stage 1: Building Batches ---")
    file_iterator = iterate_json_files(csv_path, root_dir, error_log_path, cpu_filter=cpu_to_process)
    
    # 先取得第一個檔案並顯示範例
    first_file = next(file_iterator)
    example_sequences = extract_sentences_from_file_BERT(first_file)
    
    print("\n=== 資料格式範例 ===")
    if example_sequences:
        print(f"總資料筆數: {len(example_sequences)}")
        print("\n前三筆資料範例:")
        for i, seq in enumerate(example_sequences[:3]):
            print(f"\n第 {i+1} 筆:")
            print(seq)
    
    print("\n=== Starting Batch Processing ===")
    
    def task_generator():
        for i, file_data in enumerate(file_iterator):
            output_path = temp_output_dir / f"file_{i}.pkl"
            yield (file_data, output_path)

    file_names_for_count = read_filenames_from_csv(csv_path, cpu_filter=cpu_to_process)
    total_files = len(file_names_for_count)
    del file_names_for_count

    if total_files == 0:
        print("No files found for the specified CPU. Exiting.")
        return

    print(f"Found {total_files} files to process. Using {cpu_count()} CPU cores.")

    with Pool(cpu_count()) as pool:
        results_iterator = pool.imap_unordered(process_and_save_worker, task_generator())
        
        for _ in tqdm(results_iterator, total=total_files, desc="Processing files"):
            pass 

    print("\nBatch processing complete.")

def merge_batches(temp_dir: Path, final_output_path: Path):
    print(f"\n--- Stage 2: Merging Batches ---")
    if final_output_path.exists():
        final_output_path.unlink()
    temp_files = sorted(temp_dir.glob("file_*.pkl"))
    if not temp_files:
        print("No temporary files found to merge.")
        return

    print(f"Found {len(temp_files)} temporary files to merge.")
    
    total_sentences = 0
    with open(final_output_path, "wb") as f_final:
        for temp_file in tqdm(temp_files, desc="Merging files"):
            with open(temp_file, "rb") as f_in:
                sentences_from_file = pickle.load(f_in)
                pickle.dump(sentences_from_file, f_final)
                total_sentences += len(sentences_from_file)

    print(f"\nTotal sentences merged: {total_sentences}")
    print("Merge complete!")

if __name__ == "__main__":
    build_batches(CSV_PATH, RAW_DATA_PATH, TEMP_DIR, ERROR_LOG_PATH, cpu_to_process=TARGET_CPU)
    merge_batches(TEMP_DIR, FINAL_CORPUS_PATH)

    print(f"\nCleaning up temporary directory: {TEMP_DIR}")
    shutil.rmtree(TEMP_DIR)
    
    print("\nCorpus building process finished successfully!")
    print(f"Final corpus file is located at: {FINAL_CORPUS_PATH.resolve()}")