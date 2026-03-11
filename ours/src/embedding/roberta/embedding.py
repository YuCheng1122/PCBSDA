import os
import sys
import pickle
import torch
import numpy as np
from transformers import RobertaForMaskedLM, AutoTokenizer
import glob
from tqdm import tqdm
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def load_pretrained_model():
    """載入預訓練的模型和tokenizer"""
    model_path = "/home/tommy/Project/PcodeBERT/outputs/models/RoBERTa/model_epoch_25"
    
    print(f"Loading model from: {model_path}")
    
    # 載入tokenizer和model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = RobertaForMaskedLM.from_pretrained(model_path)
    
    # 設定device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model Config: {model.config}")
    print(f"Model loaded successfully on device: {device}")
    return model, tokenizer, device

def get_embeddings_batch(sentences, model, tokenizer, device, batch_size=1000):
    all_embeddings = []
    
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i : i + batch_size]
        
        inputs = tokenizer(
            batch_sentences, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.roberta(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            print(f"\nEmbedding stats:")
            print(f"  Mean: {embeddings.mean().item():.6f}")
            print(f"  Std: {embeddings.std().item():.6f}")
            print(f"  Min: {embeddings.min().item():.6f}")
            print(f"  Max: {embeddings.max().item():.6f}")     
            all_embeddings.append(embeddings.cpu().numpy())
    
    if not all_embeddings:
        return np.array([])
        
    return np.concatenate(all_embeddings, axis=0)

def process_single_graph(graph_path, model, tokenizer, device):
    """處理單個graph檔案（使用批次處理）"""
    try:
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
        
        node_ids_with_sentence = []
        sentences_to_process = []
        for node_id, node_data in graph.nodes(data=True):
            sentence = node_data.get('sentence', '')
            if sentence:
                node_ids_with_sentence.append(node_id)
                sentences_to_process.append(sentence)

        if not sentences_to_process:
            return {
                'file_path': graph_path,
                'node_embeddings': {},
                'node_sentences': {},
                'num_nodes': 0,
                'embedding_dim': model.config.hidden_size 
            }
        all_embeddings = get_embeddings_batch(sentences_to_process, model, tokenizer, device)
        
        node_embeddings = {}
        node_sentences = {}
        embedding_dim = 0
        
        if all_embeddings.size > 0:
            embedding_dim = all_embeddings.shape[1]
            for i, node_id in enumerate(node_ids_with_sentence):
                node_embeddings[node_id] = all_embeddings[i]
                node_sentences[node_id] = sentences_to_process[i]
        
        return {
            'file_path': graph_path,
            'node_embeddings': node_embeddings,
            'node_sentences': node_sentences,
            'num_nodes': len(node_embeddings),
            'embedding_dim': embedding_dim
        }
        
    except Exception as e:
        print(f"Error processing {graph_path}: {e}")
        return None

def find_all_gpickle_files(base_path):
    gpickle_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.gpickle'):
                gpickle_files.append(os.path.join(root, file))
    
    return gpickle_files

def batch_process_graphs(base_path, output_base_dir):
    """批量處理所有graph檔案，保持相同目錄結構"""
    # 載入模型
    model, tokenizer, device = load_pretrained_model()
    
    # 找到所有gpickle檔案
    gpickle_files = find_all_gpickle_files(base_path)
    print(f"Found {len(gpickle_files)} gpickle files")
    
    # 統計資訊
    processed_count = 0
    failed_count = 0
    total_nodes = 0
    embedding_dim = 0
    
    for i, file_path in enumerate(tqdm(gpickle_files, desc="Processing graphs")):
        print(f"\nProcessing {i+1}/{len(gpickle_files)}: {os.path.basename(file_path)}")
        
        result = process_single_graph(file_path, model, tokenizer, device)
        
        if result:
            # 建立對應的輸出路徑結構
            # 從原始路徑取得相對路徑
            rel_path = os.path.relpath(file_path, base_path)
            output_path = os.path.join(output_base_dir, rel_path)
            
            # 創建輸出目錄
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 儲存單個檔案的embedding結果
            with open(output_path, 'wb') as f:
                pickle.dump(result, f)
            
            processed_count += 1
            total_nodes += result['num_nodes']
            if embedding_dim == 0:
                embedding_dim = result['embedding_dim']
        else:
            failed_count += 1
    
    # 儲存統計資訊
    stats = {
        'total_files': len(gpickle_files),
        'processed_files': processed_count,
        'failed_files': failed_count,
        'total_nodes': total_nodes,
        'embedding_dim': embedding_dim
    }
    
    stats_path = os.path.join(output_base_dir, "processing_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Results saved to: {output_base_dir}")
    print(f"Stats saved to: {stats_path}")

def main():
    # 設定路徑
    base_path = "/home/tommy/Project/PcodeBERT/outputs/data/GNN/gpickle_merged_adjusted_filtered"
    output_dir = "/home/tommy/Project/PcodeBERT/outputs/data/GNN/gpickle_merged_adjusted_filtered_temp"
    
    print(f"Starting batch processing...")
    print(f"Input directory: {base_path}")
    print(f"Output directory: {output_dir}")
    
    # 批量處理
    batch_process_graphs(base_path, output_dir)

if __name__ == "__main__":
    main() 
