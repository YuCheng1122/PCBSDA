import json
import numpy as np
from os import name
import os
import pickle
import re
import torch
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import List, Dict, Tuple, Generator, Optional, Union
from datasets import Dataset, IterableDataset
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR


#Regex pattern preprocessing
#1)  opcode_pattern: Extract P-Code
#2)  opcode_pattern: Extract Calculation
OPCODE_PAT = re.compile(r"(?:\)\s+|---\s+)([A-Z_]+)")
OPERAND_PAT = re.compile(r"\(([^ ,]+)\s*,\s*[^,]*,\s*([0-9]+)\)")

def read_filenames_from_csv(csv_file_path: str | Path, cpu_filter: Optional[str] = None) -> List[str]:
    try:
        df = pd.read_csv(csv_file_path)
        if cpu_filter:
            print(f"Filtering files for CPU: {cpu_filter}")
            df_filtered = df[df['CPU'] == cpu_filter]
            print(f"Found {len(df_filtered)} files matching the filter.")
            return df_filtered['file_name'].tolist()
    
        return df['file_name'].tolist()
        
    except (FileNotFoundError, KeyError) as e:
        print(f"Error reading CSV: {e}")
        return []


def iterate_json_files(csv_file_path: Path, root_dir: Path, error_log_path: Path, cpu_filter: Optional[str] = None) -> Generator[Tuple[str, Dict], None, None]:
    file_names = read_filenames_from_csv(csv_file_path, cpu_filter=cpu_filter)
    for file_name in file_names:
        json_path = root_dir / file_name / f"{file_name}.json"
        if not json_path.exists():
            with open(error_log_path, "a", encoding="utf-8") as f_err:
                f_err.write(f"{file_name}\n")
            continue  
        try:
            with json_path.open(encoding="utf-8") as fp:
                yield file_name, json.load(fp)
        except json.JSONDecodeError:
            with open(error_log_path, "a", encoding="utf-8") as f_err:
                f_err.write(f"{file_name}\n")
            continue 



def _map_operand(op_type: str) -> str:
    op_type_l = op_type.lower()
    if op_type_l == 'register':
        return "REG"
    if op_type_l == 'ram':
        return "MEM"
    if op_type_l in {'const', 'constant'}:
        return "CONST"
    if op_type_l == 'unique':
        return "UNIQUE"
    if op_type_l == 'stack':
        return "STACK"
    return "UNK"


def _append_to_pickle(file_path: Path, new_data):
    """將新資料追加到現有的 pickle 檔案中"""
    if file_path.exists():
        with open(file_path, "rb") as f:
            existing_data = pickle.load(f)
        existing_data.extend(new_data)
    else:
        existing_data = new_data
    
    with open(file_path, "wb") as f:
        pickle.dump(existing_data, f)
        

def create_instruction_sentence(instruction_dict: Dict) -> Optional[List[str]]:
    operation_str = instruction_dict.get("operation", "")
    if not operation_str:
        return None
    
    command_match = OPCODE_PAT.search(operation_str)
    if not command_match:
        return None

    command = command_match.group(1)
    sentence = [command]
    
    operands = OPERAND_PAT.findall(operation_str)
    for op_type, _ in operands:
        sentence.append(_map_operand(op_type))
    
    return sentence

def extract_sentences_from_file(file_name_data: Tuple[str, Dict]) -> List[List[str]]:
    file_name, pcode_dict = file_name_data
    sentences = []
    try:
        for func_data in pcode_dict.values():
            if not isinstance(func_data, dict): continue
            for instruction in func_data.get("instructions", []):
                sentence = create_instruction_sentence(instruction)
                if sentence:
                    sentences.append(sentence)
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
    return sentences

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


def get_device():
    """Get the best available device (GPU if available, otherwise CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def create_transformer_scheduler(optimizer, num_training_steps, scheduler_type="linear", warmup_ratio=0.1):
    """
    Create a learning rate scheduler
    
    Args:
        optimizer: The optimizer to schedule
        num_training_steps: Total number of training steps
        scheduler_type: Type of scheduler ("linear", "cosine")
        warmup_ratio: Ratio of warmup steps to total steps
    
    Returns:
        Learning rate scheduler
    """
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    
    if scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        print(f"Created linear scheduler with {num_warmup_steps} warmup steps")
    elif scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        print(f"Created cosine scheduler with {num_warmup_steps} warmup steps")
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    return scheduler


def setup_training_environment():
    """Setup training environment with GPU support"""
    device = get_device()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    return device



def plot_training_curves(train_losses, val_losses, test_losses, val_accuracies, seed, save_dir="outputs/plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train / Validation / Test Loss')
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'loss_curves_{seed}.png')
    plt.savefig(save_path)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(val_accuracies, label='Val Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'val_accuracy_curve_{seed}.png')
    plt.savefig(save_path)
    plt.close()


def save_experiment_results(results_dict, save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    if 'all_results' in results_dict:
        results_df = pd.DataFrame(results_dict['all_results'])
        results_df.to_csv(os.path.join(save_dir, f'results_{timestamp}.csv'), index=False)
    
    summary_data = {k: v for k, v in results_dict.items() if k != 'all_results'}
    with open(os.path.join(save_dir, f'summary_{timestamp}.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    return timestamp


def plot_confusion_matrix(cm, labels, filename="confusion_matrix.png"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def test_model(model, test_loader, device, label_encoder):
    """測試模型並生成詳細結果"""
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            prob = torch.softmax(out, dim=1)
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_prob.extend(prob.cpu().numpy())
    
    accuracy = accuracy_score(y_true, y_pred)
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    # Calculate precision and recall
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    y_prob_array = np.array(y_prob)
    if len(label_encoder.classes_) == 2:
        auc_score = roc_auc_score(y_true, y_prob_array[:, 1])
    else:
        auc_score = roc_auc_score(y_true, y_prob_array, multi_class='ovr', average='macro')
    
    # 分類報告
    report = classification_report(y_true, y_pred, target_names=[str(c) for c in label_encoder.classes_])
    
    # 混淆矩陣
    original_labels = label_encoder.inverse_transform(sorted(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=label_encoder.transform(original_labels))
    
    # print("LabelEncoder classes:", label_encoder.classes_)
    # print("y_true labels:", set(label_encoder.inverse_transform(y_true)))
    # print("y_pred labels:", set(label_encoder.inverse_transform(y_pred)))
    # print("y_pred counts:", dict(pd.Series(label_encoder.inverse_transform(y_pred)).value_counts()))
    print("Report:\n", report)


    return {
        'accuracy': accuracy,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'auc': auc_score,
        'precision': precision,
        'recall': recall,
        'classification_report': report,
        'confusion_matrix': cm,
        'original_labels': original_labels,
        'y_true': y_true,
        'y_pred': y_pred
    }


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    
    return total_loss / len(train_loader.dataset)


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            pred = out.argmax(dim=1)
            correct += int((pred == batch.y).sum())
            total_loss += loss.item() * batch.num_graphs
    
    accuracy = correct / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader.dataset)
    return accuracy, avg_loss


def load_test_data_by_arch(csv_path, graph_dir, target_cpus, label_encoder, classification=False):
    df = pd.read_csv(csv_path)
    test_graphs_by_arch = {}
    
    for cpu in target_cpus:
        cpu_df = df[df['CPU'] == cpu]
        print(f"載入 {cpu} 測試資料: {len(cpu_df)} 個樣本")
        
        graphs, labels = load_graphs_from_df(cpu_df, graph_dir, classification)
        
        encoded_labels = label_encoder.transform(labels)
        for i, data in enumerate(graphs):
            data.y = torch.tensor(encoded_labels[i], dtype=torch.long)
        
        test_graphs_by_arch[cpu] = graphs
    
    return test_graphs_by_arch


def load_cross_arch_data(csv_path, graph_dir, source_cpus, target_cpus, cache_file, 
                        val_size=0.2, random_state=42, force_reload=False, classification=False):
    """
    載入跨架構的圖資料
    
    Args:
        csv_path: CSV 檔案路徑 (包含 file_name, CPU, label, family 欄位)
        graph_dir: 圖資料目錄
        source_cpus: 訓練用的 CPU 架構列表
        target_cpus: 測試用的 CPU 架構列表 (如果為空列表,表示單架構)
        cache_file: 快取檔案路徑
        val_size: 驗證集比例
        random_state: 隨機種子
        force_reload: 是否強制重新載入
        classification: True 使用 'family' 欄位, False 使用 'label' 欄位
    
    Returns:
        tuple: (train_graphs, val_graphs, test_graphs, label_encoder, num_classes)
    """
    
    if force_reload and os.path.exists(cache_file):
        os.remove(cache_file)
    
    # 檢查快取
    if os.path.exists(cache_file):
        print(f"載入快取資料: {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        return (cached_data['train_graphs'], 
               cached_data['val_graphs'],
               cached_data['test_graphs'], 
               cached_data['label_encoder'], 
               cached_data['num_classes'])
    
    print("載入 CSV 資料...")
    df = pd.read_csv(csv_path)
    
    # 判斷是否為單架構 (target_cpus 為空)
    if not target_cpus:
        # 單架構模式: 只使用 source_cpus 的資料
        all_df = df[df['CPU'].isin(source_cpus)]
        print(f"單架構模式: {len(all_df)} 個樣本 (架構: {source_cpus})")
        
        # 載入圖資料
        all_graphs, all_labels = load_graphs_from_df(all_df, graph_dir, classification)
        
        # 先分出測試集 (20%)
        train_val_graphs, test_graphs, train_val_labels, test_labels = train_test_split(
            all_graphs, all_labels, test_size=0.2, 
            stratify=all_labels, random_state=random_state
        )
        
        # 再從剩餘資料中分出驗證集
        train_graphs, val_graphs, train_labels, val_labels = train_test_split(
            train_val_graphs, train_val_labels, test_size=val_size/(1-0.2), 
            stratify=train_val_labels, random_state=random_state
        )
        
    else:
        # 跨架構模式: 分離訓練和測試資料
        train_df = df[df['CPU'].isin(source_cpus)]
        test_df = df[df['CPU'].isin(target_cpus)]
        
        print(f"跨架構模式:")
        print(f"訓練資料: {len(train_df)} 個樣本 (架構: {source_cpus})")
        print(f"測試資料: {len(test_df)} 個樣本 (架構: {target_cpus})")
        
        # 載入圖資料
        train_graphs, train_labels = load_graphs_from_df(train_df, graph_dir, classification)
        test_graphs, test_labels = load_graphs_from_df(test_df, graph_dir, classification)
        
        # 分割訓練和驗證資料
        from sklearn.model_selection import train_test_split
        
        train_graphs, val_graphs, train_labels, val_labels = train_test_split(
            train_graphs, train_labels, test_size=val_size, 
            stratify=train_labels, random_state=random_state
        )
    
    # 標籤編碼
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels + val_labels + test_labels)
    
    encoded_train_labels = label_encoder.transform(train_labels)
    encoded_val_labels = label_encoder.transform(val_labels)
    encoded_test_labels = label_encoder.transform(test_labels)
    
    num_classes = len(label_encoder.classes_)
    
    # 更新圖標籤
    for i, data in enumerate(train_graphs):
        data.y = torch.tensor(encoded_train_labels[i], dtype=torch.long)
        
    for i, data in enumerate(val_graphs):
        data.y = torch.tensor(encoded_val_labels[i], dtype=torch.long)
        
    for i, data in enumerate(test_graphs):
        data.y = torch.tensor(encoded_test_labels[i], dtype=torch.long)

    # 快取資料
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    cache_data = {
        'train_graphs': train_graphs,
        'val_graphs': val_graphs,
        'test_graphs': test_graphs,
        'label_encoder': label_encoder,
        'num_classes': num_classes
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"資料已快取到: {cache_file}")
    return train_graphs, val_graphs, test_graphs, label_encoder, num_classes


def load_graphs_from_df(df, graph_dir, classification=False):
    """
    從 DataFrame 載入圖資料
    
    Args:
        df: DataFrame 包含 file_name, label, family 等欄位
        graph_dir: 圖資料目錄
        classification: True 使用 'family' 欄位, False 使用 'label' 欄位
    """
    import pickle
    from torch_geometric.data import Data
    
    graphs = []
    labels = []
    
    for _, row in df.iterrows():
        file_name = row['file_name']
        prefix = file_name[:2]
        # 根據 classification 參數選擇使用 label 或 family
        label = row['family'] if classification else row['label']
        graph_path = Path(graph_dir) / prefix / f"{file_name}.gpickle"

        if not graph_path.exists():
            continue
            
        # 載入 gpickle 檔案
        with open(graph_path, 'rb') as f:
            data = pickle.load(f)
        
        node_embeddings = data['node_embeddings']
        if not node_embeddings:
            continue
        
        # 轉換為張量
        embeddings = [list(emb) for emb in node_embeddings.values()]
        x = torch.tensor(embeddings, dtype=torch.float)
        
        # 創建序列邊
        num_nodes = len(embeddings)
        edge_list = []
        for i in range(num_nodes - 1):
            edge_list.extend([[i, i+1], [i+1, i]])
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        else:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        graph_data = Data(x=x, edge_index=edge_index)
        graphs.append(graph_data)
        labels.append(label)
    
    return graphs, labels


def simple_early_stopping(current_val_acc, best_val_acc, patience_counter, patience):
    """簡單的早停機制"""
    if current_val_acc > best_val_acc:
        best_val_acc = current_val_acc
        patience_counter = 0
        should_stop = False
    else:
        patience_counter += 1
        should_stop = patience_counter >= patience
    
    return best_val_acc, patience_counter, should_stop


def create_gnn_scheduler(optimizer, scheduler_type, **kwargs):
    """
    Create learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ("step", "plateau", "cosine")
        **kwargs: Additional parameters for scheduler
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "step":
        step_size = kwargs.get("step_size", 30)
        gamma = kwargs.get("gamma", 0.5)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type == "plateau":
        patience = kwargs.get("patience", 10)
        factor = kwargs.get("factor", 0.5)
        min_lr = kwargs.get("min_lr", 1e-6)
        return ReduceLROnPlateau(optimizer, mode='min', patience=patience, 
                               factor=factor, min_lr=min_lr)
    
    elif scheduler_type == "cosine":
        T_max = kwargs.get("T_max", 100)
        eta_min = kwargs.get("eta_min", 1e-6)
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")