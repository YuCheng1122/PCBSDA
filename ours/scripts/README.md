# Scripts 使用說明

所有指令在專案根目錄 (`PCBSDA/`) 下執行。

## 1. 訓練 Word Embedding 模型

```bash
bash ours/scripts/train_word_embeddings.sh all        # 全部
bash ours/scripts/train_word_embeddings.sh cbow       # 單跑 CBOW
bash ours/scripts/train_word_embeddings.sh skipgram   # 單跑 Skip-gram
bash ours/scripts/train_word_embeddings.sh fasttext   # 單跑 FastText
```

模型存放：`ours/outputs/models/embedding/{cbow,skipgram,fasttext}/`

## 2. 完整 Pipeline（一鍵跑完）

```bash
bash ours/scripts/run_w2v_experiments.sh all           # 全部 model
bash ours/scripts/run_w2v_experiments.sh cbow          # 只跑 cbow
```

Pipeline 三個步驟：
1. 訓練 word embedding 模型
2. 用訓練好的模型對 graph nodes 做 embedding
3. 用 embedded graphs 訓練 GNN

## 3. 分步驟手動執行

```bash
# Step 1: 訓練 embedding
bash ours/scripts/train_word_embeddings.sh all

# Step 2: 對 graph nodes 做 embedding
python -m ours.src.embedding.batch_embedding_w2v --model all
python -m ours.src.embedding.batch_embedding_w2v --model cbow  # 單跑

# Step 3: 訓練 GNN
python -m ours.src.gnn.w2v_training --model cbow
python -m ours.src.gnn.w2v_training --model skipgram
python -m ours.src.gnn.w2v_training --model fasttext
```

## 輸出路徑

| 產物 | 路徑 |
|------|------|
| Embedding 模型 | `ours/outputs/models/embedding/{method}/` |
| Embedded graphs | `ours/outputs/embedded_graphs/{method}/` |
| GNN 模型 | `ours/outputs/models/gnn/{method}/` |
| GNN 結果 | `ours/outputs/results/gnn/{method}/` |
| 訓練曲線圖 | `ours/outputs/plots/gnn/{method}/` |
| Log | `ours/outputs/logs/w2v_pipeline_{timestamp}.log` |

## Config 檔位置

| Config | 路徑 |
|--------|------|
| CBOW | `ours/configs/embedding/cbow/train.py` |
| Skip-gram | `ours/configs/embedding/skipgram/train.py` |
| FastText | `ours/configs/embedding/fast_text/train.py` |
| GNN (w2v) | `ours/configs/gnn/w2v.py` |
