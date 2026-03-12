# Scripts 使用說明

所有指令需先 `pip install -e .`（在 `ours/` 目錄下），之後可在任意目錄執行。

## 1. 訓練 Word Embedding 模型

```bash
# 個別執行（可在任意目錄）
python ours/src/embedding/cbow/main.py
python ours/src/embedding/skipgram/main.py
python ours/src/embedding/fast_text/main.py

# 或用 shell script（在 PCBSDA/ 下執行）
bash ours/scripts/train_word_embeddings.sh all
bash ours/scripts/train_word_embeddings.sh cbow
```

模型存放：`ours/outputs/models/embedding/{cbow,skipgram,fasttext}/`

## 2. 對 Graph Nodes 做 Embedding

```bash
# W2V 系列（cbow, skipgram, fasttext）
python ours/src/embedding/batch_embedding_w2v.py --model all
python ours/src/embedding/batch_embedding_w2v.py --model cbow

# RoBERTa
python ours/src/embedding/roberta/embedding.py
```

## 3. RoBERTa Pretrain

```bash
python ours/src/embedding/roberta/main.py
```

流程：load corpus → 動態建 vocab/tokenizer → pretrain（MLM）→ 存 model

## 4. 完整 W2V Pipeline（一鍵跑完）

```bash
bash ours/scripts/run_w2v_experiments.sh all
bash ours/scripts/run_w2v_experiments.sh cbow
```

Pipeline 三個步驟：
1. 訓練 word embedding 模型
2. 用訓練好的模型對 graph nodes 做 embedding
3. 用 embedded graphs 訓練 GNN

## 輸出路徑

| 產物 | 路徑 |
|------|------|
| Embedding 模型 | `ours/outputs/models/embedding/{method}/` |
| Embedded graphs | `ours/outputs/embedded_graphs/{method}/` |
| RoBERTa model | `ours/outputs/models/embedding/roberta/` |
| RoBERTa tokenizer | `ours/outputs/tokenizer/roberta/` |
| RoBERTa checkpoints | `ours/outputs/checkpoints/embedding/roberta/` |
| GNN 模型 | `ours/outputs/models/gnn/{method}/` |
| GNN 結果 | `ours/outputs/results/gnn/{method}/` |
| Log | `ours/outputs/logs/` |

## Config 檔位置

| Config | 路徑 |
|--------|------|
| CBOW | `ours/configs/embedding/cbow/train.py` |
| Skip-gram | `ours/configs/embedding/skipgram/train.py` |
| FastText | `ours/configs/embedding/fast_text/train.py` |
| RoBERTa pretrain | `ours/configs/embedding/roberta/pretrain.py` |
| GNN (w2v) | `ours/configs/gnn/w2v.py` |
