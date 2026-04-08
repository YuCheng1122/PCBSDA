# Single-Architecture Malware Family Classification — Agent Guide

## 任務描述

單架構惡意軟體家族分類（Single-Architecture Family Classification）。
對同一 CPU 架構的資料做 train/val/test 切分，進行 GNN-based 多分類任務。

## 實驗流程

1. **超參數搜尋（Optuna）**：在 train+val 資料上做 cross-validation，搜尋最佳超參數組合
2. **Cross-Validation 評估**：用 Optuna 找到的最佳超參數，做 K-Fold cross-validation 得到最終報告指標
3. **儲存結果**：metrics（accuracy、precision、recall、f1_micro、f1_macro、AUC）與 best hyperparameters

## 目錄結構

```
single-architecture/
├── AGENT.md              ← 本說明文件
├── Word2Vec/
│   ├── config.py         ← Word2Vec 版本的 config（含 CV / Optuna 參數）
│   └── run.py            ← Word2Vec 版本的主要執行腳本
└── RoBERTa/
    ├── config.py         ← RoBERTa 版本的 config（含 CV / Optuna 參數）
    └── run.py            ← RoBERTa 版本的主要執行腳本
```

## 執行方式

```bash
# Word2Vec embedding，跑所有架構
cd /home/tommy/Project/PCBSDA
python experiment/single-architecture/Word2Vec/run.py

# 指定單一架構
python experiment/single-architecture/Word2Vec/run.py --arch x86_64

# RoBERTa embedding
python experiment/single-architecture/RoBERTa/run.py --arch x86_64

# 只跑 Optuna 搜尋（不做最終 CV 評估）
python experiment/single-architecture/Word2Vec/run.py --arch x86_64 --tune-only

# 只跑最終 CV（使用已存的 best params）
python experiment/single-architecture/Word2Vec/run.py --arch x86_64 --eval-only
```

## 關鍵設計

### Cross-Validation
- 使用 `StratifiedKFold`（預設 `n_splits=5`），確保每個 fold class 分布一致
- 不設固定 random seed loop，改為 CV fold index 作為多次評估的基礎
- 最終指標回報每個 fold 的 mean ± std

### Optuna 超參數搜尋
- 搜尋空間：`learning_rate`、`hidden_channels`、`num_layers`、`dropout`、`batch_size`、`pooling`、`scheduler_type`
- objective：最大化 validation F1-macro（在 inner CV 上）
- 預設 `n_trials=50`，可透過 `--n-trials` 調整
- Study 結果存到 `outputs/optuna/{embedding}/{arch}/study.pkl`

### Embedding 差異
| | Word2Vec | RoBERTa |
|---|---|---|
| embedding dir | `outputs/embedded_graphs/cbow` 或 `skipgram` | `outputs/embedded_graphs/roberta_20` |
| node feature dim | 256 | 256（roberta_20） |
| model type | GCN or GAT | GAT |

## 程式碼依賴

共用 code 從 `ours/` 引入，不重複實作：
- `ours/src/gnn/models.py` → `GCN`, `GAT`
- `ours/src/gnn/utils.py` → `load_graphs_from_df`, `train_epoch`, `evaluate`, `test_model`, `save_experiment_results`
- `ours/configs/gnn/gnn_single.py` → `get_gnn_single_config` 作為 default 參數參考

## 輸出路徑

```
ours/outputs/
├── optuna/{embedding}/{arch}/
│   ├── study.pkl           ← Optuna study 物件
│   └── best_params.json    ← 最佳超參數
└── results/single_arch_cv/{embedding}/{arch}/
    ├── results_{timestamp}.csv
    └── summary_{timestamp}.json
```
