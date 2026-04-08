# Single-Architecture Malware Family Classification — Agent Guide

## 任務描述

單架構惡意軟體家族分類（Single-Architecture Family Classification）。
對同一 CPU 架構的資料做 dev/test 切分，進行 GNN-based 多分類任務。

## 實驗流程

1. **Dev / Test Split**：全部資料 stratified 切成 80% dev、20% held-out test（固定 seed=42）
2. **Optuna 超參數搜尋**：在 dev 上做 3-fold inner CV，搜尋最佳超參數（20 trials）
3. **Final CV 評估**：用 best params 在 dev 上做 5-fold CV，報告 mean ± std
4. **Test 評估**：用 dev 90% train、10% early-stop，最終在 held-out test 評估一次
5. **儲存結果**：metrics（accuracy、precision、recall、f1_micro、f1_macro、AUC）與 best params

## 目錄結構

```
single-architecture/
├── AGENT.md
├── Word2Vec/
│   ├── config.py         ← Word2Vec 版本的 config
│   └── run.py            ← Word2Vec 版本的主要執行腳本
├── RoBERTa/
│   ├── config.py         ← RoBERTa 版本的 config
│   └── run.py            ← RoBERTa 版本的主要執行腳本
└── ../scripts/single-architecture/
    └── run_single_arch.sh  ← 依序執行 RoBERTa → Word2Vec，跑完寄 email
```

## 執行方式

```bash
# 單次執行（從 PCBSDA 根目錄）
python experiment/single-architecture/RoBERTa/run.py --arch x86_64
python experiment/single-architecture/Word2Vec/run.py --arch x86_64 --w2v-model cbow

# 跑所有架構
python experiment/single-architecture/RoBERTa/run.py
python experiment/single-architecture/Word2Vec/run.py

# 只跑 Optuna（不做 Final CV）
python experiment/single-architecture/RoBERTa/run.py --arch x86_64 --tune-only

# 跳過 Optuna，載入已存的 best_params 直接做 Final CV + Test
python experiment/single-architecture/RoBERTa/run.py --arch x86_64 --eval-only

# 一鍵執行全部實驗 + 寄信通知
bash experiment/scripts/single-architecture/run_single_arch.sh
```

## 關鍵設計

### Data Split
- 全部資料先用 `train_test_split(test_size=0.2, stratify=..., random_state=42)` 切出 held-out test set
- Optuna 和 Final CV 只在 dev set（80%）上進行，test set 完全不碰直到最後

### Optuna 超參數搜尋
- 搜尋空間（3 個參數）：`learning_rate`（log-uniform）、`num_layers`（1~3）、`dropout`（0.1~0.5）
- 固定不搜：`hidden_channels=256`、`batch_size=32`、`output_channels=256`
- Inner CV：3-fold `StratifiedKFold`，objective 最大化 F1-macro
- `n_trials=20`，`TPESampler(seed=42)` 固定取樣順序
- 跑完自動印出 parameter importances
- Study 儲存為 `study.pkl`，可事後用 `optuna.importance.get_param_importances(study)` 分析

### Cross-Validation
- Final CV：5-fold `StratifiedKFold` on dev，回報 mean ± std
- 每個 fold seed = `random_state + fold_idx`（固定且可重現）

### Seed 固定策略
| 項目 | seed |
|------|------|
| Dev/Test split | `random_state=42` |
| Optuna TPESampler | `seed=42` |
| StratifiedKFold | `random_state=42` |
| 每個 fold 的 train | `42 + fold_idx` |

### Embedding 差異
| | Word2Vec | RoBERTa |
|---|---|---|
| embedding dir | `ours/outputs/embedded_graphs/{cbow,skipgram,fast_text}` | `ours/outputs/embedded_graphs/roberta_20` |
| node feature dim | 256 | 256 |
| model type | GAT | GAT |

## 程式碼依賴

- `ours/src/gnn/models.py` → `GCN`, `GAT`
- `ours/src/gnn/utils.py` → `load_graphs_from_df`, `train_epoch`, `evaluate`, `test_model`, `save_experiment_results`

## 輸出路徑

```
experiment/outputs/
├── results/
│   ├── roberta/{roberta_tag}/{arch}/    ← RoBERTa 結果
│   └── word2vec/{w2v_model}/{arch}/     ← Word2Vec 結果
├── optuna/
│   ├── roberta/{roberta_tag}/{arch}/
│   │   ├── study.pkl
│   │   └── best_params.json
│   └── word2vec/{w2v_model}/{arch}/
└── logs/
    ├── roberta/{roberta_tag}/{arch}/
    ├── word2vec/{w2v_model}/{arch}/
    ├── roberta_run.log    ← run_single_arch.sh 的完整 stdout
    └── w2v_run.log
```
