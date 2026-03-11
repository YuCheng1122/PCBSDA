# PCBSDA — Pcode-Based Cross-Architecture Malware Detection with Domain Adaptation

## 研究主題
跨架構（cross-architecture）惡意軟體檢測。
Source 與 target domain 為不同 CPU 架構，目標是讓模型從 source 遷移到 target 仍具備檢測能力。

## Pipeline（我們的方法）
1. **Ghidra 反編譯** → 取得 Pcode，建構 Function Call Graph（FCG）
2. **資料格式**：NetworkX DiGraph gpickle，每個 node 屬性含 `function_name` 與 `tokens`（Pcode opcode 序列）
3. **NLP Embedding**：將 node 的 opcode tokens 編碼為 node feature vector
4. **GNN**：對 FCG 做 graph-level representation learning → graph embedding
5. **Domain Adaptation**：graph embedding 層級做 supervised domain adaptation
6. **分類**：Graph-level classification（binary: malware vs benign / multi-class: malware family）

## 目錄結構
```
PCBSDA/
├── ours/                # 我們的方法
│   ├── src/
│   │   ├── embedding/   # NLP embedding（roberta, cbow, skipgram, fast_text）
│   │   ├── gnn/         # GNN backbone（models.py 含所有 GNN 架構）
│   │   └── transfer_learning/  # DA 方法，backbone 從 gnn/ import
│   ├── configs/         # 超參數設定（按方法分子目錄）
│   ├── outputs/         # 實驗結果、checkpoint、log
│   └── notebooks/       # 分析 / 視覺化
├── baselines/           # 別人的論文實作（每篇獨立子目錄）
└── datasets/            # 共用資料集與前處理腳本
    ├── csv/
    └── scripts/         # 資料前處理（build_corpus 等）
```

## 規則
- 不要刪除或覆蓋原有的 code，保留原始實作做對照
- 新方法用新檔案或新增在既有檔案中
