# Single-Architecture Malware Family Classification — Agent Guide

## 任務描述

單架構惡意軟體家族分類（Single-Architecture Family Classification）。
對同一 CPU 架構的資料做分類實驗，進行 GNN-based 多分類任務。

**資料集**：各惡意軟體家族平均取樣（per-family balanced sampling），確保類別均衡。

## 實驗流程（Nested Cross-Validation）

採用 **Nested CV** 框架，確保超參數選擇與最終評估完全分離：

1. **外層 CV（評估）**：5-fold Stratified CV，每個 outer fold 的 test set 在整個流程中完全不可見
2. **內層 CV（超參數選擇）**：在 outer train set 上再做 5-fold inner CV，以 F1-macro 為目標搜尋最佳超參數
3. **Outer Fold 評估**：用 inner CV 選出的最佳超參數，在整個 outer train set 上重新訓練，於 outer test fold 評估
4. **彙總結果**：5 個 outer fold 的 metrics 取 mean ± std 作為最終報告指標（accuracy、precision、recall、f1_micro、f1_macro、AUC）

## 比較方法

FCGAT、IMCFN、MalConv、GEMAL 均為比較的 baseline 論文，各自有獨立實作，透過 `/scripts/single-architecture/` 下的 shell script 執行，跑完自動寄信通知。

## 規則

- **資料隔離**：outer test fold 在 inner CV 超參數搜尋期間絕對不可接觸
- **可重現性**：所有隨機性來源必須固定 seed
  - Outer / Inner KFold：`random_state=42`
  - Optuna TPESampler：`seed=42`
  - 每個 fold 的模型訓練：`42 + fold_idx`
- **評估指標**：統一回報 accuracy、precision、recall、f1_micro、f1_macro、AUC
- **不修改原始實作**：baseline 論文的實作保留原始設定，僅套用 nested CV 評估框架
- **結果儲存**：每個 outer fold 的 best_params 與 metrics 均需儲存，方便事後分析
