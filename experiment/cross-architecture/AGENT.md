# Cross-Architecture Malware Family Classification — Agent Guide

## 任務描述

跨架構惡意軟體家族分類（Cross-Architecture Family Classification）。
對不同 CPU 架構的資料做分類實驗，進行 GNN-based 多分類任務。

**資料集**：各惡意軟體家族平均取樣（per-family balanced sampling），確保類別均衡。
請看 /home/tommy/Project/PCBSDA/datasets/csv/cross_architecture_dataset_family8.csv

## 實驗流程（Repeated Random Sub-sampling with 10 Seeds）
採用 10 seeds 做不同的 data split 確保取得一個穩定的平均結果:

1. **資料切分 (Data Split)**：將 Source 資料切出 Validation Set；Target 資料切分為極少量 Labeled Train（SDA用）與 Unlabeled Test。**Target Test 絕對不參與調參**。
3. **訓練與測試 (Train & Test)**：用那組鎖死的參數，跑完 10 個 random seeds 的訓練。每次訓練完都在對應的 Target Test Set 算分數，最後把這 10 次的結果取平均。

## 比較方法
- **Source-Only (下界基準)**：不用任何 DA 方法，直接 Source 訓練、Target 測試。
- **DANN**：非監督式做法 (Unsupervised DA)。    
- **CCSA**：監督式/半監督式做法 (Supervised/Semi-Supervised DA)。
  - *消融實驗 (Ablation)*：在 Test Set 上展示拆解版的表現，例如只加對齊 (CSA)、只加分離 (CS) 與完整版 (CCSA) 的對比。

## 規則

- **評估指標**：統一回報 accuracy、precision、recall、f1_micro、f1_macro、AUC。
- **報告格式**：所有指標必須列出 10 次實驗的**平均值 (Mean) ± 標準差 (Std)**。
- **參數分析圖表 (可選)**：可以放不同參數在 Test Set 的折線圖/長條圖，但要在內文註明：「最終模型是依據 Source Validation 決定的，此處僅為展示模型對參數變動的穩定度 (Sensitivity Analysis)」。