# Pcode 抽取腳本(Ghidra Headless)

對應 pipeline 第 1 步:批次用 Ghidra 反編譯二進位檔,抽取每個 function 的 High Pcode opcode 序列與 function call graph(DOT + JSON)。

> **使用前請注意**:這兩個腳本本身沒有任何機器專屬路徑,但需要你自己先裝好下面「需求」列的 Ghidra + Ghidrathon 與 GNU `parallel`,否則跑不起來(Ghidrathon 沒裝會找不到 Python 3 執行環境;`parallel` 沒裝會直接報 command not found)。

## 檔案

- `get_function_call_Pcode.sh`:批次處理入口,對資料夾內所有檔案平行呼叫 Ghidra headless。
- `ghidra_function_Pcode_script.py`:Ghidra postScript(Ghidrathon / Python 3),實際做反編譯與 Pcode 抽取。

## 需求

- Ghidra(已測試 11.3.2)並安裝 [Ghidrathon](https://github.com/mandiant/Ghidrathon)(讓 postScript 用 Python 3 執行)
- GNU `parallel`(`sudo apt install parallel`)

## 用法

```bash
./get_function_call_Pcode.sh <ghidra_headless_path> <program_folder> [output_dir] [timeout_sec]
```

範例:

```bash
./get_function_call_Pcode.sh \
  /path/to/ghidra/support/analyzeHeadless \
  ./samples \
  ./output \
  1200
```

- `program_folder` 內的每個檔案都會被當成一個待分析的二進位檔(遞迴掃描)。
- 第 5 個參數 `csv_path` 目前保留但尚未使用。

## 輸出

```
output/
├── extraction.log              # 逐檔案處理紀錄
├── timed_out_files.txt         # 逾時的檔案
├── import_failed_files.txt     # Ghidra 匯入失敗的檔案
└── results/<檔名>/
    ├── <檔名>.dot               # function call graph
    └── <檔名>.json              # 每個 function 的 entry point、function_name、instructions(address/operation/opcode)
```

## 已測試驗證

用 `/bin/ls`、`/bin/cat` 實測跑通,並確認錯誤情境(非法檔案格式)會正確歸類到 `import_failed_files.txt`。
