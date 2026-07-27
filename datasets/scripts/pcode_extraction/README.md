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

## 輸出範例

用 BinKit 資料集裡的 ARM 32-bit binary 實測(`a2ps-4.14_clang-4.0_arm_32_O0_a2ps`,~1.3MB):

```bash
./get_function_call_Pcode.sh \
  /path/to/ghidra/support/analyzeHeadless \
  /home/tommy/Projects/data/BinKit_normal/a2ps \
  ./output \
  1200
```

跑完約 161 秒,抽出 1015 個 function。`<檔名>.dot` 內容(function call graph,節點是 entry point,邊是呼叫關係):

```dot
digraph code {
  "0x11614" [label="_init"];
  "0x11614" -> "0x11b14";
  "0x11ad8" [label="_start"];
  "0x11b14" [label="call_weak_fn"];
  "0x11b38" [label="deregister_tm_clones"];
  "0x11b64" [label="register_tm_clones"];
  "0x11b9c" [label="__do_global_dtors_aux"];
  ...
}
```

`<檔名>.json` 內容(每個 function 的 opcode 序列,可餵給 embedding 步驟):

```json
{
  "0x328dc": {
    "function_name": "sshget_lineno",
    "instructions": [
      {
        "address": "000328e0",
        "operation": "(register, 0x20, 4) LOAD (const, 0x1a1, 4) , (ram, 0x328e8, 4)",
        "opcode": "LOAD"
      },
      {
        "address": "000328e4",
        "operation": " ---  RETURN (const, 0x0, 4) , (register, 0x20, 4)",
        "opcode": "RETURN"
      },
      {
        "address": "000328e4",
        "operation": "(ram, 0x328e8, 4) COPY (ram, 0x328e8, 4)",
        "opcode": "COPY"
      }
    ]
  }
}
```

## 已測試驗證

用 `/bin/ls`、`/bin/cat`,以及上面的 ARM 32-bit 真實樣本實測跑通,並確認錯誤情境(非法檔案格式)會正確歸類到 `import_failed_files.txt`。
