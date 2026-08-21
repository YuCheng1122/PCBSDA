# 語料 Normalize 規則說明

語料來源：Ghidra 逆向工具輸出的 assembly，每行為一個 basic block，instruction 之間以空格分隔。
目標：產生適合 fastText 訓練的 sentence，instruction 內部 token 以 `~` 連接，instruction 之間以空格分隔。

## 共通規則

- **立即數（immediate）**：所有數字常數（hex / decimal）統一替換為 `0`，負數替換為 `-0`
- **分支/跳轉目標地址**：`bl 0x1234`、`jal 0x1234` 等直接地址替換為 `<FOO>`
- **間接跳轉（暫存器）**：保留暫存器名稱（e.g. `CALL~RAX`）
- **Instruction 格式**：`OPCODE~OP1~OP2`，使用 `~` 連接

---

## x86-64

### Ghidra 特有調整

1. **`xword ptr` 分散 token**：Ghidra 把 `qword ptr [RBX + 0x8]` 拆成多個空白分隔的 token，normalizer 先將其重新合併
2. **`INSN.PREFIX` 格式**：Ghidra 有時將 REP prefix 寫成 `CMPSB.REPE`、`MOVSQ.REP` 等，統一展開為 `REPE~CMPSB`、`REP~MOVSQ`
3. **`ST0`–`ST7` 無括號**：x87 FPU 暫存器 Ghidra 輸出為 `ST0` 而非標準的 `ST(0)`，兩種格式都納入暫存器集合

### Normalize 規則

1. **記憶體存取**：`qword ptr [RBP + 0x18]` → `QWORD~PTR~[RBP+0]`，保留暫存器與尋址結構，立即數替換為 `0`
2. **Segment override**：`qword ptr FS:[0x28]` → `QWORD~PTR~FS:[0]`
3. **Scale index**：`[RAX + RBX*0x8]` → `[RAX+RBX*0]`，scale 常數替換為 `0`
4. **LOCK / REP prefix**：與後面的 opcode 融合，e.g. `LOCK~CMPXCHG`、`REP~MOVSB`
5. **跳過無語意 instruction**：`ENDBR64`、`NOP`、`HLT` 直接丟棄

---

## ARM-32

### Ghidra 特有調整

1. **小寫輸出**：Ghidra 輸出全小寫，normalizer 統一轉大寫
2. **空格分隔的 shift 運算元**：`mov r3,r4, lsl #0x2` 中 `lsl` 和 `#0x2` 是分開的 token，需合併處理
3. **`[reg],#imm` post-index**：`ldr r1,[sp],#0x4`（post-index 格式）bracket 後面還有 offset，需特別解析

### Normalize 規則

1. **暫存器別名統一**：`sp→R13`、`lr→R14`、`pc→R15`、`ip→R12`、`sb→R9`、`fp→R11`
2. **Condition code 剝離**：`addeq r4,r0,#0x1` → `ADD~EQ~R4,R0,0`，cond code 成為獨立 token
3. **Set-flags suffix 保留**：`movs`、`adds`、`subs` 等 `s` suffix 保留為 opcode 一部分（e.g. `MOVS`）
4. **STM/LDM 展開**：`stmdb sp!,{r3,lr}` 展開為多條 `STMDB~R13!~R3 STMDB~R13!~R14`
5. **記憶體存取**：`[r3,#0x8]` → `[R3,0]`，`[r3,r0,lsl #0x2]` → `[R3,R0,LSL 0]`
6. **STM/LDM 模式後綴**：`ia/ib/da/db/fd/fa/ed/ea` 是 addressing mode，不是 condition code，保留於 opcode（e.g. `LDMIA`）

---

## MIPS-32

### Ghidra 特有調整

1. **Delay slot 標記**：Ghidra 在 delay slot instruction 前加 `_`（e.g. `_nop`、`_addiu sp,sp,0x20`），normalizer 去掉前綴 `_` 後正常處理
2. **`sub.D D f0,f0,f2`**：浮點運算 Ghidra 有時會多輸出一個型別字元 `D`/`S`/`W` 作為獨立 token，normalizer 自動丟棄
3. **小寫輸出**：Ghidra 輸出全小寫，normalizer 統一轉大寫

### Normalize 規則

1. **ABI 暫存器名稱統一**：`zero→R0`、`at→R1`、`v0→R2`...`ra→R31`、`sp→R29`、`gp→R28` 等
2. **記憶體 offset**：`0x10(sp)` → `<OFF>R29`，去掉具體 offset 值，保留 base register
3. **Indexed load**：`ldxc1 f0,a0(a1)` 的 `a0(a1)` → `<OFF>R4R5`
4. **Opcode 變體合併**：`addiu/addi/addu→ADD`、`subu→SUB`、`lhu→LH`、`lbu→LB`、`lwl/lwr/lwc1→LW` 等
5. **浮點 opcode 保留**：`add.D`、`cvt.d.W`、`movz.D` 等浮點運算 opcode 原樣保留（大寫）
6. **FPU 控制暫存器**：`fcsr`、`fir` 納入暫存器集合
