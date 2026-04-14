#!/bin/bash
# Train Word2Vec CBOW (all 4 archs) then embed FCG graphs (FCGAT experiment).
#
# Step 1: Train CBOW for Intel, ARM-32, x86_64, MIPS  (paper: vector_size=100, window=2, epochs=100)
#   Output: experiment/outputs/models/single-architecture/FCGAT/word2vec/{arch}/
#
# Step 2: Embed all graphs using per-arch CBOW
#   Input:  experiment/outputs/raw_data/single-architecture/FCGAT/GNN/gpickle_single/
#   Output: experiment/outputs/embedded_graphs/single-architecture/FCGAT/
#
# Run from PCBSDA root:
#   bash experiment/scripts/single-architecture/run_FCGAT_embed.sh

GMAIL_APP_PASSWORD=$(cat ~/.gmail_app_password)
GMAIL_USER="yuchlin00@gmail.com"

set -e
cd "$(dirname "$0")/../../.."

LOG_DIR="experiment/outputs/logs/fcgat"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/embed_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo " FCGAT: Word2Vec Training + Graph Embedding"
echo " $(date)"
echo "============================================================"

START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

# ── Step 1: Train CBOW for all architectures ─────────────────────
echo ""
echo "--- Step 1: Train Word2Vec CBOW (Intel, ARM-32, x86_64, MIPS) ---"

for arch in Intel ARM-32 x86_64 MIPS; do
    echo ""
    echo "[CBOW] $arch"
    python experiment/single-architecture/FCGAT/train_word2vec.py --arch "$arch"
done

# ── Step 2: Embed graphs (all archs) ─────────────────────────────
echo ""
echo "--- Step 2: Embed graphs (all architectures) ---"
python experiment/single-architecture/FCGAT/batch_embed_graphs.py
EMBED_EXIT=$?

END_TIME=$(date '+%Y-%m-%d %H:%M:%S')

echo ""
echo "============================================================"
echo " Done: $END_TIME"
echo " Log:  $LOG_FILE"
echo "============================================================"

# ── Email ─────────────────────────────────────────────────────────
EMBED_EXIT=$EMBED_EXIT \
START_TIME="$START_TIME" \
END_TIME="$END_TIME" \
LOG_FILE="$LOG_FILE" \
GMAIL_USER="$GMAIL_USER" \
GMAIL_APP_PASSWORD="$GMAIL_APP_PASSWORD" \
python3 << 'PYEOF'
import smtplib, os
from email.mime.text import MIMEText

user   = os.environ["GMAIL_USER"]
pwd    = os.environ["GMAIL_APP_PASSWORD"]
status = "成功" if os.environ["EMBED_EXIT"] == "0" else f"失敗 (exit {os.environ['EMBED_EXIT']})"

body = f"""PCBSDA FCGAT Embedding 執行完畢

狀態  : {status}
開始  : {os.environ['START_TIME']}
結束  : {os.environ['END_TIME']}

輸出  : experiment/outputs/embedded_graphs/single-architecture/FCGAT/
Log   : {os.environ['LOG_FILE']}
"""

msg = MIMEText(body, "plain", "utf-8")
msg["Subject"] = f"[PCBSDA] FCGAT Embedding {status}  {os.environ['END_TIME']}"
msg["From"]    = user
msg["To"]      = user

with smtplib.SMTP("smtp.gmail.com", 587) as s:
    s.starttls()
    s.login(user, pwd)
    s.send_message(msg)
    print("Email sent!")
PYEOF
