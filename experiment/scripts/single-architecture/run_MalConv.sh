#!/bin/bash
# ============================================================
# Single-Architecture experiment: MalConv (raw bytes) → Email
# ============================================================

GMAIL_APP_PASSWORD=$(cat ~/.gmail_app_password)
GMAIL_USER="yuchlin00@gmail.com"
PROJECT_ROOT="/home/tommy/Projects/PCBSDA"
LOG_DIR="$PROJECT_ROOT/experiment/outputs/logs"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$START_TIME] ===== Start ====="

# --- MalConv ---
echo "[$(date '+%H:%M:%S')] Running MalConv..."
python experiment/single-architecture/MalConv/run.py 2>&1 | tee "$LOG_DIR/malconv_run.log"
MALCONV_EXIT=${PIPESTATUS[0]}
echo "[$(date '+%H:%M:%S')] MalConv done (exit $MALCONV_EXIT)"

END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$END_TIME] ===== Done ====="

# --- Email ---
MALCONV_EXIT=$MALCONV_EXIT \
START_TIME="$START_TIME" \
END_TIME="$END_TIME" \
GMAIL_USER="$GMAIL_USER" \
GMAIL_APP_PASSWORD="$GMAIL_APP_PASSWORD" \
python3 << 'PYEOF'
import smtplib, os
from email.mime.text import MIMEText

user = os.environ["GMAIL_USER"]
pwd  = os.environ["GMAIL_APP_PASSWORD"]
malc = "成功" if os.environ["MALCONV_EXIT"] == "0" else f"失敗 (exit {os.environ['MALCONV_EXIT']})"

body = f"""PCBSDA MalConv 實驗執行完畢

MalConv  : {malc}

開始：{os.environ['START_TIME']}
結束：{os.environ['END_TIME']}

結果路徑：/home/tommy/Projects/PCBSDA/experiment/outputs/malConv/results/
Log 路徑：/home/tommy/Projects/PCBSDA/experiment/outputs/malConv/logs/
"""

msg = MIMEText(body, "plain", "utf-8")
msg["Subject"] = f"[PCBSDA] MalConv 完成 {os.environ['END_TIME']}"
msg["From"]    = user
msg["To"]      = user

with smtplib.SMTP("smtp.gmail.com", 587) as s:
    s.starttls()
    s.login(user, pwd)
    s.send_message(msg)
    print("Email sent!")
PYEOF
