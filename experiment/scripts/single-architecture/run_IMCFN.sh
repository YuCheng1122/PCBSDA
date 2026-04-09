#!/bin/bash
# ============================================================
# Single-Architecture experiment: IMCFN → Email
# ============================================================

GMAIL_APP_PASSWORD=$(cat ~/.gmail_app_password)
GMAIL_USER="yuchlin00@gmail.com"
PROJECT_ROOT="/home/tommy/Projects/PCBSDA"
LOG_FILE="$PROJECT_ROOT/experiment/outputs/logs/single_architecture/imcfn_run.log"

mkdir -p "$(dirname "$LOG_FILE")"
cd "$PROJECT_ROOT"

START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$START_TIME] ===== Start ====="

# --- IMCFN ---
echo "[$(date '+%H:%M:%S')] Running IMCFN..."
conda run -n PcodeBERT python experiment/single-architecture/IMCFN/run.py 2>&1 | tee "$LOG_FILE"
IMCFN_EXIT=${PIPESTATUS[0]}
echo "[$(date '+%H:%M:%S')] IMCFN done (exit $IMCFN_EXIT)"

END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$END_TIME] ===== Done ====="

# --- Email ---
IMCFN_EXIT=$IMCFN_EXIT \
START_TIME="$START_TIME" \
END_TIME="$END_TIME" \
GMAIL_USER="$GMAIL_USER" \
GMAIL_APP_PASSWORD="$GMAIL_APP_PASSWORD" \
python3 << 'PYEOF'
import smtplib, os
from email.mime.text import MIMEText

user   = os.environ["GMAIL_USER"]
pwd    = os.environ["GMAIL_APP_PASSWORD"]
imcfn  = "成功" if os.environ["IMCFN_EXIT"] == "0" else f"失敗 (exit {os.environ['IMCFN_EXIT']})"

body = f"""PCBSDA IMCFN 實驗執行完畢

IMCFN    : {imcfn}

開始：{os.environ['START_TIME']}
結束：{os.environ['END_TIME']}

結果路徑：/home/tommy/Projects/PCBSDA/experiment/outputs/results/single_architecture/IMCFN/
Log 路徑：/home/tommy/Projects/PCBSDA/experiment/outputs/logs/single_architecture/imcfn_run.log
"""

msg = MIMEText(body, "plain", "utf-8")
msg["Subject"] = f"[PCBSDA] IMCFN 完成 {os.environ['END_TIME']}"
msg["From"]    = user
msg["To"]      = user

with smtplib.SMTP("smtp.gmail.com", 587) as s:
    s.starttls()
    s.login(user, pwd)
    s.send_message(msg)
    print("Email sent!")
PYEOF
