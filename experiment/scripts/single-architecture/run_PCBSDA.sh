#!/bin/bash
# ============================================================
# Single-Architecture experiments: RoBERTa → Word2Vec → Email
# ============================================================

GMAIL_APP_PASSWORD=$(cat ~/.gmail_app_password)
GMAIL_USER="yuchlin00@gmail.com"
PROJECT_ROOT="/home/tommy/Projects/PCBSDA"
LOG_DIR="$PROJECT_ROOT/experiment/outputs/logs"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$START_TIME] ===== Start ====="

# --- RoBERTa ---
echo "[$(date '+%H:%M:%S')] Running RoBERTa..."
python experiment/single-architecture/RoBERTa/run.py 2>&1 | tee "$LOG_DIR/roberta_run.log"
ROBERTA_EXIT=${PIPESTATUS[0]}
echo "[$(date '+%H:%M:%S')] RoBERTa done (exit $ROBERTA_EXIT)"

# --- Word2Vec ---
echo "[$(date '+%H:%M:%S')] Running Word2Vec..."
python experiment/single-architecture/Word2Vec/run.py 2>&1 | tee "$LOG_DIR/w2v_run.log"
W2V_EXIT=${PIPESTATUS[0]}
echo "[$(date '+%H:%M:%S')] Word2Vec done (exit $W2V_EXIT)"

END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$END_TIME] ===== Done ====="

# --- Email ---
ROBERTA_EXIT=$ROBERTA_EXIT \
W2V_EXIT=$W2V_EXIT \
START_TIME="$START_TIME" \
END_TIME="$END_TIME" \
GMAIL_USER="$GMAIL_USER" \
GMAIL_APP_PASSWORD="$GMAIL_APP_PASSWORD" \
python3 << 'PYEOF'
import smtplib, os
from email.mime.text import MIMEText

user = os.environ["GMAIL_USER"]
pwd  = os.environ["GMAIL_APP_PASSWORD"]
rob  = "成功" if os.environ["ROBERTA_EXIT"] == "0" else f"失敗 (exit {os.environ['ROBERTA_EXIT']})"
w2v  = "成功" if os.environ["W2V_EXIT"]     == "0" else f"失敗 (exit {os.environ['W2V_EXIT']})"

body = f"""PCBSDA Single-Architecture 實驗執行完畢

RoBERTa  : {rob}
Word2Vec : {w2v}

開始：{os.environ['START_TIME']}
結束：{os.environ['END_TIME']}

結果路徑：/home/tommy/Projects/PCBSDA/experiment/outputs/results/
Log 路徑：/home/tommy/Projects/PCBSDA/experiment/outputs/logs/
"""

msg = MIMEText(body, "plain", "utf-8")
msg["Subject"] = f"[PCBSDA] 實驗完成 {os.environ['END_TIME']}"
msg["From"]    = user
msg["To"]      = user

with smtplib.SMTP("smtp.gmail.com", 587) as s:
    s.starttls()
    s.login(user, pwd)
    s.send_message(msg)
    print("Email sent!")
PYEOF
