#!/bin/bash
# Run GEMAL single-architecture family classification.
# Model: GCN + attention readout, graph embedding → 300-d → classifier
#
# Pipeline per arch:
#   1. Dev/Test split  — 80% dev / 20% held-out test (stratified, seed=42)
#   2. Optuna search   — 30 trials, 5-fold inner CV, maximise F1-macro
#                        Search space: lr (log-uniform), weight_decay, batch_size
#   3. Test evaluation — train on dev 90%, early-stop on dev 10%, evaluate held-out test
#
# Run from PCBSDA root:
#   bash experiment/scripts/single-architecture/run_GEMAL.sh              # all archs
#   bash experiment/scripts/single-architecture/run_GEMAL.sh x86_64       # single arch
#   bash experiment/scripts/single-architecture/run_GEMAL.sh x86_64 --tune-only
#   bash experiment/scripts/single-architecture/run_GEMAL.sh x86_64 --eval-only

GMAIL_APP_PASSWORD=$(cat ~/.gmail_app_password)
GMAIL_USER="yuchlin00@gmail.com"

set -e
cd "$(dirname "$0")/../../.."

ARCH="${1:-}"
shift || true
EXTRA_ARGS="$@"

LOG_DIR="experiment/outputs/logs/gemal"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/gemal_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo " GEMAL Single-Architecture Classification"
echo " Model: GCN + attention readout"
echo " $(date)"
echo "============================================================"

START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

if [ -n "$ARCH" ]; then
    python experiment/single-architecture/GEMAL/run.py --arch "$ARCH" $EXTRA_ARGS
else
    python experiment/single-architecture/GEMAL/run.py $EXTRA_ARGS
fi
GEMAL_EXIT=$?

END_TIME=$(date '+%Y-%m-%d %H:%M:%S')

echo ""
echo "============================================================"
echo " Done: $END_TIME"
echo " Log:  $LOG_FILE"
echo "============================================================"

# ── Email notification ───────────────────────────────────────────────────────
GEMAL_EXIT=$GEMAL_EXIT \
ARCH="${ARCH:-all}" \
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
status = "成功" if os.environ["GEMAL_EXIT"] == "0" else f"失敗 (exit {os.environ['GEMAL_EXIT']})"

body = f"""PCBSDA GEMAL 訓練執行完畢

模型  : GEMAL (GCN + attention readout)
架構  : {os.environ['ARCH']}
狀態  : {status}
開始  : {os.environ['START_TIME']}
結束  : {os.environ['END_TIME']}

結果  : experiment/outputs/results/gemal/
Log   : {os.environ['LOG_FILE']}
"""

msg = MIMEText(body, "plain", "utf-8")
msg["Subject"] = f"[PCBSDA] GEMAL {os.environ['ARCH']} {status}  {os.environ['END_TIME']}"
msg["From"]    = user
msg["To"]      = user

with smtplib.SMTP("smtp.gmail.com", 587) as s:
    s.starttls()
    s.login(user, pwd)
    s.send_message(msg)
    print("Email sent!")
PYEOF
