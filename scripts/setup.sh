#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PCBSDA_VENV:-$PROJECT_ROOT/.venv}"
DRIVE_URL="${1:-${PST2026_DRIVE_URL:-}}"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$PROJECT_ROOT/requirements.txt"
"$VENV_DIR/bin/python" -m pip install -e "$PROJECT_ROOT/ours" gdown

if [[ -n "$DRIVE_URL" ]]; then
    mkdir -p "$PROJECT_ROOT/data"
    "$VENV_DIR/bin/gdown" --folder "$DRIVE_URL" --remaining-ok --output "$PROJECT_ROOT/data/PST2026"
else
    echo "Environment installed. To download artifacts, pass the shared PST2026 URL:"
    echo "  bash scripts/setup.sh 'GOOGLE_DRIVE_FOLDER_URL'"
fi

echo "Activate with: source '$VENV_DIR/bin/activate'"
