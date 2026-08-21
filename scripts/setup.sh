#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PCBSDA_VENV:-$PROJECT_ROOT/.venv}"
DRIVE_URL="${1:-${PST2026_DRIVE_URL:-}}"
DOWNLOAD_DIR="$PROJECT_ROOT/data/PST2026"

if [[ -z "$DRIVE_URL" ]]; then
    echo "Usage: bash scripts/setup.sh GOOGLE_DRIVE_PST2026_FOLDER_URL" >&2
    exit 2
fi

command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }
command -v tar >/dev/null || { echo "tar is required" >&2; exit 1; }
command -v zstd >/dev/null || { echo "zstd is required to unpack .tar.zst files" >&2; exit 1; }

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$PROJECT_ROOT/requirements.txt"
"$VENV_DIR/bin/python" -m pip install -e "$PROJECT_ROOT/ours" gdown

mkdir -p "$DOWNLOAD_DIR"
"$VENV_DIR/bin/gdown" --folder "$DRIVE_URL" --remaining-ok --continue --output "$DOWNLOAD_DIR"

declare -A EXPECTED_MD5=(
    ["dataset/gpickle-preprocessed-raw-data.tar.gz"]="5a4059bb403401106a597fc62faec86b"
    ["experiment/malconv/malconv-artifacts.tar.zst"]="a69b25863c57c654d87cc5d8338b499a"
    ["experiment/imcfn/imcfn-artifacts.tar.zst"]="0e72f497fd1b631a692fe950b4b064ee"
    ["experiment/fcgat/fcgat-artifacts.tar.zst"]="fc2a7f746ab2523050ae7748764b11e5"
    ["experiment/gemal/gemal-artifacts.tar.zst"]="c645ed77d678881661c7155f549c87e1"
    ["experiment/word2vec/word2vec-training.tar.zst"]="843f8535f1fb3302e38df885daf98d96"
    ["experiment/word2vec/word2vec-cbow-00.tar.zst"]="ec90c59ad280eecc5286c94d5bd5e99d"
    ["experiment/word2vec/word2vec-cbow-01.tar.zst"]="f670fadd91d105c03125aff1720ae577"
    ["experiment/word2vec/word2vec-cbow-02.tar.zst"]="52b7bce00474bf2535cbd5ffb35176b6"
    ["experiment/word2vec/word2vec-cbow-03.tar.zst"]="8dfd5488e34ce84766787c2c23d488cf"
    ["experiment/unimap/unimap-artifacts.tar.zst"]="9513e5af5b1409aeb5458a59c6c2c67e"
    ["experiment/unimap/unimap-embeddings.tar.zst"]="6564c4cf893c4097f376fc41c73e1470"
    ["pretrained/pcbsda-pretraining-artifacts.tar.zst"]="55533a2368fdad7d729ad5b91188b6fb"
)

echo "Verifying downloaded archives..."
for relative_path in "${!EXPECTED_MD5[@]}"; do
    archive="$DOWNLOAD_DIR/$relative_path"
    [[ -f "$archive" ]] || { echo "Missing archive: $relative_path" >&2; exit 1; }
    actual="$(md5sum "$archive" | cut -d' ' -f1)"
    [[ "$actual" == "${EXPECTED_MD5[$relative_path]}" ]] || {
        echo "Checksum mismatch: $relative_path" >&2
        exit 1
    }
done

echo "Restoring data into the repository..."
mkdir -p "$PROJECT_ROOT/ours/outputs/raw_data/gnn"
tar -xzf "$DOWNLOAD_DIR/dataset/gpickle-preprocessed-raw-data.tar.gz" \
    -C "$PROJECT_ROOT/ours/outputs/raw_data/gnn"

for archive in \
    "$DOWNLOAD_DIR/experiment/malconv/malconv-artifacts.tar.zst" \
    "$DOWNLOAD_DIR/experiment/imcfn/imcfn-artifacts.tar.zst" \
    "$DOWNLOAD_DIR/experiment/fcgat/fcgat-artifacts.tar.zst" \
    "$DOWNLOAD_DIR/experiment/gemal/gemal-artifacts.tar.zst" \
    "$DOWNLOAD_DIR/experiment/word2vec/word2vec-training.tar.zst" \
    "$DOWNLOAD_DIR/experiment/word2vec/word2vec-cbow-00.tar.zst" \
    "$DOWNLOAD_DIR/experiment/word2vec/word2vec-cbow-01.tar.zst" \
    "$DOWNLOAD_DIR/experiment/word2vec/word2vec-cbow-02.tar.zst" \
    "$DOWNLOAD_DIR/experiment/word2vec/word2vec-cbow-03.tar.zst" \
    "$DOWNLOAD_DIR/experiment/unimap/unimap-artifacts.tar.zst" \
    "$DOWNLOAD_DIR/experiment/unimap/unimap-embeddings.tar.zst" \
    "$DOWNLOAD_DIR/pretrained/pcbsda-pretraining-artifacts.tar.zst"
do
    tar -I zstd -xf "$archive" -C "$PROJECT_ROOT"
done

echo "Activate with: source '$VENV_DIR/bin/activate'"
echo "Data restored. See README.md for experiment entry points."
