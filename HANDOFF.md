# PCBSDA Handoff

## Quick start

Clone the repository, then run:

```bash
bash scripts/setup.sh 'PST2026_GOOGLE_DRIVE_FOLDER_URL'
source .venv/bin/activate
```

The script creates an isolated Python environment, installs `requirements.txt` and the local `ours` package, then downloads the shared `PST2026` folder into `data/PST2026`.

For a CUDA GPU, install the PyTorch build matching the machine's CUDA driver if the default wheel is unsuitable.

## Drive layout

```text
PST2026/
├── dataset/
│   └── gpickle-preprocessed-raw-data.tar.gz
├── experiment/
│   ├── fcgat/
│   ├── gemal/
│   ├── imcfn/
│   ├── malconv/
│   └── unimap/
├── pretrained/
└── manifests/
```

Extract `.tar.zst` archives from the repository root so their saved paths are restored correctly:

```bash
tar -I zstd -xf ARCHIVE.tar.zst
```

Uni-MAP is sourced from server `.41`. Pretraining and early FCGAT/GEMAL artifacts are sourced from `.40`.

## Experiments

- MalConv: `experiment/single-architecture/MalConv/`
- IMCFN: `experiment/single-architecture/IMCFN/`
- FCGAT: `experiment/single-architecture/FCGAT/`
- GEMAL: `experiment/single-architecture/GEMAL/`
- RoBERTa: `experiment/single-architecture/RoBERTa/`
- Word2Vec: `experiment/single-architecture/Word2Vec/`
- Uni-MAP: `experiment/cross-architecture/unimap/`
- PCBSDA/domain adaptation: `ours/src/`

See `REPRODUCIBILITY.md`, `ARTIFACTS.md`, and `VALIDATION.md` for commands, checksums, and validation scope. Long training jobs were smoke-tested but not run to completion.
