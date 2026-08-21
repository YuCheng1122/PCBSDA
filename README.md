# Domain Adaptation for Cross-Architecture IoT Malware Detection under Data Scarcity

PCBSDA detects IoT malware across CPU architectures using Pcode, graph neural networks, and domain adaptation.

## Quick start

```bash
git clone git@github.com:YuCheng1122/PCBSDA.git
cd PCBSDA
bash scripts/setup.sh 'PST2026_GOOGLE_DRIVE_FOLDER_URL'
source .venv/bin/activate
```

The setup script creates an isolated Python environment, installs `requirements.txt` and the local `ours` package, then downloads the shared artifacts into `data/PST2026` when a Drive URL is provided. Install a CUDA-compatible PyTorch build separately when required by the target GPU.

## Data and artifacts

The preprocessed dataset already contains reverse-engineered graphs; rerunning Ghidra is optional.

```text
PST2026/
├── dataset/gpickle-preprocessed-raw-data.tar.gz
├── experiment/
│   ├── malconv/malconv-artifacts.tar.zst
│   ├── imcfn/imcfn-artifacts.tar.zst
│   ├── fcgat/fcgat-artifacts.tar.zst
│   ├── gemal/gemal-artifacts.tar.zst
│   └── unimap/unimap-artifacts.tar.zst
├── pretrained/pcbsda-pretraining-artifacts.tar.zst
└── manifests/
```

Download only the required archive and extract it from the repository root:

```bash
tar -I zstd -xf data/PST2026/experiment/EXPERIMENT/ARCHIVE.tar.zst
```

Machine-readable MD5 and SHA-256 files are stored under `PST2026/manifests/`. All uploaded archives were checked against their source byte size and MD5.

## Experiments

| Experiment | Code |
|---|---|
| MalConv | `experiment/single-architecture/MalConv/` |
| IMCFN | `experiment/single-architecture/IMCFN/` |
| FCGAT | `experiment/single-architecture/FCGAT/` |
| GEMAL | `experiment/single-architecture/GEMAL/` |
| RoBERTa | `experiment/single-architecture/RoBERTa/` |
| Word2Vec | `experiment/single-architecture/Word2Vec/` |
| Uni-MAP | `experiment/cross-architecture/unimap/` |
| PCBSDA / GNN | `ours/src/gnn/` |
| DANN, CCSA, d-SNE | `ours/src/transfer_learning/` |

Run an entry point with `--help` before training, for example:

```bash
python experiment/single-architecture/MalConv/run.py --help
python experiment/cross-architecture/unimap/train.py --help
```

Uni-MAP code and artifacts use server `.41` as the authoritative source. Earlier experiments and pretraining artifacts originate from `.40`.

## Validation

Python compilation, shell syntax, core imports, experiment CLI startup, portable paths, and representative MalConv/Uni-MAP forward passes were checked on Python 3.12, PyTorch 2.8.0, PyTorch Geometric 2.6.1, and an NVIDIA RTX 3080 Ti.

Full embedding training, hyperparameter search, cross-validation, and long model training were intentionally not run to completion. These jobs were limited to startup and smoke testing.
