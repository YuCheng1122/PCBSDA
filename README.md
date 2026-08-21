# Domain Adaptation for Cross-Architecture IoT Malware Detection under Data Scarcity

PCBSDA detects IoT malware across CPU architectures using Pcode, graph neural networks, and domain adaptation.

## Quick start

Requirements: Ubuntu/Linux, Python 3.12, about 250 GB free disk space, and `zstd`. GPU training was validated with PyTorch 2.8.0, CUDA, and an RTX 3080 Ti.

```bash
git clone https://github.com/YuCheng1122/PCBSDA.git
cd PCBSDA
bash scripts/setup.sh 'https://drive.google.com/drive/folders/14n-fKv-TNy3M9LahkET5IMVcDRt-Gdbk?usp=sharing'
source .venv/bin/activate
```

The setup script does the complete handoff setup: it creates `.venv`, installs all Python packages, downloads every archive, verifies MD5 checksums, and restores all data to the paths used by the code. It is safe to rerun after an interrupted download.

If the target GPU needs a different CUDA build, replace `torch` and `torchvision` in the virtual environment with the matching packages from PyTorch.

## Reproduce experiments

Run commands from the repository root after activating `.venv`. Each single-architecture command runs all four architectures when `--arch` is omitted. Evaluation is performed inside the training run by nested cross-validation; these programs do not have a separate checkpoint-only test command.

| Experiment | Command | Main restored input | Output |
|---|---|---|---|
| MalConv | `python experiment/single-architecture/MalConv/run.py` | `experiment/outputs/raw_data/single_architecture/MalConv/results_raw_byte` | `experiment/outputs/results/malconv` |
| IMCFN | `python experiment/single-architecture/IMCFN/run.py` | `experiment/outputs/raw_data/single_architecture/IMCFN/results_image` | `experiment/outputs/results/single_architecture/IMCFN` |
| FCGAT | `python experiment/single-architecture/FCGAT/run.py` | `experiment/outputs/embedded_graphs/single-architecture/FCGAT` | `experiment/outputs/results/fcgat` |
| GEMAL | `python experiment/single-architecture/GEMAL/run.py` | FCGAT embedded graphs | `experiment/outputs/results/gemal` |
| RoBERTa | `python experiment/single-architecture/RoBERTa/run.py --roberta-tag roberta_20` | `ours/outputs/embedded_graphs/roberta_20` | `experiment/outputs/results/roberta` |
| Word2Vec | `python experiment/single-architecture/Word2Vec/run.py --w2v-model cbow` | `ours/outputs/embedded_graphs/cbow` | `experiment/outputs/results/word2vec` |
| Uni-MAP | See the two commands below | Uni-MAP sequences and mapped vectors | `experiment/outputs/results/cross-architecture/unimap` |
| GNN / PCBSDA | `python -m src.gnn.main_cross` | `ours/outputs/embedded_graphs/roberta_20` | `ours/outputs/results/gnn` |
| ML baseline | `python -m src.ml.main` | `ours/outputs/embedded_graphs/cbow` | `ours/outputs/results/ml` |
| DANN | `python -m src.transfer_learning.dann.main` | `ours/outputs/embedded_graphs/roberta_20` | `ours/outputs/results/dann` |
| CCSA | `python -m src.transfer_learning.ccsa.main` | `ours/outputs/embedded_graphs/roberta_20` | `ours/outputs/results/ccsa` |
| d-SNE | `python -m src.transfer_learning.dsne.main` | `ours/outputs/embedded_graphs/cbow` | `ours/outputs/results/dsne` |

Uni-MAP uses the server `.41` version as authoritative:

```bash
python experiment/cross-architecture/unimap/embedSequences.py --mode cross --pair x86_64,ARM-32
python experiment/cross-architecture/unimap/train.py --mode cross --pair x86_64,ARM-32
```

To reproduce the Word2Vec preprocessing chain instead of using the restored CBOW graphs:

```bash
python -m src.embedding.cbow.main
python ours/src/embedding/batch_embedding_w2v.py
python experiment/single-architecture/Word2Vec/run.py --w2v-model cbow
```

FCGAT preprocessing can likewise be rerun before training:

```bash
python experiment/single-architecture/FCGAT/train_word2vec.py
python experiment/single-architecture/FCGAT/batch_embed_graphs.py
python experiment/single-architecture/FCGAT/run.py
```

## Drive layout

```text
PST2026/
├── dataset/gpickle-preprocessed-raw-data.tar.gz
├── experiment/
│   ├── malconv/malconv-artifacts.tar.zst
│   ├── imcfn/imcfn-artifacts.tar.zst
│   ├── fcgat/fcgat-artifacts.tar.zst
│   ├── gemal/gemal-artifacts.tar.zst
│   ├── word2vec/word2vec-training.tar.zst + four cbow parts
│   ├── unimap/unimap-embeddings.tar.zst
│   └── unimap/unimap-artifacts.tar.zst
├── pretrained/pcbsda-pretraining-artifacts.tar.zst
└── manifests/
```

The raw `gpickle` dataset is already reverse-engineered and preprocessed, so Ghidra is not required for these experiments. Server `.40` contains early experiments and pretraining artifacts; server `.41` is the source of truth for Uni-MAP.

## Validation

Python compilation, shell syntax, imports, CLI startup, portable paths, and representative MalConv/Uni-MAP forward passes were checked. Full long-running training was not rerun during repository cleanup.
