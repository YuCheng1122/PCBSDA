# PCBSDA Reproducibility Guide

This repository contains the code and configuration needed to reproduce the PCBSDA experiments. Large datasets, embeddings, checkpoints, and experiment outputs are stored separately in Google Drive under `islab/PST2026`.

## Sources consolidated

- `10.118.126.42`: final integration repository and PCBSDA/CCSA work.
- `10.118.126.41`: authoritative Uni-MAP implementation and experiment outputs.
- `10.118.126.40`: earlier experiment outputs and pretraining artifacts.
- GitHub `master`: authoritative version-controlled base.

## Data

Download `islab/PST2026/gpickle.tar.gz`. It contains the reverse-engineered and preprocessed raw graphs, so reproducing the Ghidra extraction step is optional.

```bash
tar -xzf gpickle.tar.gz
```

Additional archives and their SHA-256 checksums are listed in `islab/PST2026/artifacts/manifests/` and in `ARTIFACTS.md`.

## Environment

The experiments were run with the existing `PcodeBERT` Conda environment. From the repository root:

```bash
conda activate PcodeBERT
python -m pip install -r requirements.txt
python -m pip install -e ./ours
```

GPU experiments require a CUDA-compatible PyTorch installation. Output directories are ignored by Git and are created below `experiment/outputs/` or `ours/outputs/`.

## Single-architecture experiments

Run the Python entry points directly; the shell wrappers additionally send personal email notifications and are not required for reproduction.

```bash
python experiment/single-architecture/MalConv/run.py
python experiment/single-architecture/IMCFN/run.py
python experiment/single-architecture/FCGAT/run.py --help
python experiment/single-architecture/GEMAL/run.py --help
python experiment/single-architecture/RoBERTa/run.py
python experiment/single-architecture/Word2Vec/run.py
```

FCGAT preprocessing and embedding are implemented in:

```bash
python experiment/single-architecture/FCGAT/train_word2vec.py --help
python experiment/single-architecture/FCGAT/batch_embed_graphs.py --help
```

## Cross-architecture PCBSDA experiments

The maintained implementations are under `ours/src/`:

```bash
python ours/src/gnn/main_cross.py --help
python ours/src/transfer_learning/dann/main.py --help
python ours/src/transfer_learning/ccsa/main.py --help
python ours/src/transfer_learning/dsne/main.py --help
```

Configurations are under `ours/configs/`.

## Uni-MAP

The `.41` implementation is authoritative. Place the downloaded Uni-MAP artifacts back under `experiment/outputs/` and embeddings under `experiment/cross-architecture/unimap/embeddings/`.

```bash
python experiment/cross-architecture/unimap/normalize.py --help
python experiment/cross-architecture/unimap/trainEmbedding.py --help
python experiment/cross-architecture/unimap/embedSequences.py --help
python experiment/cross-architecture/unimap/train.py --help
```

Example cross-architecture run:

```bash
python experiment/cross-architecture/unimap/train.py --mode cross --pair x86_64,ARM-32
```

Embedding training is intentionally not run to completion during smoke testing. Successful argument parsing, data loading, model construction, and first-batch execution are sufficient validation for long-running jobs.

## Validation scope

The release validation checks syntax, imports, CLI startup, required paths, and short smoke tests. Full embedding and model training are excluded because they are long-running. See `VALIDATION.md` for the recorded results.
