# Validation Record

Validation date: 2026-08-21

## Environment

- Host role: integration server (`.42`)
- Python: 3.12.11 (`PcodeBERT` Conda environment)
- PyTorch: 2.8.0+cu128
- PyTorch Geometric: 2.6.1
- GPU: NVIDIA GeForce RTX 3080 Ti

Missing experiment dependencies discovered during validation were added to `requirements.txt`; `torchvision 0.23.0` and `fasttext 0.9.3` were installed and rechecked.

## Passed checks

- Python compilation: `ours/src`, `ours/configs`, all single-architecture experiments, Uni-MAP, and dataset scripts.
- Shell syntax: every script under `experiment/scripts/single-architecture` and `ours/scripts`.
- Core imports: torch, torchvision, torch-geometric, NumPy, pandas, scikit-learn, SciPy, gensim, fasttext, NetworkX, Optuna, matplotlib, seaborn, tqdm, Pillow, and joblib.
- CLI startup: FCGAT, GEMAL, IMCFN, MalConv, RoBERTa, Word2Vec, and all Uni-MAP command-line entry points.
- Model forward smoke tests: MalConv and Uni-MAP PaperLSTM with small random tensors.
- Portable path checks: Uni-MAP and CCSA sensitivity configs resolve paths from the current repository clone.
- Python and shell checks completed without starting full training.

## Intentionally not run to completion

- Word/graph embedding training.
- Hyperparameter searches and cross-validation.
- Full IMCFN, MalConv, FCGAT, GEMAL, RoBERTa, Word2Vec, PCBSDA, DANN, CCSA, d-SNE, and Uni-MAP training.

These jobs are long-running. Their entry points, imports, configuration loading, and representative model forward paths were checked instead.

## Data and artifacts

- Preprocessed graph input: `islab/PST2026/dataset/gpickle-preprocessed-raw-data.tar.gz`
- MD5: `5a4059bb403401106a597fc62faec86b`
- Server `.41` is authoritative for Uni-MAP.
- Server `.40` supplies earlier experiment and pretraining artifacts.
- Uploaded archives and SHA-256 values are recorded in `ARTIFACTS.md`.
