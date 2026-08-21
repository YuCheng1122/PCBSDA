# Artifact Manifest

Google Drive root: `islab/PST2026`

## Preprocessed dataset

| File | Size | Checksum | Description |
|---|---:|---|---|
| `gpickle.tar.gz` | 2,198,546,340 bytes | MD5 `5a4059bb403401106a597fc62faec86b` | Reverse-engineered and preprocessed raw graph data |

## Consolidated experiment artifacts

| Drive path | Size | MD5 | SHA-256 | Source |
|---|---:|---|---|---|
| `artifacts/server40/server40-early-experiment-outputs.tar.zst` | 6,869,836,082 bytes | `d0bef0e105c709bf045c6fd59e01a986` | `babdbd32fd9cd78b9ecd9058d645f947d9c02b4642a4bcc2f8671e6ab63e04ac` | `.40` early `experiment/outputs` |
| `artifacts/server40/server40-pretraining-artifacts.tar.zst` | 15,979,859,152 bytes | `55533a2368fdad7d729ad5b91188b6fb` | `300e04752fcffe8f722442ef2259b80bdc1aefaf00ba8ee518b5d4dd5cf7437d` | `.40` embeddings, embedded graphs, models, checkpoints, logs, results, and plots; rebuildable cache excluded |
| `artifacts/server41/server41-experiment-outputs.tar.zst` | 21,976,069,425 bytes | `f77212d8bba5d934d60e57869f1e7725` | `ad65645fd867ca356580d25d9f2b5006e02cf3368eccdb8b5927b24ffdaf25cf` | `.41` complete `experiment/outputs`; authoritative Uni-MAP results |

The `.md5` and `.sha256` files beside the archives are the machine-readable manifests. The hashes above were calculated on the source servers before upload and verified against Google Drive metadata after upload.

## Restore

Run from the repository root. Each archive stores paths relative to that root.

```bash
tar -I zstd -xf server40-early-experiment-outputs.tar.zst
tar -I zstd -xf server40-pretraining-artifacts.tar.zst
tar -I zstd -xf server41-experiment-outputs.tar.zst
```

When restoring overlapping outputs, use `.41` for Uni-MAP. Keep `.40` as the early-experiment/pretraining source.

Verify a downloaded archive with:

```bash
sha256sum -c FILE.tar.zst.sha256
```
