# Artifact Manifest

Google Drive root: `islab/PST2026`

```text
PST2026/
├── dataset/
├── experiment/{malconv,imcfn,fcgat,gemal,unimap}/
├── experiment/{roberta,word2vec,pcbsda,dann,ccsa,dsne}/
├── pretrained/
└── manifests/
```

## Dataset

| Drive path | Size | Checksum | Description |
|---|---:|---|---|
| `dataset/gpickle-preprocessed-raw-data.tar.gz` | 2,198,546,340 bytes | MD5 `5a4059bb403401106a597fc62faec86b` | Reverse-engineered and preprocessed raw graphs |

## Experiment artifacts

| Drive path | Size | MD5 | SHA-256 |
|---|---:|---|---|
| `experiment/malconv/malconv-artifacts.tar.zst` | 2,118,881,178 bytes | `a69b25863c57c654d87cc5d8338b499a` | `bceb8b93e332d4664ba52345481f0ee208e7dba500fb0bb447ae19f07ec521a3` |
| `experiment/imcfn/imcfn-artifacts.tar.zst` | 444,723,053 bytes | `0e72f497fd1b631a692fe950b4b064ee` | `747db2f044115c2cdeaf0ffcdf85287afbda325769f6cf56c08e0463a3489268` |
| `experiment/fcgat/fcgat-artifacts.tar.zst` | 2,002,536,352 bytes | `fc2a7f746ab2523050ae7748764b11e5` | `2ffa67b69783c71551922b8b5705df7890339403de9e3325fe6a7f74c9fce9c5` |
| `experiment/gemal/gemal-artifacts.tar.zst` | 105,754 bytes | `c645ed77d678881661c7155f549c87e1` | `80e095eea45ef6588447aa9282e54453a9a6b72ffc98d7518311df05fb470af5` |
| `experiment/unimap/unimap-artifacts.tar.zst` | 19,413,221,655 bytes | `9513e5af5b1409aeb5458a59c6c2c67e` | `bcd3282576e73926bbca97278336743c1a88bac913ca95409a145846aa94025e` |

RoBERTa, Word2Vec, PCBSDA, DANN, CCSA, and d-SNE have named Drive folders and version-controlled code. Their reusable embeddings, embedded graphs, models, and checkpoints are grouped in the pretraining archive:

| Drive path | Size | MD5 | SHA-256 |
|---|---:|---|---|
| `pretrained/pcbsda-pretraining-artifacts.tar.zst` | 15,979,859,152 bytes | `55533a2368fdad7d729ad5b91188b6fb` | `300e04752fcffe8f722442ef2259b80bdc1aefaf00ba8ee518b5d4dd5cf7437d` |

Checksums were calculated on the source servers; every new Drive archive was verified against its source byte size and MD5. The `.md5` and `.sha256` files under `manifests/` are machine-readable copies.

## Restore

Download only the experiment needed, then extract from the repository root:

```bash
tar -I zstd -xf data/PST2026/experiment/malconv/malconv-artifacts.tar.zst
tar -I zstd -xf data/PST2026/experiment/imcfn/imcfn-artifacts.tar.zst
tar -I zstd -xf data/PST2026/experiment/fcgat/fcgat-artifacts.tar.zst
tar -I zstd -xf data/PST2026/experiment/gemal/gemal-artifacts.tar.zst
tar -I zstd -xf data/PST2026/experiment/unimap/unimap-artifacts.tar.zst
```

Verify a downloaded archive from its directory:

```bash
sha256sum -c FILE.sha256
```
