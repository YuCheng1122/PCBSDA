"""
Build single-architecture classification datasets.
For each of the 4 target architectures, keep only families with >500 samples,
then sample exactly 500 per (arch, family) pair.
"""

import pandas as pd
from pathlib import Path

SEED = 42
THRESHOLD = 250
SAMPLE_N = 250
TARGET_ARCHS = ['Intel', 'ARM-32', 'x86_64', 'MIPS']

src = Path(__file__).parent.parent / 'csv' / 'cross_architecture_dataset_clean.csv'
df = pd.read_csv(src)

df_target = df[df['CPU'].isin(TARGET_ARCHS)].copy()

# For each (arch, family), keep only if count > THRESHOLD, then sample SAMPLE_N
parts = []
for arch in TARGET_ARCHS:
    arch_df = df_target[df_target['CPU'] == arch]
    fam_counts = arch_df['family'].value_counts()
    valid_families = fam_counts[fam_counts > THRESHOLD].index.tolist()
    for fam in valid_families:
        subset = arch_df[arch_df['family'] == fam]
        sampled = subset.sample(n=SAMPLE_N, random_state=SEED)
        parts.append(sampled)
    print(f"{arch}: families {valid_families} → {len(valid_families) * SAMPLE_N} rows")

result = pd.concat(parts).reset_index(drop=True)

out_path = src.parent / 'single_arch_dataset.csv'
result.to_csv(out_path, index=False)
print(f"\nSaved {len(result)} rows to {out_path}")
print(result.groupby(['CPU', 'family']).size().unstack(fill_value=0))
