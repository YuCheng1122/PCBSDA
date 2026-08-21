from pathlib import Path


def get_lstm_config():
    base_path = Path(__file__).resolve().parents[3]
    unimap = base_path / "experiment" / "cross-architecture" / "unimap"
    emb = unimap / "embeddings"
    out = base_path / "experiment" / "outputs"

    # ── Experiment mode ───────────────────────────────────────────────────
    # "cross" : train on source_cpu, test on target_cpu (cross-architecture)
    # "mono"  : train/val/test all from the same cpu (single-architecture)
    mode = "cross"

    # ── Cross-architecture pairs (used when mode == "cross") ──────────────
    # Each entry: (source_cpu, target_cpu)
    # CPU names must match the 'CPU' column in the CSV.
    cross_pairs = [
        ("x86_64", "ARM-32"),
        ("x86_64", "MIPS"),
        # ("MIPS",   "x86_64"),
        ("ARM-32", "x86_64"),
        # ("MIPS",   "ARM-32"),
        ("ARM-32", "MIPS"),
    ]

    # ── Mono-architecture CPUs (used when mode == "mono") ─────────────────
    mono_cpus = ["x86_64", "ARM-32", "MIPS"]

    # ── Per-pair embedding paths ──────────────────────────────────────────
    # Maps (source_cpu, target_cpu) → (source_mapped_vec, target_mapped_vec, embed_cache_dir)
    # CPU name → directory name under embeddings/
    _cpu_to_dir = {
        "x86_64": "x86_64",
        "ARM-32": "arm_32",
        "MIPS":   "mips_32",
        "Intel":  "x86_64",
    }

    def _pair_cfg(src, tgt):
        sd, td = _cpu_to_dir[src], _cpu_to_dir[tgt]
        pair   = f"{sd}_{td}"
        return {
            "vec_src":         str(emb / pair / f"{sd}_mapped.vec"),
            "vec_tgt":         str(emb / pair / f"{td}_mapped.vec"),
            "embed_cache_dir": str(out / "embedded" / "cross-architecture" / "unimap" / pair),
            "model_output_dir":str(out / "model" / "cross-architecture" / "unimap" / pair),
        }

    def _mono_cfg(cpu):
        d    = _cpu_to_dir[cpu]
        # mono uses the raw (non-mapped) fastText vec
        return {
            "vec_src":         str(emb / f"{d}.vec"),
            "vec_tgt":         None,
            "embed_cache_dir": str(out / "embedded" / "cross-architecture" / "unimap" / f"mono_{d}"),
            "model_output_dir":str(out / "model" / "cross-architecture" / "unimap" / f"mono_{d}"),
        }

    config = {
        # ── Mode ─────────────────────────────────────────────────────────
        "mode":        mode,
        "cross_pairs": cross_pairs,
        "mono_cpus":   mono_cpus,
        "cpu_to_dir":  _cpu_to_dir,
        "pair_cfg_fn": _pair_cfg,
        "mono_cfg_fn": _mono_cfg,

        # ── Data ─────────────────────────────────────────────────────────
        "csv_path": str(base_path / "datasets" / "csv" / "cross_architecture_dataset_family8.csv"),
        "seq_dir":  str(out / "raw_data" / "cross-architecture" / "unimap" / "sequences" / "classification"),

        # family → integer label (multi-class)
        "families": ["dnsamp", "dofloo", "gafgyt", "kaiji",
                     "meterpreter", "mirai", "mobidash", "tsunami"],

        # ── Embedding ────────────────────────────────────────────────────
        "max_len":   6000,
        "embed_dim": 200,

        # ── Training ─────────────────────────────────────────────────────
        "seeds":         [42, 123],
        "test_size":     0.2,   # mono mode: fraction held out as test
        "val_size":      0.2,   # mono mode: fraction of train held out as val
        "batch_size":    32,
        "learning_rate": 0.001,
        "epochs":        200,
        "patience":      20,
        "device":        "cuda",
    }

    return config
