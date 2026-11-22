"""Single source of truth for the second-half core grid (pure Python, no torch).

Three comparisons, seeds 42/43/44, two-model swap pair per (config, seed):
  location : early-3 / middle-3 / late-3 / all-12 at eps=5 (+ clean all-12 for utility context)
  budget   : late-3 at eps = 1 / 5 / 8, same delta (late-3 eps=5 reused from location)
  count    : last-1 / last-3 / last-6 / all-12 blocks + head at eps=5, anchored at the output end
             (last-3 = late-3 and all-12 reused from location)

Layer definitions follow the repo's existing configs (early 0,1,2; middle 4,5,6; late 9,10,11).
Every DP config shares one training recipe (the pilot v2 recipe minus the private warmup).
"""
SEEDS = [42, 43, 44]
NUM_BLOCKS = 12
DELTA = 1.0e-5

# Training settings shared by all DP configs (matched across comparisons).
DP_BASE = {
    "model_name": "roberta-base",
    "freeze_embeddings": True,
    "use_dp": True,
    "target_delta": DELTA,
    "max_grad_norm": 1.0,
    "lr": 1.0e-3,            # backbone; head gets 3x in src/train.py
    "weight_decay": 0.0,
    "warmup_frac": 0.05,     # LR-schedule warmup (data independent), not head warmup
    "max_steps": 300,
    "logical_batch_size": 64,
    "physical_batch_size": 16,
    "max_length": 128,
    "head_warmup_source": "none",   # overridden by the runner's --warmup choice
    "head_warmup_steps": 100,       # only used when head_warmup_source is public
    "head_warmup_lr": 1.0e-3,
    "head_warmup_n_examples": 4096,
}

# Clean baseline keeps the repo's existing clean recipe (configs/baseline_full_clean.yaml).
CLEAN_BASE = {
    "model_name": "roberta-base",
    "freeze_embeddings": True,
    "use_dp": False,
    "target_delta": DELTA,
    "max_grad_norm": 1.0,
    "lr": 3.0e-5,
    "weight_decay": 0.0,
    "warmup_frac": 0.05,
    "max_steps": 1000,
    "logical_batch_size": 64,
    "physical_batch_size": 16,
    "max_length": 128,
    "head_warmup_source": "none",
}


def _last(k):
    return ",".join(str(i) for i in range(NUM_BLOCKS - k, NUM_BLOCKS))


# name -> (train_layers, target_epsilon or None, comparisons, short label)
CONFIGS = {
    "clean_all":     ("all",       None, ["location"],                   "clean all-12"),
    "dp_early3_e5":  ("0,1,2",     5.0,  ["location"],                   "early-3"),
    "dp_mid3_e5":    ("4,5,6",     5.0,  ["location"],                   "middle-3"),
    "dp_late3_e5":   (_last(3),    5.0,  ["location", "budget", "count"], "late-3"),
    "dp_all_e5":     ("all",       5.0,  ["location", "count"],          "all-12"),
    "dp_late3_e1":   (_last(3),    1.0,  ["budget"],                     "late-3 eps=1"),
    "dp_late3_e8":   (_last(3),    8.0,  ["budget"],                     "late-3 eps=8"),
    "dp_last1_e5":   (_last(1),    5.0,  ["count"],                      "last-1"),
    "dp_last6_e5":   (_last(6),    5.0,  ["count"],                      "last-6"),
}

# Order of arms within each comparison, and the reference arm for matched-seed differences.
COMPARISONS = {
    "location": {"arms": ["dp_early3_e5", "dp_mid3_e5", "dp_late3_e5", "dp_all_e5"],
                 "context": ["clean_all"], "reference": "dp_late3_e5"},
    "budget":   {"arms": ["dp_late3_e1", "dp_late3_e5", "dp_late3_e8"], "context": [],
                 "reference": "dp_late3_e5"},
    "count":    {"arms": ["dp_last1_e5", "dp_late3_e5", "dp_last6_e5", "dp_all_e5"], "context": [],
                 "reference": "dp_late3_e5"},
}

# Configs needing a single-run validation before launching remaining seeds
# (settings not run in the pilot). Location arms at eps=5 were run in the pilot recipe,
# but the warmup change makes all DP settings new, so the gate covers those.
GATE_CONFIGS = ["dp_early3_e5", "dp_late3_e5", "dp_all_e5"]   # step-1 warmup check, seed 42
NEW_SETTING_CONFIGS = ["dp_mid3_e5", "dp_late3_e1", "dp_late3_e8", "dp_last1_e5", "dp_last6_e5", "clean_all"]

HEAD_PARAMS = 592_130
BLOCK_PARAMS = 7_087_872
TOTAL_PARAMS = 124_647_170
SST2_VAL_MAJORITY_ACC = 50.92   # 444 / 872 positive labels in GLUE SST-2 validation


def expected_trainable(train_layers: str) -> int:
    if train_layers == "all":
        return HEAD_PARAMS + NUM_BLOCKS * BLOCK_PARAMS
    if train_layers == "head":
        return HEAD_PARAMS
    return HEAD_PARAMS + len(train_layers.split(",")) * BLOCK_PARAMS


def exp_name(config_name: str, warmup: str) -> str:
    """Record / logit name. Clean runs never warm up, so they carry no warmup suffix."""
    if CONFIGS[config_name][1] is None:
        return config_name
    return f"{config_name}__w-{warmup}"


def build_config(config_name: str, warmup: str) -> dict:
    layers, eps, comps, _ = CONFIGS[config_name]
    cfg = dict(CLEAN_BASE if eps is None else DP_BASE)
    cfg["train_layers"] = layers
    if eps is not None:
        cfg["target_epsilon"] = eps
        cfg["head_warmup_source"] = warmup
    cfg["config_name"] = config_name
    cfg["comparisons"] = comps
    cfg["exp_name"] = exp_name(config_name, warmup)
    return cfg


def check_spec() -> None:
    """Static checks on the grid definition."""
    for name, (layers, eps, comps, _) in CONFIGS.items():
        for c in comps:
            assert name in COMPARISONS[c]["arms"] + COMPARISONS[c]["context"], (name, c)
    for c, spec in COMPARISONS.items():
        for a in spec["arms"] + spec["context"]:
            assert c in CONFIGS[a][2], (a, c)
        assert spec["reference"] in spec["arms"]
    # count comparison: blocks anchored at the output end
    for a in COMPARISONS["count"]["arms"]:
        layers = CONFIGS[a][0]
        idx = list(range(NUM_BLOCKS)) if layers == "all" else [int(i) for i in layers.split(",")]
        assert idx == list(range(NUM_BLOCKS - len(idx), NUM_BLOCKS)), (a, idx)
        assert CONFIGS[a][1] == 5.0
    # budget comparison: same layer set, same delta
    assert len({CONFIGS[a][0] for a in COMPARISONS["budget"]["arms"]}) == 1
    # location: equal block count for the three k=3 arms
    assert {len(CONFIGS[a][0].split(",")) for a in ["dp_early3_e5", "dp_mid3_e5", "dp_late3_e5"]} == {3}


def write_configs(out_dir: str, warmup: str = "none") -> list:
    """Writes configs/grid/<exp_name>.yaml for every config (for single manual runs)."""
    import os
    import yaml
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name in CONFIGS:
        cfg = build_config(name, warmup)
        path = os.path.join(out_dir, f"{cfg['exp_name']}.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Generated by scripts/grid_spec.py (do not edit by hand). Comparisons: {', '.join(cfg['comparisons'])}\n")
            yaml.safe_dump(cfg, f, sort_keys=False)
        paths.append(path)
    return paths


if __name__ == "__main__":
    import sys
    check_spec()
    if "--write-configs" in sys.argv:
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for w in ("none", "public_imdb"):
            for p in write_configs(os.path.join(root, "configs", "grid"), w):
                print("wrote", os.path.relpath(p, root))
    for n in CONFIGS:
        print(f"{n:14s} layers={CONFIGS[n][0]:22s} eps={CONFIGS[n][1]} "
              f"trainable={expected_trainable(CONFIGS[n][0]):,} ({100*expected_trainable(CONFIGS[n][0])/TOTAL_PARAMS:.2f}%)")
    print(f"{len(CONFIGS)} configs x {len(SEEDS)} seeds = {len(CONFIGS)*len(SEEDS)} swap pairs "
          f"({2*len(CONFIGS)*len(SEEDS)} models)")
