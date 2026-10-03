"""Train ONE model (role A or B of a swap pair), then cache its logits on every split.

All attacks run later, in post-processing, on the cached logits (see run_pair.py / src/audit.py).
No weights are ever written to disk. Run each role in its own process (run_pair.py does this).
"""
import argparse
import collections
import json
import os
import random

import numpy as np
import torch
import yaml
from datasets import load_dataset
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from src import data as D
from src.audit import compute_logits
from src.models import get_model, param_groups

C_BIG = 1e4  # "clipping off" threshold for the noise-only diagnostic arm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="YAML hyperparameters; CLI flags override.")
    p.add_argument("--model_name", type=str, default="roberta-base")
    p.add_argument("--train_layers", type=str, default="all", help="'all', 'head', or '9,10,11'")
    p.add_argument("--freeze_embeddings", type=lambda s: s.lower() != "false", default=True)
    p.add_argument("--use_dp", action="store_true")
    p.add_argument("--diag_arm", type=str, default="none", choices=["none", "clip_only", "noise_only"],
                   help="Non-DP diagnostic arms of the 2x2 clipping/noise factorial (needs --use_dp).")
    p.add_argument("--target_epsilon", type=float, default=5.0)
    p.add_argument("--target_delta", type=float, default=1e-5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=64, help="Logical batch size")
    p.add_argument("--physical_batch", type=int, default=16, help="Max physical batch under DP (BatchMemoryManager)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--role", type=str, default="A", choices=["A", "B"])
    p.add_argument("--exp_name", type=str, default="test_run")
    p.add_argument("--cache_dir", type=str, default="results/cache")
    p.add_argument("--pilot", action="store_true",
                   help="LR pilot: train on D_A, report accuracy on D_tune, write results/pilot.csv only.")
    p.add_argument("--debug_small", action="store_true", help="Tiny splits for pipeline smoke tests.")
    return p


def parse_args(argv=None):
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        with open(pre.config) as f:
            cfg = yaml.safe_load(f) or {}
        valid = {a.dest for a in parser._actions} - {"help", "config"}
        unknown = set(cfg) - valid
        if unknown:
            raise ValueError(f"Unknown keys in {pre.config}: {sorted(unknown)}")
        if "train_layers" in cfg:
            cfg["train_layers"] = str(cfg["train_layers"])
        parser.set_defaults(**cfg)
    args = parser.parse_args(argv)
    if args.diag_arm != "none" and not args.use_dp:
        raise ValueError("--diag_arm requires --use_dp")
    return args


# --------------------------------------------------------------------------- telemetry
def attach_telemetry(optimizer, gmap, clip_c):
    """Wraps DPOptimizer to record per-sample grad norms, clip fraction, group energy shares, noise/signal."""
    st = {"n": 0, "clipped": 0, "norms": [], "share": collections.defaultdict(float), "ns": []}
    orig_clip, orig_noise = optimizer.clip_and_accumulate, optimizer.add_noise

    def clip_and_accumulate():
        ps = [p for p in optimizer.params if getattr(p, "grad_sample", None) is not None]
        sq = collections.defaultdict(lambda: 0.0)
        for p in ps:
            g = p.grad_sample.reshape(p.grad_sample.shape[0], -1)
            sq[gmap[id(p)]] = sq[gmap[id(p)]] + g.float().pow(2).sum(dim=1)
        total = sum(sq.values())
        norms = total.sqrt()
        n = norms.numel()
        st["n"] += n
        st["clipped"] += int((norms > clip_c).sum())
        if len(st["norms"]) < 20000:
            st["norms"].extend(norms.detach().cpu().tolist())
        for name, v in sq.items():
            st["share"][name] += float((v / total.clamp_min(1e-30)).sum())
        return orig_clip()

    def add_noise():
        signal = torch.sqrt(sum(p.summed_grad.float().pow(2).sum() for p in optimizer.params))
        orig_noise()
        noise = torch.sqrt(sum((p.grad.float() - p.summed_grad.float()).pow(2).sum() for p in optimizer.params))
        st["ns"].append(float(noise / signal.clamp_min(1e-30)))

    optimizer.clip_and_accumulate, optimizer.add_noise = clip_and_accumulate, add_noise
    return st


def summarize_telemetry(st):
    if not st or st["n"] == 0:
        return {}
    norms = np.array(st["norms"])
    share = {k: v / st["n"] for k, v in st["share"].items()}
    return {
        "clip_fraction": st["clipped"] / st["n"],
        "median_grad_norm": float(np.median(norms)),
        "p10_grad_norm": float(np.percentile(norms, 10)),
        "p90_grad_norm": float(np.percentile(norms, 90)),
        "head_energy_share": share.get("head", 0.0),
        "group_energy_share": share,
        "noise_signal_ratio": float(np.mean(st["ns"])) if st["ns"] else None,
    }


# --------------------------------------------------------------------------- data
def load_splits(args, tokenizer):
    raw = load_dataset("nyu-mll/glue", "sst2")
    sizes = ({"A": 32, "B": 32, "out": 32, "pop": 32, "tune": 64} if args.debug_small else D.SPLIT_SIZES)
    items = D.assign_families(D.build_pool(raw["train"]["sentence"], raw["train"]["label"]))
    part = D.make_partition(items, args.seed, sizes)
    idx = D.partition_indices(part)

    def tok(examples):
        return tokenizer(examples["sentence"], truncation=True, max_length=128)

    def make(ds):
        ds = ds.map(tok, batched=True).remove_columns(["sentence", "idx"])
        ds.set_format("torch")
        return ds

    splits = {name: make(raw["train"].select(i)) for name, i in idx.items()}
    val = raw["validation"]
    if args.debug_small:
        val = val.select(range(64))
    splits["val"] = make(val)
    return splits, D.partition_hash(part)


# --------------------------------------------------------------------------- training
def train_once(args, splits, collator, device, phys_batch):
    init_seed = D._h(args.seed, "init", args.role) % (2 ** 31)
    set_seed(init_seed)
    train_ds = splits[args.role]
    n_train = len(train_ds)
    gen = torch.Generator()
    gen.manual_seed(init_seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collator, generator=gen)

    model, trainable_params, fraction = get_model(args.model_name, 2, args.train_layers, args.freeze_embeddings)
    gmap = param_groups(model)
    model.to(device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    privacy_engine, sigma, telemetry = None, None, None
    use_bmm = False
    if args.use_dp:
        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)
            model.to(device)
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=args.lr)
        sample_rate = 1.0 / len(train_loader)
        sigma0 = get_noise_multiplier(target_epsilon=args.target_epsilon, target_delta=args.target_delta,
                                      sample_rate=sample_rate, steps=args.max_steps, accountant="rdp")
        clip_c, sigma = args.max_grad_norm, sigma0
        if args.diag_arm == "clip_only":
            sigma = 0.0
        elif args.diag_arm == "noise_only":
            clip_c, sigma = C_BIG, sigma0 * args.max_grad_norm / C_BIG  # same absolute noise std, no clipping
        privacy_engine = PrivacyEngine(accountant="rdp")
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model, optimizer=optimizer, data_loader=train_loader,
            noise_multiplier=sigma, max_grad_norm=clip_c)
        telemetry = attach_telemetry(optimizer, gmap, clip_c)
        use_bmm = True
        print(f"DP on | arm={args.diag_arm} | sigma0={sigma0:.4f} | used sigma={sigma:.6f} | C={clip_c} | "
              f"d_trainable={trainable_params:,}")

    warm = max(1, int(args.warmup_frac * args.max_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warm))
    criterion = torch.nn.CrossEntropyLoss()

    losses, steps = [], 0
    mem = BatchMemoryManager(data_loader=train_loader, max_physical_batch_size=phys_batch,
                             optimizer=optimizer) if use_bmm else None
    loader = mem if mem is not None else train_loader
    ctx = loader if mem is not None else _Null(loader)
    with ctx as active_loader:
        while steps < args.max_steps:
            for batch in active_loader:
                ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
                labels = (batch["labels"] if "labels" in batch else batch["label"]).to(device)
                loss = criterion(model(input_ids=ids, attention_mask=mask).logits, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss.item())
                if not use_bmm or not optimizer._is_last_step_skipped:
                    steps += 1
                    sched.step()
                    if steps % 100 == 0:
                        print(f"  step {steps}/{args.max_steps} loss(last100)={np.mean(losses[-100:]):.4f}")
                    if steps >= args.max_steps:
                        break

    eps_spent = None
    if args.use_dp and args.diag_arm == "none":
        eps_spent = float(privacy_engine.get_epsilon(args.target_delta))
    raw_model = model._module if args.use_dp else model
    return {
        "model": raw_model, "trainable_params": trainable_params, "fraction": fraction,
        "noise_multiplier": sigma, "epsilon_spent": eps_spent, "n_train": n_train,
        "final_train_loss": float(np.mean(losses[-50:])),
        "telemetry": summarize_telemetry(telemetry),
    }


class _Null:
    def __init__(self, x): self.x = x
    def __enter__(self): return self.x
    def __exit__(self, *a): return False


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Device: {device} | exp={args.exp_name} seed={args.seed} role={args.role} cfg={args.config}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    splits, p_hash = load_splits(args, tokenizer)

    result, phys = None, args.physical_batch
    while result is None:
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            result = train_once(args, splits, collator, device, phys)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if phys <= 1:
                raise
            phys //= 2
            print(f"[!] CUDA OOM: retrying with physical batch {phys}")
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0

    model = result["model"]
    if args.pilot:
        logits, labels = compute_logits(model, splits["tune"], device, collate_fn=collator)
        acc = float((logits.argmax(1) == labels).mean() * 100)
        os.makedirs("results", exist_ok=True)
        line = (f"{args.exp_name},{args.train_layers},{args.use_dp},{args.lr},{args.seed},"
                f"{acc:.2f},{result['final_train_loss']:.4f}\n")
        new = not os.path.exists("results/pilot.csv")
        with open("results/pilot.csv", "a") as f:
            if new:
                f.write("exp_name,train_layers,use_dp,lr,seed,tune_accuracy,final_train_loss\n")
            f.write(line)
        print(f"[pilot] tune accuracy {acc:.2f}%")
        return

    arrays = {}
    for name in ("A", "B", "out", "pop", "val"):
        lg, lb = compute_logits(model, splits[name], device, collate_fn=collator)
        arrays[f"logits_{name}"], arrays[f"labels_{name}"] = lg, lb

    tag_dir = os.path.join(args.cache_dir, f"{args.exp_name}_s{args.seed}")
    os.makedirs(tag_dir, exist_ok=True)
    tmp_npz = os.path.join(tag_dir, f"model_{args.role}.tmp.npz")
    np.savez_compressed(tmp_npz, **arrays)
    os.replace(tmp_npz, os.path.join(tag_dir, f"model_{args.role}.npz"))
    meta = {k: v for k, v in result.items() if k != "model"}
    meta.update({"args": vars(args), "partition_hash": p_hash, "peak_vram_gb": peak,
                 "physical_batch_used": phys})
    tmp_json = os.path.join(tag_dir, f"model_{args.role}.tmp.json")
    with open(tmp_json, "w") as f:
        json.dump(meta, f, indent=1, default=float)
    os.replace(tmp_json, os.path.join(tag_dir, f"model_{args.role}.json"))
    print(f"[✓] cached {tag_dir}/model_{args.role}.npz (peak VRAM {peak:.2f} GB)")


if __name__ == "__main__":
    main()
