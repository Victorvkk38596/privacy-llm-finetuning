"""Smoke tests: DP accounting starts before any private training example is used.

Run:  python -m pytest -q tests/test_privacy_accounting.py   (or python tests/test_privacy_accounting.py)

Each case trains a tiny DP run (late block 11 + head, 3 logical steps) and checks:
  * private_accesses_before_dp == 0 unless the legacy private warmup is explicitly requested;
  * the RDP accountant recorded exactly as many noisy steps as the optimizer took;
  * the first private batch is fetched only after the PrivacyEngine is attached
    (verified by hooking PrivacyEngine.make_private and the dataset access counter).
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import opacus
import src.train as T

BASE = {
    "exp_name": "smoke_accounting",
    "train_layers": "11",
    "use_dp": True,
    "target_epsilon": 5.0,
    "target_delta": 1e-5,
    "max_steps": 3,
    "logical_batch_size": 16,
    "physical_batch_size": 8,
    "lr": 1e-3,
}


def _run(extra):
    events = []
    orig_make_private = opacus.PrivacyEngine.make_private
    orig_getitem = T.TextClassificationDataset.__getitem__

    def make_private(self, *a, **k):
        events.append(("make_private", None))
        return orig_make_private(self, *a, **k)

    def getitem(self, idx):
        if getattr(self, "_is_private_train", False) and not any(e[0] == "first_private" for e in events):
            events.append(("first_private", None))
        return orig_getitem(self, idx)

    # Tag the private train dataset: it is the first dataset built from partitions[D_A/D_B].
    orig_init = T.TextClassificationDataset.__init__
    state = {"tagged": False}

    def init(self, items, tokenizer, max_length=128):
        orig_init(self, items, tokenizer, max_length)
        if not state["tagged"]:
            self._is_private_train = True
            state["tagged"] = True

    opacus.PrivacyEngine.make_private = make_private
    T.TextClassificationDataset.__getitem__ = getitem
    T.TextClassificationDataset.__init__ = init
    try:
        with tempfile.TemporaryDirectory() as d:
            res = T.train_single_model("M_A", "A", {**BASE, **extra}, seed=42, output_dir=d)
    finally:
        opacus.PrivacyEngine.make_private = orig_make_private
        T.TextClassificationDataset.__getitem__ = orig_getitem
        T.TextClassificationDataset.__init__ = orig_init
    return res, [e[0] for e in events]


def test_no_warmup():
    res, ev = _run({"head_warmup_source": "none"})
    assert res["private_accesses_before_dp"] == 0
    assert res["accountant_steps"] == res["steps_completed"] == 3
    assert ev.index("make_private") < ev.index("first_private"), ev
    assert res["epsilon_spent"] <= 5.0 + 1e-6


def test_public_warmup():
    res, ev = _run({"head_warmup_source": "public", "head_warmup_steps": 2})
    assert res["private_accesses_before_dp"] == 0
    assert res["accountant_steps"] == 3
    assert ev.index("make_private") < ev.index("first_private"), ev


def test_legacy_private_warmup_is_detected():
    res, ev = _run({"head_warmup_source": "private_unaccounted", "head_warmup_steps": 2})
    # 2 warmup batches of 16 private examples were fetched before accounting started.
    assert res["private_accesses_before_dp"] == 32
    assert ev.index("first_private") < ev.index("make_private"), ev


if __name__ == "__main__":
    for fn in (test_no_warmup, test_public_warmup, test_legacy_private_warmup_is_detected):
        fn()
        print(f"[PASS] {fn.__name__}")
