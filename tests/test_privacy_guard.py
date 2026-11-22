"""Unit tests for src/privacy_guard.py (pure Python; run: python tests/test_privacy_guard.py or pytest)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.privacy_guard import (PrivacyLedger, GuardedDataset, PrivateDataAccessError,  # noqa: E402
                               accountant_total_steps)


class FakeAccountant:
    """Mimics Opacus RDPAccountant.history merging: (sigma, q, steps)."""
    def __init__(self):
        self.history = []

    def step(self, sigma=1.3, q=0.0625):
        if self.history and self.history[-1][:2] == (sigma, q):
            s, qq, n = self.history[-1]
            self.history[-1] = (s, qq, n + 1)
        else:
            self.history.append((sigma, q, 1))


def _simulate(n_steps=5, batch=4, warmup_private=False, activate=True, skip_step_hook=False):
    data = list(range(32))
    ledger = PrivacyLedger(require_accounting=True)
    ds = GuardedDataset(data, ledger)
    acct = FakeAccountant()
    if warmup_private:
        _ = [ds[i] for i in range(batch)]          # the old bug: private reads before accounting
    ledger.record_public_warmup("public_imdb", 4096, 100)
    if activate:
        ledger.activate_accounting(accountant_total_steps(acct))
    for step in range(1, n_steps + 1):
        _ = [ds[(step * batch + j) % len(ds)] for j in range(batch)]
        if not skip_step_hook:
            acct.step()
        ledger.on_logical_step(step, accountant_total_steps(acct))
    return ledger.finalize(accountant_total_steps(acct))


def test_correct_order_passes():
    s = _simulate()
    assert s["ordering_ok"] and s["steps_match_ok"]
    assert s["accounting_activated_event"] < s["first_private_read_event"]
    assert s["accountant_steps_at_first_private_read"] == 0
    assert s["n_private_reads_before_accounting"] == 0
    assert s["accountant_total_steps"] == s["logical_steps"] == 5
    assert s["public_warmup"]["source"] == "public_imdb"


def test_private_read_before_accounting_raises():
    try:
        _simulate(warmup_private=True)
    except PrivateDataAccessError:
        return
    raise AssertionError("private warmup read was not blocked")


def test_never_activated_raises():
    try:
        _simulate(activate=False)
    except PrivateDataAccessError:
        return
    raise AssertionError("training without accounting was not blocked")


def test_unaccounted_steps_detected():
    s = _simulate(skip_step_hook=True)
    assert s["ordering_ok"] and not s["steps_match_ok"]


def test_clean_run_not_enforced():
    ledger = PrivacyLedger(require_accounting=False)
    ds = GuardedDataset([1, 2, 3], ledger)
    _ = ds[0]
    ledger.on_logical_step(1, None)
    s = ledger.finalize(None)
    assert s["ordering_ok"] and s["steps_match_ok"] and s["n_private_reads_before_accounting"] == 1


def test_metadata_window_allows_shape_probe():
    ledger = PrivacyLedger(require_accounting=True)
    ds = GuardedDataset(list(range(8)), ledger)
    with ledger.metadata_access("opacus_dpdataloader_shape_probe"):
        _ = ds[0]
        _ = ds[0]
    try:
        _ = ds[1]          # outside the window, before activation: must still raise
        raise AssertionError("read outside metadata window was not blocked")
    except PrivateDataAccessError:
        pass
    acct = FakeAccountant()
    ledger.activate_accounting(0)
    _ = ds[2]
    acct.step()
    ledger.on_logical_step(1, accountant_total_steps(acct))
    s = ledger.finalize(accountant_total_steps(acct))
    assert s["metadata_reads"] == {"opacus_dpdataloader_shape_probe": 2}
    assert s["ordering_ok"] is False   # the blocked read above counts against ordering
    assert s["n_private_reads_before_accounting"] == 1


def test_metadata_window_clean_order_ok():
    ledger = PrivacyLedger(require_accounting=True)
    ds = GuardedDataset(list(range(8)), ledger)
    with ledger.metadata_access("probe"):
        _ = ds[0]
    ledger.activate_accounting(0)
    acct = FakeAccountant()
    _ = ds[3]
    acct.step()
    ledger.on_logical_step(1, accountant_total_steps(acct))
    s = ledger.finalize(accountant_total_steps(acct))
    assert s["ordering_ok"] and s["steps_match_ok"] and s["n_private_reads_before_accounting"] == 0


def test_step_inside_metadata_window_flagged():
    ledger = PrivacyLedger(require_accounting=True)
    with ledger.metadata_access("probe"):
        ledger.on_logical_step(1, 0)
    ledger.activate_accounting(0)
    s = ledger.finalize(1)
    assert not s["ordering_ok"] and s["metadata_window_violations"]


def test_history_merging():
    a = FakeAccountant()
    for _ in range(3):
        a.step()
    a.step(sigma=2.0)
    assert accountant_total_steps(a) == 4
    assert accountant_total_steps(object()) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"[PASS] {name}")
    print("all privacy_guard tests passed")
