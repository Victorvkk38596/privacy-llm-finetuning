"""Private-data access ledger for DP training runs.

Pure Python (no torch import) so it can be unit-tested anywhere.

Purpose: make the ordering "privacy accounting starts before the first update that
uses private training data" a checked invariant instead of a convention.

Usage inside src/train.py:
  ledger = PrivacyLedger(require_accounting=use_dp)
  train_ds = GuardedDataset(TextClassificationDataset(D_A, ...), ledger)
  ... model/optimizer setup, optional PUBLIC warmup (never touches train_ds) ...
  model, optimizer, loader = privacy_engine.make_private(...)
  ledger.activate_accounting(accountant_steps=0)
  ... each logical optimizer step: ledger.on_logical_step(step, accountant_total_steps(acct))
  ledger.finalize(accountant_total_steps(acct))

Any read of a private training example while accounting is not active raises
PrivateDataAccessError when require_accounting=True.
"""
from contextlib import contextmanager
from typing import Any, Dict, List, Optional


class PrivateDataAccessError(RuntimeError):
    """Raised when private training data is read before DP accounting is active."""


class PrivacyLedger:
    def __init__(self, require_accounting: bool, split_name: str = "private_train"):
        self.require_accounting = bool(require_accounting)
        self.split_name = split_name
        self.accounting_active = False
        self.events: List[Dict[str, Any]] = []
        self.n_private_reads = 0
        self.n_private_reads_before_accounting = 0
        self.first_private_read_event: Optional[int] = None
        self.accounting_activated_event: Optional[int] = None
        self.accountant_steps_at_activation: Optional[int] = None
        self.accountant_steps_at_first_private_read: Optional[int] = None
        self.logical_steps = 0
        self.step_mismatches: List[Dict[str, int]] = []
        self.public_warmup: Optional[Dict[str, Any]] = None
        self.finalized: Optional[Dict[str, Any]] = None
        self._current_acct_steps = 0
        self._metadata_reason: Optional[str] = None
        self.metadata_reads: Dict[str, int] = {}
        self.metadata_window_violations: List[str] = []

    # ------------------------------------------------------------------ events
    def _event(self, kind: str, **info) -> int:
        self.events.append({"i": len(self.events), "kind": kind, **info})
        return len(self.events) - 1

    def record_public_warmup(self, source: str, n_examples: int, steps: int, **info) -> None:
        """Documents a non-private stage that used only public data."""
        self.public_warmup = {"source": source, "n_examples": int(n_examples), "steps": int(steps), **info}
        self._event("public_warmup", source=source, n_examples=int(n_examples), steps=int(steps))

    def activate_accounting(self, accountant_steps: int = 0) -> None:
        if self.accounting_active:
            return
        self.accounting_active = True
        self.accountant_steps_at_activation = int(accountant_steps)
        self._current_acct_steps = int(accountant_steps)
        self.accounting_activated_event = self._event("accounting_activated", accountant_steps=int(accountant_steps))

    @contextmanager
    def metadata_access(self, reason: str):
        """Window for library-internal reads that do not feed a parameter update.

        Needed because Opacus' DPDataLoader constructor reads dataset[0] (inside make_private)
        to get tensor shapes/dtypes for empty Poisson batches. Reads in this window are counted
        separately; any optimizer step inside the window is flagged as a violation.
        """
        steps_before = self.logical_steps
        self._metadata_reason = reason
        try:
            yield
        finally:
            self._metadata_reason = None
            if self.logical_steps != steps_before:
                self.metadata_window_violations.append(f"optimizer step inside metadata window '{reason}'")

    def on_private_read(self, index: int) -> None:
        if self._metadata_reason is not None and not self.accounting_active:
            self.metadata_reads[self._metadata_reason] = self.metadata_reads.get(self._metadata_reason, 0) + 1
            if self.metadata_reads[self._metadata_reason] == 1:
                self._event("metadata_read", reason=self._metadata_reason, index=int(index))
            return
        if not self.accounting_active:
            self.n_private_reads_before_accounting += 1
            if self.require_accounting:
                self._event("private_read_before_accounting", index=int(index))
                raise PrivateDataAccessError(
                    f"Private example {index} of split '{self.split_name}' was read before DP "
                    f"accounting was activated. Non-private stages may only use public data."
                )
        if self.first_private_read_event is None:
            self.first_private_read_event = self._event("first_private_read", index=int(index))
            self.accountant_steps_at_first_private_read = int(self._current_acct_steps)
        self.n_private_reads += 1

    def on_logical_step(self, step: int, accountant_steps: Optional[int]) -> None:
        """Call once per completed logical optimizer step (after optimizer.step())."""
        self.logical_steps = int(step)
        if accountant_steps is not None:
            self._current_acct_steps = int(accountant_steps)
            if self.require_accounting and int(accountant_steps) != int(step):
                self.step_mismatches.append({"step": int(step), "accountant_steps": int(accountant_steps)})

    # ------------------------------------------------------------------ summary
    def finalize(self, accountant_steps: Optional[int]) -> Dict[str, Any]:
        ordering_ok = (
            not self.require_accounting
            or (
                self.accounting_activated_event is not None
                and self.first_private_read_event is not None
                and self.accounting_activated_event < self.first_private_read_event
                and self.n_private_reads_before_accounting == 0
                and self.accountant_steps_at_first_private_read == 0
                and not self.metadata_window_violations
            )
        )
        steps_ok = (
            not self.require_accounting
            or (accountant_steps is not None and int(accountant_steps) == self.logical_steps and not self.step_mismatches)
        )
        self.finalized = {
            "require_accounting": self.require_accounting,
            "accounting_activated_event": self.accounting_activated_event,
            "first_private_read_event": self.first_private_read_event,
            "accountant_steps_at_activation": self.accountant_steps_at_activation,
            "accountant_steps_at_first_private_read": self.accountant_steps_at_first_private_read,
            "n_private_reads": self.n_private_reads,
            "n_private_reads_before_accounting": self.n_private_reads_before_accounting,
            "logical_steps": self.logical_steps,
            "accountant_total_steps": None if accountant_steps is None else int(accountant_steps),
            "step_mismatches": self.step_mismatches[:10],
            "public_warmup": self.public_warmup,
            "metadata_reads": dict(self.metadata_reads),
            "metadata_window_violations": list(self.metadata_window_violations),
            "ordering_ok": bool(ordering_ok),
            "steps_match_ok": bool(steps_ok),
            "events_head": self.events[:6],
        }
        return self.finalized


class GuardedDataset:
    """Wraps a map-style dataset; every __getitem__ is reported to the ledger.

    Deliberately not a torch Dataset subclass so this module stays torch-free;
    DataLoader only needs __len__ and __getitem__.
    """

    def __init__(self, base, ledger: PrivacyLedger):
        self.base = base
        self.ledger = ledger

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        self.ledger.on_private_read(int(idx))
        return self.base[idx]


def accountant_total_steps(accountant) -> Optional[int]:
    """Total steps recorded by an Opacus accountant (RDP/PRV/GDP keep `history`)."""
    hist = getattr(accountant, "history", None)
    if hist is None:
        return None
    total = 0
    for entry in hist:
        # Opacus history entries are (noise_multiplier, sample_rate, num_steps)
        total += int(entry[-1])
    return total
