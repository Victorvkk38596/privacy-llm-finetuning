"""Worker-failure handling: a training worker that exits without queuing a result must surface as a
failure (not hang), and the runner must turn missing records into visible failure records that are
retried on resume. Uses real spawned processes and SYNTHETIC records in temp dirs.

  python tests/test_worker_failure.py      (or: pytest tests/)
"""
import multiprocessing
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
from src.worker_utils import wait_for_result, WorkerFailure  # noqa: E402


def _die_without_result(q):
    os._exit(3)            # like an OOM kill: no exception handler runs, nothing is queued


def _return_result(q):
    q.put(("SUCCESS", {"val_accuracy": 80.0}))


def _sleep_forever(q):
    time.sleep(600)


def _spawn(target):
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=target, args=(q,))
    p.start()
    return p, q


def test_worker_exit_without_result_raises_quickly():
    p, q = _spawn(_die_without_result)
    t0 = time.time()
    try:
        wait_for_result(p, q, role="M_A", poll_sec=0.5, grace_sec=0.5)
        raise AssertionError("expected WorkerFailure")
    except WorkerFailure as e:
        assert "exited with code 3" in str(e), e
    assert time.time() - t0 < 30


def test_worker_result_returned():
    p, q = _spawn(_return_result)
    msg = wait_for_result(p, q, role="M_A", poll_sec=0.5)
    p.join()
    assert msg[0] == "SUCCESS" and msg[1]["val_accuracy"] == 80.0


def test_worker_timeout_terminates():
    p, q = _spawn(_sleep_forever)
    try:
        wait_for_result(p, q, role="M_B", poll_sec=0.5, timeout_sec=2)
        raise AssertionError("expected WorkerFailure")
    except WorkerFailure as e:
        assert "timeout" in str(e)
    assert not p.is_alive()


def test_runner_missing_records_become_failures_and_are_retried():
    import run_core_grid as rcg
    from grid_records import load_records, index_records, grid_pair_states
    from synthetic_records import make_record, write_records
    with tempfile.TemporaryDirectory() as td:
        rec_dir = os.path.join(td, "records")
        os.makedirs(rec_dir)
        # subprocess exited (even with code 0) but wrote no records
        missing = rcg.write_missing_failure_records(rec_dir, "dp_late3_e5", "none", 42, returncode=0)
        assert sorted(missing) == ["M_A", "M_B"]
        st = grid_pair_states(index_records(load_records(rec_dir)), "none")[("dp_late3_e5", 42)]
        assert st["status"] == "failed", st
        # only the missing role is filled in when one record exists
        write_records(rec_dir, [make_record("dp_mid3_e5", 42, "M_A")])
        assert rcg.write_missing_failure_records(rec_dir, "dp_mid3_e5", "none", 42, returncode=-9) == ["M_B"]
        idx = index_records(load_records(rec_dir))
        assert idx[("dp_mid3_e5__w-none", 42, "M_A")]["status"] == "ok"
        assert idx[("dp_mid3_e5__w-none", 42, "M_B")]["status"] == "failed"
        # retry: failed pair's records are archived (kept on disk, out of the statistics)
        moved = rcg.archive_pair_records(rec_dir, "dp_late3_e5__w-none", 42)
        assert len(moved) == 2 and all(os.path.exists(m) for m in moved)
        assert "superseded" in moved[0]
        st = grid_pair_states(index_records(load_records(rec_dir)), "none")[("dp_late3_e5", 42)]
        assert st["status"] == "pending"


def test_runner_resume_selects_not_ok_pairs():
    """Pending list used by run_core_grid main(): everything not 'ok' is rerun, including invalid."""
    from grid_records import load_records, index_records, grid_pair_states
    from synthetic_records import make_record, write_records
    with tempfile.TemporaryDirectory() as td:
        write_records(td, [make_record("dp_late3_e5", 42, r) for r in ("M_A", "M_B")]
                      + [make_record("dp_mid3_e5", 42, r, **{"privacy.accounted_epsilon": 6.0}) for r in ("M_A", "M_B")])
        states = grid_pair_states(index_records(load_records(td)), "none")
        pending = [k for k, v in states.items() if v["status"] != "ok"]
        assert ("dp_late3_e5", 42) not in pending
        assert ("dp_mid3_e5", 42) in pending
        assert len(states) == 27 and len(pending) == 26


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print(f"[PASS] {name}")
    print(f"all {n} worker-failure tests passed")
