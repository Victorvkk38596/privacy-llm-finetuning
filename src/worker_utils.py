"""Waiting for a training worker without hanging (pure Python, no torch).

multiprocessing.Queue.get() with no timeout blocks forever if the worker dies (OOM-kill, segfault,
os._exit) before putting its result. wait_for_result polls the queue and the process instead.
"""
import queue as _queue
import time


class WorkerFailure(RuntimeError):
    pass


def wait_for_result(proc, q, role: str, poll_sec: float = 5.0, timeout_sec: float = None,
                    grace_sec: float = 2.0):
    """Returns the worker's queued message, or raises WorkerFailure if the worker exits without one
    (or exceeds timeout_sec, in which case it is terminated)."""
    t0 = time.time()
    while True:
        try:
            return q.get(timeout=poll_sec)
        except _queue.Empty:
            pass
        if not proc.is_alive():
            # the result may have been flushed just before exit
            try:
                return q.get(timeout=grace_sec)
            except _queue.Empty:
                proc.join(timeout=1)
                raise WorkerFailure(f"worker {role} exited with code {proc.exitcode} before returning a result "
                                    f"(e.g. out-of-memory kill or crash)")
        if timeout_sec is not None and time.time() - t0 > timeout_sec:
            proc.terminate()
            proc.join(timeout=10)
            raise WorkerFailure(f"worker {role} exceeded timeout of {timeout_sec:.0f}s and was terminated")
