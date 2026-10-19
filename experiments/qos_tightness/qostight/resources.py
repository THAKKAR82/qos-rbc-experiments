"""Memory gate mirroring the repository's: >= 4 GiB free to start a stage, >= 2 GiB while running,
process RSS <= 4 GiB. Peak RSS is absolute process RSS sampled every 50 ms."""
import threading
import time
from contextlib import contextmanager

import psutil

GIB = 2**30


def gate_ok(available, rss, before_stage=False):
    return available >= (4 if before_stage else 2) * GIB and rss <= 4 * GIB


@contextmanager
def measured():
    proc = psutil.Process()
    state = {"peak_rss_bytes": proc.memory_info().rss, "gate_violation": None}
    stop = threading.Event()
    def sample():
        while not stop.is_set():
            rss, avail = proc.memory_info().rss, psutil.virtual_memory().available
            state["peak_rss_bytes"] = max(state["peak_rss_bytes"], rss)
            if not gate_ok(avail, rss) and state["gate_violation"] is None:
                state["gate_violation"] = f"available={avail}, rss={rss}"
                import _thread
                _thread.interrupt_main()
            stop.wait(0.05)
    t = threading.Thread(target=sample, daemon=True)
    t.start()
    start = time.perf_counter()
    try:
        yield state
    finally:
        state["seconds"] = time.perf_counter() - start
        stop.set(); t.join()
