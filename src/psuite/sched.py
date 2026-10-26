"""Work scheduling shared by the sampling / state / trace stages: one process per GPU (shard), resumable chunk files, datasets interleaved
in proportion to their size so that a run cut short by the time budget still has every dataset represented."""
import math
import time
from typing import List

from .bench import DS_ORDER


def schedule(problems: List[dict], shard: int, nshards: int, chunk: int, sub: str = ""):
    by = {}
    for p in problems:
        if p["id"] % nshards == shard:
            by.setdefault(p["ds"], []).append(p)
    items = []
    for ds, ps in by.items():
        nch = math.ceil(len(ps) / chunk)
        for k in range(nch):
            items.append(((k + 1) / nch, DS_ORDER.index(ds) if ds in DS_ORDER else 99, ds, k, ps[k * chunk:(k + 1) * chunk]))
    items.sort(key=lambda t: (t[0], t[1]))
    out = [(ds, k, batch) for _, _, ds, k, batch in items]
    if sub:                                   # 'w/W': this worker takes every W-th chunk of the shard (rescue: several GPUs finish one shard's leftovers)
        w, W = (int(x) for x in sub.split("/"))
        out = [o for i, o in enumerate(out) if i % W == w]
    return out


class Budget:
    def __init__(self, seconds: float = 0):
        self.t0, self.s = time.time(), seconds

    def spent(self) -> bool:
        return bool(self.s) and (time.time() - self.t0) > self.s
