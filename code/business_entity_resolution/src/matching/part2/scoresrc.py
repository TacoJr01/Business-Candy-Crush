"""Parse Part-1 `candidates_scored.tsv` files into pair arrays.

Format: `s1_id \\t cand:total:g0:g1:g2:g3,cand:...,...` per row (empty when no candidates).
Produces (per file, cached under work/part2/<tag>/):

  s1_keys   u32 (n_s1)          entity key of every row (source order preserved)
  row_off   u32 (n_s1+1)        pair index range per row
  c_keys    u32 (n_pairs)       candidate key
  total     f32, g0..g3 f32     blocking score + per-group contributions
  rank      u16                 position in the row's list (lists are score-sorted)
"""

import mmap
import os
import time
from array import array
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import resources


def parse_range(path, start, end, out_dir, shard_i):
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    pos = start
    hi = min(end, len(mm))
    if start > 0:
        nl0 = mm.find(b"\n", start - 1)
        pos = len(mm) if nl0 < 0 else nl0 + 1
    else:
        pos = mm.find(b"\n") + 1  # skip header line
    s1_keys = array("I")
    row_off = array("I", [0]) if start == 0 else array("I")
    c_keys = array("I")
    total = array("f")
    g0 = array("f"); g1 = array("f"); g2 = array("f"); g3 = array("f")
    while pos < hi:
        nl = mm.find(b"\n", pos)
        if nl < 0:
            nl = len(mm)
        line = mm[pos:nl]
        pos = nl + 1
        if not line:
            continue
        t = line.find(b"\t")
        idb = line[:t].strip()
        s1_keys.append(int(idb[3:]) * 4 + (idb[1] - 48))
        if t + 1 >= len(line):
            row_off.append(len(c_keys))
            continue
        for c in line[t + 1:].split(b","):
            i = c.find(b":")
            idb = c[:i]
            c_keys.append(int(idb[3:]) * 4 + (idb[1] - 48))
            fs = c[i + 1:].split(b":")
            total.append(float(fs[0]))
            g0.append(float(fs[1])); g1.append(float(fs[2]))
            g2.append(float(fs[3])); g3.append(float(fs[4]))
        row_off.append(len(c_keys))
    mm.close()
    np.savez(out_dir / f"shard_{shard_i:02d}.npz",
             s1_keys=s1_keys, row_off=row_off, c_keys=c_keys, total=total,
             g0=g0, g1=g1, g2=g2, g3=g3)


def parse_scored(path, out_dir, workers=None):
    """Parse a scored file (cached as .npz arrays in out_dir). Returns dict of arrays."""
    out_dir = Path(out_dir)
    done = out_dir / "done.npz"
    if done.exists():
        z = np.load(done)
        return {k: z[k] for k in ("s1_keys", "row_off", "c_keys", "total", "g0", "g1", "g2", "g3", "rank")}
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("shard_*.npz"):
        stale.unlink()
    size = Path(path).stat().st_size
    # Capped at ~80% of CPUs by default (see part2.resources).
    nw = resources.resolve_workers(workers)
    nw = max(1, min(nw, size // 4_000_000 + 1))  # avoid a pile of empty shards on small files
    bounds = [size * i // nw for i in range(nw + 1)]
    with ProcessPoolExecutor(max_workers=nw) as ex:
        list(ex.map(parse_range, [path] * nw, bounds[:-1], bounds[1:], [out_dir] * nw, range(nw)))
    s1_keys = np.concatenate([np.load(out_dir / f"shard_{i:02d}.npz")["s1_keys"] for i in range(nw)]).astype(np.uint32)
    c_keys = np.concatenate([np.load(out_dir / f"shard_{i:02d}.npz")["c_keys"] for i in range(nw)]).astype(np.uint32)
    A = {}
    for key in ("total", "g0", "g1", "g2", "g3"):
        A[key] = np.concatenate([np.load(out_dir / f"shard_{i:02d}.npz")[key] for i in range(nw)]).astype(np.float32)
    # row offsets: concatenate, fixing each shard's offsets by the running pair total
    offs = [np.load(out_dir / f"shard_{i:02d}.npz")["row_off"] for i in range(nw)]
    base, row_off = 0, []
    for o in offs:
        o = np.asarray(o, dtype=np.uint32)
        if len(o):
            row_off.append(o + base)
            base = int(o[-1]) + base  # global pair count through this shard
    row_off = np.concatenate(row_off).astype(np.uint32)
    rank = (np.arange(len(c_keys), dtype=np.uint32) - np.repeat(row_off[:-1], np.diff(row_off)))
    A.update(s1_keys=s1_keys, row_off=row_off, c_keys=c_keys,
             rank=rank.astype(np.uint16))
    np.savez(done, **A)
    for i in range(nw):
        os.remove(out_dir / f"shard_{i:02d}.npz")
    print(f"  parsed {path}: {len(s1_keys)} S1 rows, {len(c_keys)} pairs ({time.time()-t0:.0f}s)", flush=True)
    return A
