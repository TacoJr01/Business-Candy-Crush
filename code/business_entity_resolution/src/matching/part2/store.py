"""RecordStore: per-record feature primitives built from a source TSV.

Streams `*_sourceN.tsv` files (optionally filtered to an id set), normalises
name/address, hashes tokens, and stores everything as flat numpy arrays plus an
on-disk cache (a directory of .npy files + blobs), so reruns are instant. All
arrays and blobs are memory-mapped: parallel workers share OS pages instead of
keeping private copies.

Record key: ids look like "S2-637340732" -> key = num * 4 + src (fits u32).
"""

import hashlib
import mmap
import os
import time
from array import array
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import norm
from . import resources

_ARRS = ("keys", "ntok1", "ntok2", "tok1", "tok2", "nums", "pins", "hname", "haddr",
         "nlen", "alen", "noff", "aoff")


def id_key(id_bytes: bytes) -> int:
    src = id_bytes[1] - 48  # 'S1' / 'S2' / 'S3'
    num = int(id_bytes[3:])
    return num * 4 + src


def parse_source_lines(path, start, end, keep, out_dir, shard_i):
    """Worker: parse one byte range of a source TSV into a compact .npz shard."""
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    hi = min(end, len(mm))
    pos = start
    if start > 0:
        nl0 = mm.find(b"\n", start - 1)
        pos = len(mm) if nl0 < 0 else nl0 + 1
    else:
        pos = mm.find(b"\n") + 1  # skip header line
    keys = array("I")
    ntok1 = array("B")
    ntok2 = array("B")
    t1 = array("I")
    t1off = array("I", [0])
    t2 = array("I")
    t2off = array("I", [0])
    nums = array("I")
    pins = array("I")
    hname = array("I")
    haddr = array("I")
    nlen = array("H")
    alen = array("H")
    nblob = bytearray()
    noff = array("Q", [0])
    ablob = bytearray()
    aoff = array("Q", [0])
    h4 = lambda s: int.from_bytes(hashlib.blake2b(s.encode("utf-8", "ignore"), digest_size=4).digest(), "little") & 0xFFFFFFFF
    capn, capa = norm.CAP_NAME, norm.CAP_ADDR
    while pos < hi:
        nl = mm.find(b"\n", pos)
        if nl < 0:
            nl = len(mm)
        line = mm[pos:nl]
        pos = nl + 1
        c1 = line.find(b"\t")
        if c1 < 0:
            continue
        c2 = line.find(b"\t", c1 + 1)
        c3 = line.find(b"\t", c2 + 1)
        key = id_key(line[:c1].strip())
        if keep is not None and key not in keep:
            continue
        name = line[c1 + 1:c2].decode("utf-8", "ignore")
        addr = line[c2 + 1:c3 if c3 > 0 else len(line)].decode("utf-8", "ignore")
        keys.append(key)
        nn = norm.normalize(name)
        an = norm.normalize(addr)
        v1, cnt1 = norm.hash_words(norm.name_tokens(name), capn)
        at = norm.addr_tokens(addr)
        v2, cnt2 = norm.hash_words(at, capa)
        ntok1.append(min(cnt1, 255))
        ntok2.append(min(cnt2, 255))
        t1.fromlist(v1)
        t1off.append(len(t1))
        t2.fromlist(v2)
        t2off.append(len(t2))
        pn, pin = 0, 0
        for w in at:
            if w.isdigit() and len(w) <= 10:
                if len(w) >= 5 and not pin:
                    pin = int(w) & 0xFFFFFFFF
                if pn < 4:
                    nums.append(int.from_bytes(hashlib.blake2b(w.encode(), digest_size=4).digest(), "little") | 1)
                    pn += 1
        while pn < 4:
            nums.append(0)
            pn += 1
        pins.append(pin)
        hname.append(h4(nn))
        haddr.append(h4(an))
        nlen.append(min(len(nn), 65535))
        alen.append(min(len(an), 65535))
        nblob += nn.encode("utf-8", "ignore") + b"\x00"
        noff.append(len(nblob))
        ablob += an.encode("utf-8", "ignore") + b"\x00"
        aoff.append(len(ablob))
    mm.close()
    np.savez(out_dir / f"src_shard_{shard_i:03d}.npz",
             keys=keys, ntok1=ntok1, ntok2=ntok2, t1=t1, t1off=t1off, t2=t2, t2off=t2off,
             nums=nums, pins=pins, hname=hname, haddr=haddr, nlen=nlen, alen=alen,
             noff=noff, aoff=aoff)
    (out_dir / f"src_shard_{shard_i:03d}.nblob").write_bytes(bytes(nblob))
    (out_dir / f"src_shard_{shard_i:03d}.ablob").write_bytes(bytes(ablob))
    return len(keys)


class Store:
    """Flat arrays + blobs for one record table, addressable by row index."""

    def __init__(self, d):
        d = Path(d)
        self.d = d
        for a in _ARRS:
            setattr(self, a, np.load(d / f"{a}.npy", mmap_mode="r"))
        self.n = len(self.keys)
        self._blobs = {}

    def _blob(self, which):
        if which not in self._blobs:
            f = open(self.d / f"{which}blob.bin", "rb")
            self._blobs[which] = (f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
        return self._blobs[which][1]

    def lookup(self, keys):
        """Row index for each query key, or -1 when absent."""
        i = np.clip(np.searchsorted(self.keys, keys), 0, self.n - 1)
        return np.where(self.keys[i] == keys, i, -1).astype(np.int64)

    def _strs(self, off, blob, idx):
        b = memoryview(blob)
        return [b[off[i]:off[i + 1] - 1].tobytes().decode("utf-8") for i in idx]

    def names(self, idx):
        return self._strs(self.noff, self._blob("n"), idx)

    def addrs(self, idx):
        return self._strs(self.aoff, self._blob("a"), idx)


def build(source_paths, out_dir, keep_by_path=None, workers=None):
    """Build (or load cached) Store from one or more source TSVs.

    keep_by_path: optional {resolved path str: set-of-u32-keys} filter per file.
    Shards are parsed in parallel, then merged into key-sorted .npy files so the
    Store can resolve entity keys with searchsorted.
    """
    out_dir = Path(out_dir)
    complete = all((out_dir / f"{a}.npy").exists() for a in _ARRS) and \
        (out_dir / "nblob.bin").exists() and (out_dir / "ablob.bin").exists()
    if complete:
        return Store(out_dir)
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("src_shard_*"):  # leftovers from an interrupted run
        stale.unlink()
    if not complete:  # half-written assembly; start clean
        for stale in list(out_dir.glob("*.npy")) + list(out_dir.glob("*blob.bin")):
            stale.unlink()
    nw = resources.resolve_workers(workers)
    si = 0
    for p in source_paths:
        p = Path(p).resolve()
        size = p.stat().st_size
        pw = max(1, min(nw, size // 4_000_000 + 1))
        keep = (keep_by_path or {}).get(str(p))
        keep_f = frozenset(keep) if keep is not None else None
        bounds = [size * i // pw for i in range(pw + 1)]
        args = [(str(p), bounds[i], bounds[i + 1], keep_f, out_dir, si + i) for i in range(pw)]
        si += pw
        with ProcessPoolExecutor(max_workers=pw) as ex:
            kept = [f.result() for f in [ex.submit(parse_source_lines, *a) for a in args]]
        print(f"  parsed {p.name}: {sum(kept)} rows ({time.time()-t0:.0f}s)", flush=True)

    # ---- merge shards (file order, then shard index) into key-sorted arrays ----
    shard_files = sorted(out_dir.glob("src_shard_*.npz"), key=lambda f: int(f.stem.rsplit("_", 1)[-1]))
    flat = {k: [] for k in ("keys", "ntok1", "ntok2", "pins", "hname", "haddr", "nlen",
                            "alen", "nums", "t1", "t1off", "t2", "t2off", "noff", "aoff")}
    blobs = {"n": [], "a": []}
    for sf in shard_files:
        z = np.load(sf)
        for k in flat:
            flat[k].append(z[k])
        blobs["n"].append((sf.parent / (sf.stem + ".nblob")).read_bytes())
        blobs["a"].append((sf.parent / (sf.stem + ".ablob")).read_bytes())
        z.close()

    def cat_off(parts, dt):
        """Concatenate per-shard offset arrays (each starts at 0) into global offsets."""
        acc, base = [], 0
        for x in parts:
            x = np.asarray(x, dt)
            if not len(x):
                continue
            acc.append(x + base if not acc else x[1:] + base)
            base += int(x[-1])
        return np.concatenate(acc) if acc else np.zeros(1, dt)

    keys_all = np.concatenate(flat["keys"])
    n = len(keys_all)
    order = np.argsort(keys_all, kind="stable")
    for name_, dt in [("keys", np.uint32), ("ntok1", np.uint8), ("ntok2", np.uint8),
                      ("pins", np.uint32), ("hname", np.uint32), ("haddr", np.uint32),
                      ("nlen", np.uint16), ("alen", np.uint16)]:
        arr = np.concatenate(flat[name_]).astype(dt)
        if name_ == "ntok1":
            arr = np.minimum(arr, norm.CAP_NAME)
        if name_ == "ntok2":
            arr = np.minimum(arr, norm.CAP_ADDR)
        np.save(out_dir / f"{name_}.npy", arr[order])
    np.save(out_dir / "nums.npy", np.concatenate(flat["nums"]).astype(np.uint32).reshape(n, 4)[order])

    # token matrices, filled directly in sorted row order
    for ck, arrk, cap in (("t1", "tok1", norm.CAP_NAME), ("t2", "tok2", norm.CAP_ADDR)):
        fa = np.concatenate(flat[ck])
        offs = cat_off(flat[ck + "off"], np.int64)
        cnt = np.minimum(np.diff(offs), cap)
        t = np.zeros((n, cap), dtype=np.uint32)
        for i, r in enumerate(order):
            a, b = offs[r], offs[r] + cnt[r]
            t[i, :b - a] = fa[a:b]
        np.save(out_dir / f"{arrk}.npy", t)
        del fa, offs, cnt, t

    # blobs, rewritten in sorted row order
    for which in ("n", "a"):
        src = b"".join(blobs[which])
        offs = cat_off(flat[f"{which}off"], np.uint64)
        out = bytearray(len(src))
        newoffs = np.zeros(n + 1, np.uint64)
        pos = 0
        for i, r in enumerate(order):
            piece = src[offs[r]:offs[r + 1]]
            out[pos:pos + len(piece)] = piece
            pos += len(piece)
            newoffs[i + 1] = pos
        np.save(out_dir / f"{which}off.npy", newoffs)
        (out_dir / f"{which}blob.bin").write_bytes(bytes(out))
        del src, out

    for f in out_dir.glob("src_shard_*"):
        os.remove(f)
    print(f"  store cached to {out_dir} ({time.time()-t0:.0f}s)", flush=True)
    return Store(out_dir)
