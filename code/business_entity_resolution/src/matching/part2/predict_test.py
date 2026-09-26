"""Score all test candidates with the trained matcher and write the submission.

    python -m part2.predict_test [--root WORKSPACE_ROOT] [--tau T] [--cand-min-total X]

Reads work/test_candidates_scored.tsv (Part 1 output), the cached test record
stores and work/part2/model.txt, and writes:

    output/matching_results.tsv    candidates with model prob >= tau (per row,
                                   exact ids copied from the scored file)
    output/candidate_pairs.tsv     unchanged unless --cand-min-total X is given,
                                   in which case pairs with blocking total < X
                                   are dropped (they are never shown to the model)

Also dumps work/part2/test_probs.bin (float16, pair order) for post-analysis.
"""

import argparse
import json
import mmap
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from . import feats, scoresrc, store

CHUNK = 250_000


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[5]))
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--cand-min-total", type=float, default=0.0)
    a = ap.parse_args(argv)
    root = Path(a.root)
    wp = root / "work/part2"
    meta = json.loads((wp / "model_meta.json").read_text())
    tau = a.tau if a.tau is not None else meta["tau"]
    t0 = time.time()

    # 1. parsed pairs (cached)
    A = scoresrc.parse_scored(root / "work/test_candidates_scored.tsv", wp / "test_pairs")
    n_pairs, n_rows = len(A["c_keys"]), len(A["s1_keys"])
    print(f"test: {n_rows} S1 rows, {n_pairs} pairs, tau={tau:.2f}", flush=True)

    # 2. record stores (cached)
    te = root / "dataset/test"
    s1p = (te / "test_source1.tsv").resolve()
    cpaths = [(te / "test_source2.tsv").resolve(), (te / "test_source3.tsv").resolve()]
    sst = store.build([s1p], wp / "stores/test_s1")
    cst = store.build(cpaths, wp / "stores/test_cand")

    # 3. indices + context + fuzz
    s_row = sst.lookup(A["s1_keys"]).astype(np.int32)
    assert (s_row >= 0).all()
    s_idx = np.repeat(s_row, np.diff(A["row_off"])).astype(np.int32)
    c_idx = cst.lookup(A["c_keys"]).astype(np.int32)
    assert (c_idx >= 0).all()
    ctx = feats.context(s_idx, c_idx, A["total"], sst.n, cst.n)
    fz = feats.fuzz_features(str(wp / "stores/test_s1"), str(wp / "stores/test_cand"),
                             s_idx, c_idx, A["total"])

    # 4. chunked inference
    model = lgb.Booster(model_file=str(wp / "model.txt"))
    feats.assert_features_match(model)
    sc_all = {k: A[k] for k in ("total", "g0", "g1", "g2", "g3", "rank")}
    probs = np.lib.format.open_memmap(wp / "test_probs.bin", dtype=np.float16,
                                      shape=(n_pairs,), mode="w+")
    Xc = np.empty((min(CHUNK, n_pairs), len(feats.FEATURES)), np.float32)
    for lo in range(0, n_pairs, CHUNK):
        hi = min(lo + CHUNK, n_pairs)
        feats.fill_matrix(sst, cst, s_idx, c_idx, sc_all, ctx, fz, Xc[:hi - lo], lo, hi)
        probs[lo:hi] = model.predict(Xc[:hi - lo], num_threads=0).astype(np.float16)
        if lo % (CHUNK * 20) == 0:
            print(f"  inferred {lo}/{n_pairs} ({time.time()-t0:.0f}s)", flush=True)
    probs.flush()
    above = np.asarray(probs[:] >= tau)
    n_pred = int(above.sum())
    per_row = np.bincount(s_idx[above], minlength=sst.n)
    n_empty = int((per_row[s_row] == 0).sum())
    print(f"predicts {n_pred} pairs; {n_empty}/{n_rows} S1 singletons ({time.time()-t0:.0f}s)", flush=True)
    del s_idx, c_idx, ctx, fz

    # 5. write matching_results.tsv (+ optional pruned candidate_pairs.tsv)
    src = root / "work/test_candidates_scored.tsv"
    out_m = root / "output/matching_results.tsv"
    out_m.parent.mkdir(exist_ok=True)
    out_c = root / "output/candidate_pairs.tsv" if a.cand_min_total > 0 else None
    if out_c is not None:
        tmp = out_c.with_suffix(".tmp")
        tc = open(tmp, "wb")
    fm = open(out_m, "wb")
    fm.write(b"source1_entity_id\tmatched_entity_ids\n")
    if out_c is not None:
        tc.write(b"source1_entity_id\tcandidate_entity_ids\n")
    ptr = 0
    f = open(src, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    pos = mm.find(b"\n") + 1  # skip header line
    while pos < len(mm):
        nl = mm.find(b"\n", pos)
        if nl < 0:
            nl = len(mm)
        line = mm[pos:nl]
        pos = nl + 1
        t = line.find(b"\t")
        s1id = line[:t]
        kept, cands = bytearray(), []
        if t + 1 < len(line):
            for c in line[t + 1:].split(b","):
                i = c.find(b":")
                if above[ptr]:
                    cands.append(memoryview(c)[:i])
                if out_c is not None:
                    j = c.find(b":", i + 1)
                    if float(c[i + 1:j]) >= a.cand_min_total:
                        kept += memoryview(c)[:i]
                        kept += b","
                ptr += 1
        fm.write(s1id + b"\t" + b",".join(cands) + b"\n")
        if out_c is not None:
            tc.write(s1id + b"\t" + kept.rstrip(b",") + b"\n")
    fm.close()
    mm.close()
    f.close()
    if out_c is not None:
        tc.close()
        out_c.unlink(missing_ok=True)
        tmp.rename(out_c)
    assert ptr == n_pairs, (ptr, n_pairs)

    # 6. validate
    r = subprocess.run([sys.executable, str(root / "utils/validate_submission.py"),
                        "--matching", str(out_m), "--candidate",
                        str(root / "output/candidate_pairs.tsv"),
                        "--test-dir", str(root / "dataset/test")],
                       capture_output=True, text=True)
    print(r.stdout[-4000:])
    if r.returncode != 0:
        print(r.stderr[-2000:])
    print(f"validator exit={r.returncode}  total {time.time()-t0:.0f}s", flush=True)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
