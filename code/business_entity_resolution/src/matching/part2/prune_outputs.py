"""Rewrite output/candidate_pairs.tsv + output/matching_results.tsv with a
blocking-total cutoff applied to what the matcher "sees" (and therefore may match).

    python -m part2.prune_outputs [--cutoff 0.8]

Uses work/part2/test_probs.bin (written by predict_test) and streams
work/test_candidates_scored.tsv once. Matches and candidates both keep only
pairs with blocking total >= cutoff, so matching_results ⊂ candidate_pairs
stays exact and the candidate file is auditable as "what the model scored".
"""

import argparse
import json
import mmap
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CHUNK = 4_000_000


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[5]))
    ap.add_argument("--cutoff", type=float, default=0.8)
    a = ap.parse_args(argv)
    root = Path(a.root)
    wp = root / "work/part2"
    tau = json.loads((wp / "model_meta.json").read_text())["tau"]
    probs = np.load(wp / "test_probs.bin", mmap_mode="r")
    src = root / "work/test_candidates_scored.tsv"
    t0 = time.time()

    out_c = root / "output/candidate_pairs.tsv"
    out_m = root / "output/matching_results.tsv"
    tmp_c = out_c.with_suffix(".tmp")
    tmp_m = out_m.with_suffix(".tmp")
    tc, tm = open(tmp_c, "wb"), open(tmp_m, "wb")
    tc.write(b"source1_entity_id\tcandidate_entity_ids\n")
    tm.write(b"source1_entity_id\tmatched_entity_ids\n")

    n_pairs = len(probs)
    ptr = 0
    kept_c = kept_m = 0
    with open(src, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        pos = mm.find(b"\n") + 1
        while pos < len(mm):
            nl = mm.find(b"\n", pos)
            if nl < 0:
                nl = len(mm)
            line = mm[pos:nl]
            pos = nl + 1
            t = line.find(b"\t")
            s1id = line[:t]
            cands = bytearray()
            n_c = 0
            if t + 1 < len(line):
                toks = line[t + 1:].split(b",")
                k = len(toks)
                sc = np.empty(k, np.float32)
                for i, c in enumerate(toks):
                    j = c.find(b":")
                    sc[i] = float(c[j + 1:c.find(b":", j + 1)])
                sel = sc >= a.cutoff
                pb = np.asarray(probs[ptr:ptr + k])
                hit = sel & (pb >= tau)
                for i, c in enumerate(toks):
                    if sel[i]:
                        j = c.find(b":")
                        cands += c[:j] + b","
                        n_c += 1
                kept = [c[:c.find(b":")] for i, c in enumerate(toks) if hit[i]]
                tm.write(s1id + b"\t" + b",".join(kept) + b"\n")
                kept_m += len(kept)
                ptr += k
            tc.write(s1id + b"\t" + cands.rstrip(b",") + b"\n")
            kept_c += n_c
        assert ptr == n_pairs, (ptr, n_pairs)
    mm.close()
    tc.close()
    tm.close()
    tmp_c.replace(out_c)
    tmp_m.replace(out_m)
    print(f"cutoff={a.cutoff} tau={tau:.2f}: candidates {kept_c} ({kept_c/n_pairs:.1%} of {n_pairs}), "
          f"matches {kept_m} ({time.time()-t0:.0f}s)", flush=True)

    r = subprocess.run([sys.executable, str(root / "utils/validate_submission.py"),
                        "--matching", str(out_m), "--candidate", str(out_c),
                        "--test-dir", str(root / "dataset/test")], cwd=root)
    print(f"validator exit={r.returncode}", flush=True)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
