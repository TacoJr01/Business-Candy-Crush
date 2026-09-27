"""Analyse how the model's decisions distribute over blocking-score bands on the
validation holdout, to decide whether the bottom of candidate_pairs.tsv can be
pruned without hurting F0.5 (smaller candidate sets are ranked better in the audit).

    python -m part2.prune_analysis
"""

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from . import resources, scores, train_model


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[5]))
    a = ap.parse_args(argv)
    wp = Path(a.root) / "work/part2"
    tag = (wp / "val_tag.txt").read_text().strip()
    meta = json.loads((wp / "model_meta.json").read_text())
    tau = meta["tau"]
    t0 = time.time()

    M = np.load(wp / f"val_meta_{tag}.npz")
    y, s_idx, total, gt_count = M["y"], M["s_idx"], M["total"], M["gt_count"]
    s1_keys = M["s1_keys"]
    model = lgb.Booster(model_file=str(wp / "model.txt"))
    folds = train_model.fold_of(s1_keys)
    cal_rows = np.flatnonzero(folds == 4)
    mask = folds[s_idx] == 4
    X = np.load(wp / f"val_X_{tag}.npy", mmap_mode="r")
    probs = model.predict(X[mask], num_threads=resources.lgb_threads())
    del X
    print(f"{int(mask.sum())} calib pairs; tau={tau:.2f} (model best {meta['f05']:.4f})", flush=True)

    above = probs >= tau
    print("band | pairs | true pairs | pairs kept | P(pred|band) | rows affected")
    edges = [0, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 10]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (total[mask] >= lo) & (total[mask] < hi)
        if not m.any():
            continue
        print(f"  [{lo:.1f},{hi:.1f}) {int(m.sum()):8d} {int(y[mask][m].sum()):11d} "
              f"{int(above[m].sum()):10d} {above[m].mean():12.4f}")
    print("\ncutoff sweep: drop ALL candidates with blocking total < X (they can never be matches)")
    base_f, _, _ = scores.f05_at(tau, probs, y[mask], s_idx[mask], gt_count, rows=cal_rows)
    # recompute with preds restricted: set above=False below cutoff
    for X_ in [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]:
        keep = above & (total[mask] >= X_)
        probs2 = np.where(keep, np.float32(1.0), np.float32(0.0))
        f, p, r = scores.f05_at(0.5, probs2, y[mask], s_idx[mask], gt_count, rows=cal_rows)
        cand_frac = (total[mask] >= X_).mean()
        print(f"  cutoff {X_:.1f}: candidates kept {cand_frac:.3f}  F0.5 {f:.4f} (delta {f - base_f:+.4f})")
    print(f"done ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
