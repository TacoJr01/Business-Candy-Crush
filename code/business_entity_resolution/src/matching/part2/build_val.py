"""Build the validation feature matrix + labels from Part-1 outputs.

Run from code/business_entity_resolution/src/matching:
    python -m part2.build_val [--root WORKSPACE_ROOT] [--limit N]

Inputs (Part 1 artefacts + dataset):
    work/val/candidates_scored.tsv   dataset/train/train_source1.tsv
    dataset/train/train_source2.tsv  dataset/train/train_source3.tsv
    dataset/train/train_ground_truth.tsv

Outputs under work/part2/ (the <tag> suffix records the exact entity/candidate
counts so a smoke run can never poison the cached full run):
    val_pairs/       cached parsed pair arrays
    stores/val_s1_<tag>, stores/val_cand_<tag>   cached record stores
    val_X_<tag>.npy  float32 (n_pairs, n_features) matrix
    val_meta_<tag>.npz  y, s_idx (row per pair), row_off, s1_keys, gt_count, countries
    val_tag.txt      the newest tag (picked up by part2.train_model)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from . import feats, scores, scoresrc, store


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[5]))
    ap.add_argument("--limit", type=int, default=0, help="restrict to first N S1 rows (smoke test)")
    a = ap.parse_args(argv)
    root = Path(a.root)
    t0 = time.time()

    # 1. parse scored candidates (cached)
    A = scoresrc.parse_scored(root / "work/val/candidates_scored.tsv", root / "work/part2/val_pairs")
    if a.limit and a.limit < len(A["s1_keys"]):
        cut = int(A["row_off"][a.limit])
        for k in ("total", "g0", "g1", "g2", "g3", "rank", "c_keys"):
            A[k] = A[k][:cut]
        A["row_off"] = A["row_off"][:a.limit + 1]
        A["s1_keys"] = A["s1_keys"][:a.limit]
    n_pairs = len(A["c_keys"])
    n_rows = len(A["s1_keys"])
    print(f"val: {n_rows} S1 rows, {n_pairs} pairs", flush=True)

    # 2. record stores (cached): S1 side + candidate side, filtered to appearing ids
    keep_s1 = set(map(int, np.unique(A["s1_keys"])))
    keep_cand = set(map(int, np.unique(A["c_keys"])))
    tag = f"{len(keep_s1)}x{len(keep_cand)}"
    tr = root / "dataset/train"
    s1p = (tr / "train_source1.tsv").resolve()
    cpaths = [(tr / "train_source2.tsv").resolve(), (tr / "train_source3.tsv").resolve()]
    sst = store.build([s1p], root / f"work/part2/stores/val_s1_{tag}",
                      keep_by_path={str(s1p): keep_s1})
    cst = store.build(cpaths, root / f"work/part2/stores/val_cand_{tag}",
                      keep_by_path={str(cpaths[0]): keep_cand, str(cpaths[1]): keep_cand})

    # 3. resolve pair -> row / candidate indices
    s_row = sst.lookup(A["s1_keys"])
    assert (s_row >= 0).all(), "S1 id missing from store"
    s_idx = np.repeat(s_row, np.diff(A["row_off"])).astype(np.int32)
    c_idx = cst.lookup(A["c_keys"].astype(np.uint32)).astype(np.int32)
    assert (c_idx >= 0).all(), "candidate id missing from store"

    # 4. labels from ground truth
    gt = scores.load_gt(tr / "train_ground_truth.tsv", keep=set(map(int, A["s1_keys"])))
    gt_count_file = np.array([len(gt.get(int(k), set())) for k in A["s1_keys"]], dtype=np.float64)
    codes = ((np.repeat(A["s1_keys"], np.diff(A["row_off"])).astype(np.uint64) << 32)
             | A["c_keys"].astype(np.uint64))
    gtc = np.fromiter(((np.uint64(k) << 32) | np.uint64(c) for k, s in gt.items() for c in s),
                      dtype=np.uint64)
    y = np.isin(codes, np.sort(gtc)).astype(np.uint8)
    del codes
    print(f"labels: {int(y.sum())} positive pairs of {n_pairs} ({time.time()-t0:.0f}s)", flush=True)

    # 5. features
    ctx = feats.context(s_idx, c_idx, A["total"], sst.n, cst.n)
    sd = str(root / f"work/part2/stores/val_s1_{tag}")
    cd = str(root / f"work/part2/stores/val_cand_{tag}")
    fz = feats.fuzz_features(sd, cd, s_idx, c_idx, A["total"])
    feats.build_matrix(sst, cst, s_idx, c_idx,
                       {k: A[k] for k in ("total", "g0", "g1", "g2", "g3", "rank")},
                       ctx, fz, root / f"work/part2/val_X_{tag}.npy")

    # 6. metadata — row-aligned arrays are stored in STORE index order (what s_idx uses)
    ctry = scores.countries_of(s1p, A["s1_keys"].tolist())
    countries_file = np.array([ctry.get(int(k), "?") for k in A["s1_keys"]])
    n_st = sst.n
    gt_count = np.zeros(n_st)
    gt_count[s_row] = gt_count_file
    countries = np.empty(n_st, dtype=countries_file.dtype)
    countries[s_row] = countries_file
    s1_keys_st = np.zeros(n_st, np.uint32)
    s1_keys_st[s_row] = A["s1_keys"]
    np.savez_compressed(root / f"work/part2/val_meta_{tag}.npz", y=y, s_idx=s_idx,
                        total=A["total"], row_off=A["row_off"], s1_keys=s1_keys_st,
                        gt_count=gt_count, countries=countries)
    print(f"done: tag={tag} ({time.time()-t0:.0f}s)", flush=True)
    (root / "work/part2/val_tag.txt").write_text(tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
