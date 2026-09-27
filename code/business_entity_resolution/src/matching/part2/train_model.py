"""Train the Part-2 LightGBM matcher and tune the F0.5 decision threshold.

    python -m part2.train_model [--root WORKSPACE_ROOT]

Data protocol (the val holdout entities are the only labelled candidates):
  * rows (S1 entities) are split by key hash % 5: folds 0-2 fit, 3 early-stop
    eval, 4 calibration. The threshold is chosen on calibration rows only, so
    the reported calib F0.5 is an honest estimate for the test set.
  * the final model is refit on fit+eval rows with the selected tree count.

Writes work/part2/model.txt + work/part2/train_report.txt + model_meta.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from . import embed, feats, resources, scores


def fold_of(keys):
    """Deterministic 5-fold split over u32 record keys (golden-ratio hash)."""
    return ((np.asarray(keys, dtype=np.uint64) * np.uint64(11400714819323198485)) >> 61) % 5


def macro_f05_report(tau, probs, y, s_idx, gt_count, countries, rows):
    """One summary line + per-country breakdown for a threshold on a row subset."""
    f, p, r = scores.f05_at(tau, probs, y, s_idx, gt_count, rows=rows)
    out = [f"tau={tau:.2f}  macro F0.5={f:.4f}  precision={p:.4f}  recall={r:.4f}"]
    for c in ("US", "India"):
        rr = rows[countries[rows] == c]
        if len(rr):
            fc, pc, rc = scores.f05_at(tau, probs, y, s_idx, gt_count, rows=rr)
            out.append(f"    {c:6s} ({len(rr)} S1): F0.5={fc:.4f}  P={pc:.4f}  R={rc:.4f}")
    return "\n".join(out)


def sweep_subset(probs, y, s_idx, gt_count, rows, taus):
    """Threshold sweep restricted to a subset of rows (remaps row indices)."""
    remap = np.searchsorted(rows, s_idx).astype(np.int64)
    return scores.best_threshold_fast(probs, y, remap, gt_count[rows], taus=taus)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[5]))
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "gpu"),
                    help="auto (default): PART2_DEVICE/USE_GPU env or cpu; "
                         "gpu is opt-in and runs uncapped, CPU work stays ≤80%%")
    ap.add_argument("--jobs", type=int, default=None,
                    help="explicit CPU thread/worker count (clamped to the 80%% cap "
                         "unless PART2_ALLOW_FULL=1)")
    a = ap.parse_args(argv)
    resources.apply_thread_env()
    device = resources.resolve_device(None if a.device == "auto" else a.device)
    threads = resources.lgb_threads(a.jobs)
    root = Path(a.root)
    wp = root / "work/part2"
    tag = (wp / "val_tag.txt").read_text().strip()
    t0 = time.time()

    M = np.load(wp / f"val_meta_{tag}.npz")
    y = M["y"].astype(np.int8)
    s_idx = M["s_idx"]
    gt_count = M["gt_count"]
    total = M["total"]
    countries = M["countries"]
    s1_keys = M["s1_keys"]
    X = np.load(wp / f"val_X_{tag}.npy", mmap_mode="r")

    folds = fold_of(s1_keys)
    pair_fold = folds[s_idx]
    fit = np.isin(pair_fold, (0, 1, 2))
    ev = pair_fold == 3
    cal = pair_fold == 4
    cal_rows = np.flatnonzero(folds == 4)
    print(f"pairs: fit={fit.sum()} eval={ev.sum()} cal={cal.sum()}", flush=True)

    params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=255,
                  min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, num_threads=threads, verbose=-1, seed=7,
                  **resources.lgb_device_params(device))
    print(f"device={device} num_threads={threads}", flush=True)
    dtr = lgb.Dataset(X[fit], label=y[fit], feature_name=feats.active_features(), free_raw_data=True)
    dev = lgb.Dataset(X[ev], label=y[ev], reference=dtr, free_raw_data=True)
    try:
        model = lgb.train(params, dtr, num_boost_round=4000, valid_sets=[dev],
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    except lgb.basic.LightGBMError as e:
        if device == "gpu":
            print(f"GPU training failed ({e}); falling back to CPU.", flush=True)
            params.update(resources.lgb_device_params("cpu"))
            device = "cpu"
            model = lgb.train(params, dtr, num_boost_round=4000, valid_sets=[dev],
                              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
        else:
            raise
    best_iter = int(model.best_iteration or 4000)
    print(f"best_iteration={best_iter}  ({time.time()-t0:.0f}s)", flush=True)

    # refit on fit+eval rows; calibration rows stay untouched
    both = fit | ev
    dall = lgb.Dataset(X[both], label=y[both], feature_name=feats.active_features())
    final = lgb.train(params, dall, num_boost_round=best_iter)
    final.save_model(str(wp / "model.txt"))

    pcal = final.predict(X[cal])
    tau, f05, table = sweep_subset(pcal, y[cal], s_idx[cal], gt_count, cal_rows,
                                   np.round(np.arange(0.02, 0.981, 0.01), 2))
    btau, bf05, _ = sweep_subset(total[cal], y[cal], s_idx[cal], gt_count, cal_rows,
                                 np.arange(0.5, 4.01, 0.02))

    gain = dict(zip(final.feature_name(), (final.feature_importance("gain") / max(final.feature_importance("gain").max(), 1)).round(3)))
    lines = [
        f"train on {int(both.sum())} pairs, best_iter={best_iter}",
        "top features (gain):",
        "\n".join(f"  {n:18s} {g}" for n, g in sorted(gain.items(), key=lambda x: -x[1])[:15]),
        f"== LightGBM (calibration rows, {len(cal_rows)} S1) ==",
        macro_f05_report(tau, pcal, y[cal], s_idx[cal], gt_count, countries, cal_rows),
        "",
        f"== naive blocking-score threshold (same rows): best={bf05:.4f} at tau={btau:.2f} ==",
        macro_f05_report(btau, total[cal], y[cal], s_idx[cal], gt_count, countries, cal_rows),
        "",
        "threshold sweep (tau, macro F0.5):",
        "  " + "  ".join(f"{t:.2f}:{f:.4f}" for t, f in table[::5]),
    ]
    report = "\n".join(lines)
    print(report, flush=True)
    (wp / "train_report.txt").write_text(report, encoding="utf-8")
    (wp / "model_meta.json").write_text(json.dumps(
        dict(tau=float(tau), f05=float(f05), best_iteration=best_iter,
             features=feats.active_features(), val_tag=tag, baseline_tau=float(btau),
             baseline_f05=float(bf05), device=device, num_threads=threads,
             embedded=embed.enabled()), indent=2))
    print(f"saved model + report ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
