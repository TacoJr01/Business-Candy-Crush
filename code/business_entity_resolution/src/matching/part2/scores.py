"""Macro F_0.5 scorer (exactly the challenge definition) + ground-truth loader."""

import numpy as np


def f05_entity(pred: set, true: set) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    p = tp / len(pred)
    r = tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(preds, trues):
    """preds/trues: aligned sequences of id sets. All entities count in the average."""
    return sum(f05_entity(p, t) for p, t in zip(preds, trues)) / max(len(trues), 1)


def load_gt(path, keep=None):
    """ground_truth.tsv -> {s1_key: set(cand_keys)}. keep: optional set of s1 keys."""
    out = {}
    with open(path, "rb") as f:
        f.readline()
        for line in f:
            a, _, b = line.partition(b"\t")
            a = a.strip()
            key = int(a[3:]) * 4 + (a[1] - 48)
            if keep is not None and key not in keep:
                continue
            b = b.strip().rstrip(b"\n")
            if b:
                s = set()
                for cid in b.split(b","):
                    s.add(int(cid[3:]) * 4 + (cid[1] - 48))
                out[key] = s
            else:
                out[key] = set()
    return out


def countries_of(path, keys):
    """Map record key -> country string (reporting only; never a model feature)."""
    want = set(map(int, keys))
    out = {}
    with open(path, "rb") as f:
        f.readline()
        for line in f:
            a, _, rest = line.partition(b"\t")
            k = int(a[3:]) * 4 + (a[1] - 48)
            if k in want:
                parts = rest.rstrip(b"\n").split(b"\t")
                out[k] = parts[-1].decode()
    return out


def best_threshold_fast(probs, y, s_idx, gt_count, taus=None):
    """Vectorised threshold sweep maximising macro F0.5.

    probs (n_pairs,) model probabilities; y (n_pairs,) 0/1 pair labels;
    s_idx (n_pairs,) row (S1) index per pair; gt_count (n_rows,) true #matches
    per row INCLUDING matches outside the candidate list (they cap recall).

    For a row: f0.5 = 1.25*TP / (NPred + 0.25*G); empty-pred rows score 1 if G==0.
    Returns (best_tau, best_f05, table of (tau, macro_f05)).
    """
    n_rows = len(gt_count)
    g_pos = gt_count > 0
    if taus is None:
        taus = np.round(np.arange(0.05, 0.951, 0.01), 2)
    table = []
    for tau in taus:
        above = probs >= tau
        npred = np.bincount(s_idx[above], minlength=n_rows).astype(np.float64)
        tp = np.bincount(s_idx[above & (y > 0)], minlength=n_rows).astype(np.float64)
        f = np.zeros(n_rows)
        f[g_pos] = 1.25 * tp[g_pos] / (npred[g_pos] + 0.25 * gt_count[g_pos])
        f[~g_pos] = np.where(npred[~g_pos] == 0, 1.0, 0.0)
        table.append((float(tau), float(f.mean())))
    best = max(table, key=lambda x: (x[1], x[0]))  # ties -> stricter (higher) tau
    return best[0], best[1], table


def f05_at(tau, probs, y, s_idx, gt_count, rows=None):
    """(macro_f05, pooled precision, pooled recall) at one threshold.

    rows: optional set of row indices to restrict both the average and P/R to.
    """
    n_rows = len(gt_count)
    above = probs >= tau
    npred = np.bincount(s_idx[above], minlength=n_rows).astype(np.float64)
    tp = np.bincount(s_idx[above & (y > 0)], minlength=n_rows).astype(np.float64)
    g = gt_count.astype(np.float64)
    f = np.zeros(n_rows)
    pos = g > 0
    f[pos] = 1.25 * tp[pos] / (npred[pos] + 0.25 * g[pos])
    f[~pos] = np.where(npred[~pos] == 0, 1.0, 0.0)
    if rows is None:
        keep = np.ones(n_rows, bool)
    else:
        keep = np.zeros(n_rows, bool)
        keep[rows] = True
    prec = tp[keep].sum() / max(npred[keep].sum(), 1)
    rec = tp[keep].sum() / max(g[keep].sum(), 1)
    return float(f[keep].mean()), float(prec), float(rec)
