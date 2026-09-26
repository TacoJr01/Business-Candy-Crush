"""Pairwise feature computation for the Part-2 matching model.

Features come from the two RecordStores (S1 side, candidate side) plus the Part-1
blocking score columns, so the model reuses the TF-IDF signals rather than
recomputing them. Everything is vectorised numpy and *chunked*, so the ~72M-pair
test pass runs in minutes within a 16 GB budget:

  * `context()` builds small per-row / per-candidate arrays over a whole file;
  * `pair_chunk()` materialises the feature dict for one chunk of pairs only;
  * `fuzz_features()` adds rapidfuzz columns for pairs with total >= FUZZ_MIN
    (those dominate the precision frontier), computed in parallel; others get 0.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

FUZZ_MIN = 1.4  # rapidfuzz features are computed only above this blocking total

FEATURES = [
    "sc_total", "sc_name", "sc_join", "sc_fuzzy", "sc_addr", "agree",
    "rank", "n_cands", "rank_frac", "sc_rel_best", "sc_margin_best",
    "row_n_high", "row_n_med",
    "ninter", "njacc", "ncont_a", "ncont_b",
    "ainter", "ajacc", "acont_a", "acont_b",
    "num_match", "num_first", "pin_eq", "pin_both",
    "name_eq", "addr_eq",
    "nlen_ratio", "alen_ratio", "ntok_a", "ntok_b", "ntok_diff", "atok_a", "atok_b",
    "has_addr_a", "has_addr_b",
    "cand_is_s3", "cand_freq", "cand_best", "cand_mean", "cand_comp",
    "f_lev_name", "f_lev_addr", "f_tokset_name",
]

F32 = np.float32


def context(s_idx, c_idx, total, n_s1, n_cand):
    """Per-entity context arrays (small; gathered per pair inside pair_chunk)."""
    t0 = time.time()
    cnt_row = np.bincount(s_idx, minlength=n_s1)
    row_best = np.zeros(n_s1, F32)
    np.maximum.at(row_best, s_idx, total)
    row_hi = np.bincount(s_idx, weights=(total >= 2.0), minlength=n_s1)
    row_med = np.bincount(s_idx, weights=(total >= 1.5), minlength=n_s1)

    cnt_cand = np.bincount(c_idx, minlength=n_cand)
    csum = np.zeros(n_cand)
    np.add.at(csum, c_idx, total)
    cmax = np.zeros(n_cand, F32)
    np.maximum.at(cmax, c_idx, total)
    # strongest *other* score this candidate received elsewhere (hub / competition signal)
    cmax2 = np.zeros(n_cand, F32)
    not_best = total < cmax[c_idx]
    np.maximum.at(cmax2, c_idx[not_best], total[not_best])
    print(f"  context ({time.time() - t0:.0f}s)", flush=True)
    return dict(cnt_row=cnt_row, row_best=row_best, row_hi=row_hi, row_med=row_med,
                cnt_cand=cnt_cand, csum=csum, cmax=cmax, cmax2=cmax2)


def _fuzz_chunk(sdir, cdir, s_idx, c_idx):
    """Worker: normalised edit distances for one chunk of pairs (mmap-backed stores)."""
    from . import store as store_mod
    from rapidfuzz import fuzz
    from rapidfuzz.distance import Levenshtein
    sst = store_mod.Store(sdir)
    cst = store_mod.Store(cdir)
    n = len(s_idx)
    out = (np.zeros(n, F32), np.zeros(n, F32), np.zeros(n, F32))
    names_a = sst.names(s_idx)
    names_b = cst.names(c_idx)
    addrs_a = sst.addrs(s_idx)
    addrs_b = cst.addrs(c_idx)
    nsim, asr = Levenshtein.normalized_similarity, fuzz.token_set_ratio
    o1, o2, o3 = out
    for i in range(n):
        na, nb = names_a[i], names_b[i]
        if na and nb:
            o1[i] = nsim(na, nb)
            o3[i] = asr(na, nb) * 0.01
        aa, ab = addrs_a[i], addrs_b[i]
        if aa and ab:
            o2[i] = nsim(aa, ab)
    return out


def fuzz_features(sdir, cdir, s_idx, c_idx, total, workers=None):
    """rapidfuzz columns for pairs with total >= FUZZ_MIN (the rest stay zero)."""
    n = len(total)
    f1 = np.zeros(n, np.float16)
    f2 = np.zeros(n, np.float16)
    f3 = np.zeros(n, np.float16)
    idx = np.flatnonzero(total >= FUZZ_MIN)
    if len(idx) == 0:
        return f1, f2, f3
    nw = workers or max(1, (os.cpu_count() or 4) - 1)
    step = -(-len(idx) // nw)
    bounds = [(i * step, min((i + 1) * step, len(idx))) for i in range(nw)]
    bounds = [(a, b) for a, b in bounds if b > a]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(len(bounds), nw)) as ex:
        futs = [ex.submit(_fuzz_chunk, sdir, cdir, s_idx[idx[a:b]], c_idx[idx[a:b]]) for a, b in bounds]
        for (a, b), fut in zip(bounds, futs):
            o1, o2, o3 = fut.result()
            sl = idx[a:b]
            f1[sl] = o1
            f2[sl] = o2
            f3[sl] = o3
    print(f"  fuzz features over {len(idx)} pairs ({time.time() - t0:.0f}s)", flush=True)
    return f1, f2, f3


def pair_chunk(sstore, cstore, s_idx, c_idx, sc, ctx):
    """Vectorised feature dict for ONE chunk of pairs (keeps memory bounded)."""
    g = {}
    # ---- Part-1 blocking scores
    total = sc["total"]
    g["sc_total"] = total
    g["sc_name"] = sc["g0"]
    g["sc_join"] = sc["g1"]
    g["sc_fuzzy"] = sc["g2"]
    g["sc_addr"] = sc["g3"]
    g["agree"] = np.minimum((sc["g0"] + sc["g1"] + sc["g2"]) * 0.5, sc["g3"])
    # ---- ranking / context (Part-1 rows are score-sorted)
    g["rank"] = sc["rank"].astype(F32)
    n_cands = ctx["cnt_row"][s_idx].astype(F32)
    g["n_cands"] = n_cands
    g["rank_frac"] = (g["rank"] + 1) / np.maximum(n_cands, 1)
    row_best = ctx["row_best"][s_idx]
    g["sc_rel_best"] = total / np.maximum(row_best, 1e-6)
    g["sc_margin_best"] = total - row_best
    g["row_n_high"] = ctx["row_hi"][s_idx].astype(F32)
    g["row_n_med"] = ctx["row_med"][s_idx].astype(F32)

    # ---- name token intersection (u32 hash sets padded with 0)
    A = sstore.tok1[s_idx]
    B = cstore.tok1[c_idx]
    inter = ((A[:, :, None] == B[:, None, :]) & (A[:, :, None] != 0)).sum((1, 2))
    na = sstore.ntok1[s_idx].astype(F32)
    nb = cstore.ntok1[c_idx].astype(F32)
    g["ninter"] = inter.astype(F32)
    g["njacc"] = inter / np.maximum(na + nb - inter, 1)
    g["ncont_a"] = inter / np.maximum(na, 1)
    g["ncont_b"] = inter / np.maximum(nb, 1)

    # ---- address token intersection
    A = sstore.tok2[s_idx]
    B = cstore.tok2[c_idx]
    inter = ((A[:, :, None] == B[:, None, :]) & (A[:, :, None] != 0)).sum((1, 2))
    aa = sstore.ntok2[s_idx].astype(F32)
    ab = cstore.ntok2[c_idx].astype(F32)
    g["ainter"] = inter.astype(F32)
    g["ajacc"] = inter / np.maximum(aa + ab - inter, 1)
    g["acont_a"] = inter / np.maximum(aa, 1)
    g["acont_b"] = inter / np.maximum(ab, 1)

    # ---- house numbers and postal codes
    NA = sstore.nums[s_idx]
    NB = cstore.nums[c_idx]
    g["num_match"] = ((NA[:, :, None] == NB[:, None, :]) & (NA[:, :, None] != 0)).sum((1, 2)).astype(F32)
    g["num_first"] = ((NA[:, 0] == NB[:, 0]) & (NA[:, 0] != 0)).astype(F32)
    PA = sstore.pins[s_idx]
    PB = cstore.pins[c_idx]
    g["pin_eq"] = ((PA == PB) & (PA != 0)).astype(F32)
    g["pin_both"] = ((PA != 0) & (PB != 0)).astype(F32)

    # ---- exact string agreement and sizes
    g["name_eq"] = (sstore.hname[s_idx] == cstore.hname[c_idx]).astype(F32)
    g["addr_eq"] = (sstore.haddr[s_idx] == cstore.haddr[c_idx]).astype(F32)
    la = sstore.nlen[s_idx].astype(F32)
    lb = cstore.nlen[c_idx].astype(F32)
    g["nlen_ratio"] = np.abs(la - lb) / np.maximum(np.maximum(la, lb), 1)
    la = sstore.alen[s_idx].astype(F32)
    lb = cstore.alen[c_idx].astype(F32)
    g["alen_ratio"] = np.abs(la - lb) / np.maximum(np.maximum(la, lb), 1)
    g["ntok_a"] = na
    g["ntok_b"] = nb
    g["ntok_diff"] = np.abs(na - nb)
    g["atok_a"] = aa
    g["atok_b"] = ab
    g["has_addr_a"] = (aa > 0).astype(F32)
    g["has_addr_b"] = (ab > 0).astype(F32)

    # ---- candidate provenance / competition
    g["cand_is_s3"] = ((cstore.keys[c_idx] & 3) == 3).astype(F32)
    g["cand_freq"] = np.log1p(ctx["cnt_cand"][c_idx]).astype(F32)
    cmax = ctx["cmax"][c_idx]
    g["cand_best"] = cmax
    g["cand_mean"] = (ctx["csum"][c_idx] / np.maximum(ctx["cnt_cand"][c_idx], 1)).astype(F32)
    g["cand_comp"] = total - ctx["cmax2"][c_idx]
    return g


def fill_matrix(sstore, cstore, s_idx, c_idx, sc, ctx, fuzz, out, lo, hi):
    """Compute pairs [lo:hi) into `out`, a view of exactly (hi-lo) feature rows.

    `sc` holds the per-pair score arrays (total, g0..g3, rank) in FULL length.
    """
    sl = slice(lo, hi)
    rows = slice(0, hi - lo)
    g = pair_chunk(sstore, cstore, s_idx[sl], c_idx[sl], {k: v[sl] for k, v in sc.items()}, ctx)
    g["f_lev_name"] = fuzz[0][sl].astype(F32)
    g["f_lev_addr"] = fuzz[1][sl].astype(F32)
    g["f_tokset_name"] = fuzz[2][sl].astype(F32)
    for j, name in enumerate(FEATURES):
        out[rows, j] = g[name]


def assert_features_match(model):
    got = list(model.feature_name())
    if got and got != FEATURES:
        raise RuntimeError(f"model features {got} != feats.FEATURES {FEATURES}")


def build_matrix(sstore, cstore, s_idx, c_idx, sc, ctx, fuzz, path, chunk=250_000):
    """Fill an on-disk float32 (n_pairs, n_features) matrix in chunks."""
    n = len(s_idx)
    out = np.lib.format.open_memmap(path, dtype=F32, shape=(n, len(FEATURES)), mode="w+")
    t0 = time.time()
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        fill_matrix(sstore, cstore, s_idx, c_idx, sc, ctx, fuzz, out[lo:hi], lo, hi)
        if lo % (chunk * 10) == 0:
            print(f"  features {lo}/{n} ({time.time()-t0:.0f}s)", flush=True)
    out.flush()
    return out
