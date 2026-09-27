"""CPU char-level semantic feature for the Part-2 matcher (off by default).

Enable with ``PART2_CHARVEC=1``. Adds one feature, ``c_tfidf`` — the cosine
between 64-dim char-3gram TF-IDF+SVD vectors of the two records' normalized
names. Prototype on 20k val pairs: pos mean 0.82 vs neg mean 0.45, single-
feature AUC 0.81. Catches compounds / multi-typo / transliteration pairs
where token-level lexical features go blind, with zero GPU need.

Pipeline (all cached under work/part2/charvec/, reruns are free):
  * fit once on a 1M-name sample of train S2+S3 (unsupervised: no labels,
    S1/test never touch the fit) -> vec.pkl + svd.pkl;
  * transform each record store in blocks -> per-store fp16 memmap;
  * blocked cosine per pair (fp32 accumulation), like part2.embed.
"""

import os
import pickle
import time
from pathlib import Path

import numpy as np

VER = 1
DIM = 64
N_FIT = 1_000_000
VOCAB = 100_000


def enabled():
    return os.environ.get("PART2_CHARVEC", "").strip().lower() in ("1", "true", "yes")


def _dir(root):
    d = Path(root) / "work/part2/charvec"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sample_names(paths, n, seed=7):
    rng = np.random.default_rng(seed)
    counts = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            next(f)
            counts.append(sum(1 for _ in f))
    per = [max(1, int(n * c / sum(counts))) for c in counts]
    out = []
    for p, k in zip(paths, per):
        idx = set(rng.choice(counts[paths.index(p)], size=min(k, counts[paths.index(p)]),
                             replace=False).tolist())
        with open(p, encoding="utf-8") as f:
            next(f)
            for i, line in enumerate(f):
                if i in idx:
                    from . import norm as norm_mod
                    c1 = line.find("\t")
                    c2 = line.find("\t", c1 + 1)
                    out.append(norm_mod.normalize(line[c1 + 1:c2]))
    return out


def _fit(root):
    """Fit vectorizer + SVD on a 1M train S2+S3 sample; cache pickles."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    d = _dir(root)
    mark = d / "fit.done"
    if mark.exists() and mark.read_text().strip() == f"v{VER}":
        vec = pickle.loads((d / "vec.pkl").read_bytes())
        svd = pickle.loads((d / "svd.pkl").read_bytes())
        return vec, svd
    t0 = time.time()
    tr = Path(root) / "dataset/train"
    texts = _sample_names([tr / "train_source2.tsv", tr / "train_source3.tsv"], N_FIT)
    print(f"  charvec fit texts: {len(texts)}", flush=True)
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), max_features=VOCAB,
                          sublinear_tf=True)
    X = vec.fit_transform(texts)
    svd = TruncatedSVD(n_components=DIM, random_state=7).fit(X)
    print(f"  charvec fit done, explained={svd.explained_variance_ratio_.sum():.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    (d / "vec.pkl").write_bytes(pickle.dumps(vec))
    (d / "svd.pkl").write_bytes(pickle.dumps(svd))
    mark.write_text(f"v{VER}", encoding="utf-8")
    return vec, svd


def ensure(sdir, root):
    """Fp16 memmap (n, DIM) of normalised name vectors for a Store (cached)."""
    from . import store as store_mod

    sdir = Path(sdir)
    path, done = sdir / "cvec_fp16.npy", sdir / "cvec.done"
    st = store_mod.Store(str(sdir))
    n = st.n
    if done.exists() and done.read_text().strip() == f"v{VER}" and path.exists():
        return np.lib.format.open_memmap(path, dtype=np.float16, shape=(n, DIM), mode="r")
    vec, svd = _fit(root)
    out = np.lib.format.open_memmap(path, dtype=np.float16, shape=(n, DIM), mode="w+")
    bs = 200_000
    for lo in range(0, n, bs):
        hi = min(lo + bs, n)
        idx = np.arange(lo, hi, dtype=np.int64)
        Z = svd.transform(vec.transform(st.names(idx))).astype(np.float64)
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
        out[lo:hi] = Z.astype(np.float16)
        out.flush()
        print(f"  charvec {sdir.name}: {hi}/{n}", flush=True)
    done.write_text(f"v{VER}", encoding="utf-8")
    return np.lib.format.open_memmap(path, dtype=np.float16, shape=(n, DIM), mode="r")


def pair_cosine(sdir, cdir, s_idx, c_idx, root, block=1 << 20):
    """Cosine per pair (float16 array) from the two stores' char-vector caches."""
    se = ensure(sdir, root)
    ce = ensure(cdir, root)
    assert se.shape[1] == ce.shape[1] == DIM, (se.shape, ce.shape)
    n = len(s_idx)
    out = np.empty(n, np.float16)
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        a = se[np.asarray(s_idx[lo:hi], dtype=np.int64)].astype(np.float32)
        b = ce[np.asarray(c_idx[lo:hi], dtype=np.int64)].astype(np.float32)
        out[lo:hi] = np.einsum("ij,ij->i", a, b)
        del a, b
        if lo % (block * 8) == 0:
            print(f"  c_tfidf {lo}/{n}", flush=True)
    return out
