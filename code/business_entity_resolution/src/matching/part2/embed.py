"""Optional semantic-similarity feature for the Part-2 matcher (off by default).

Enable with ``PART2_EMBED=1`` (needs ``sentence-transformers`` + a CUDA torch;
the repo ``.venv`` already has torch+cu126 — run the pipeline with it, or
``pip install sentence-transformers`` into your pipeline python):

    PART2_EMBED=1 .venv/Scripts/python -m part2.build_val --root .

What it does: embeds each record's ``"<name> | <address>"`` string with a
multilingual MiniLM model and adds one feature, ``e_cos`` — the cosine between
the S1 record and the candidate. This catches transliteration pairs whose
lexical overlap is ~0 (the biggest India-recall gap), where Jaccard/Levenshtein
features are blind.

Cost notes: embeddings are cached per store (``emb_fp16.npy`` + ``emb.done``)
so reruns are free; first pass is hours on a 4 GB GPU for the full test
stores. Val stores are small — validate the gain there first.
"""

import os
from pathlib import Path

import numpy as np

MODEL_ENV = "PART2_EMBED_MODEL"
DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DIM = 384


def enabled():
    return os.environ.get("PART2_EMBED", "").strip().lower() in ("1", "true", "yes")


def _model_name():
    return os.environ.get(MODEL_ENV, "").strip() or DEFAULT_MODEL


def ensure(sdir):
    """Return an fp16 memmap (n, DIM) of L2-normalised embeddings for a Store.

    Computes in 200k-row blocks with the CUDA venv model on first call, then
    reuses the on-disk cache.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit(
            "PART2_EMBED=1 needs sentence-transformers: "
            "run with the repo .venv python (.venv/Scripts/python -m part2.....) "
            "after `.venv/Scripts/pip install sentence-transformers`, or pip-install "
            "it into your pipeline python.")
    import torch
    from . import store as store_mod

    sdir = Path(sdir)
    done, path = sdir / "emb.done", sdir / "emb_fp16.npy"
    st = store_mod.Store(str(sdir))
    n = st.n
    if done.exists() and path.exists():
        return np.lib.format.open_memmap(path, dtype=np.float16, shape=(n, DIM), mode="r")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("  WARNING: no CUDA — embedding on CPU will be very slow", flush=True)
    mdl = SentenceTransformer(_model_name(), device=dev)
    try:
        mdl.max_seq_length = 64
    except Exception:
        pass
    bs = int(os.environ.get("PART2_EMBED_BATCH", "1024"))
    out = np.lib.format.open_memmap(path, dtype=np.float16, shape=(n, DIM), mode="w+")
    for lo in range(0, n, bs * 200):
        hi = min(lo + bs * 200, n)
        idx = np.arange(lo, hi, dtype=np.int64)
        texts = [a + " | " + b for a, b in zip(st.names(idx), st.addrs(idx))]
        vec = mdl.encode(texts, batch_size=bs, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True,
                         precision="float16" if dev == "cuda" else "float32")
        out[lo:hi] = vec.astype(np.float16)
        out.flush()
        print(f"  embedded {hi}/{n} ({_model_name()} on {dev})", flush=True)
    done.write_text(_model_name(), encoding="utf-8")
    return np.lib.format.open_memmap(path, dtype=np.float16, shape=(n, DIM), mode="r")


def pair_cosine(sdir, cdir, s_idx, c_idx, block=1 << 20):
    """Cosine per pair (float16 array) from the two stores' embedding caches."""
    se = ensure(sdir)
    ce = ensure(cdir)
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
            print(f"  e_cos {lo}/{n}", flush=True)
    return out
