"""Shared CPU/RAM budget (≤80% by default) + optional full-GPU selection.

Policy
------
* CPU + RAM are restrained: every parallel stage (ProcessPool workers, LightGBM
  threads, BLAS/OpenMP threads, Rust rayon pool) defaults to
  ``floor(logical_cpus * 0.8)``. On a 22-thread box that is 17 workers.
* GPU is *opt-in and uncapped*: ``--device gpu`` / ``PART2_DEVICE=gpu`` /
  ``USE_GPU=1`` lets LightGBM train/infer on the GPU at full power, while all
  CPU-side work (parsing, features, BLAS) stays under the 80% cap.
* Escape hatch: ``PART2_ALLOW_FULL=1`` disables capping (explicit user wish).

Environment knobs (all optional)
--------------------------------
* ``PART2_MAX_WORKERS`` / ``LGB_NUM_THREADS`` — explicit worker/thread counts
  (still clamped to the 80% cap unless ``PART2_ALLOW_FULL=1``).
* ``PART2_FRAC`` — cap fraction, default ``0.8``.
* ``PART2_DEVICE`` — ``cpu`` (default) or ``gpu``. ``USE_GPU=1`` implies gpu.
* ``PART2_CHUNK`` — pair-chunk size for feature/inference loops (RAM restraint).
* ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS`` /
  ``NUMEXPR_NUM_THREADS`` / ``RAYON_NUM_THREADS`` — honoured / exported by the
  shell wrappers; :func:`apply_thread_env` fills any that are unset.
"""

import os

DEFAULT_FRAC = 0.8
DEFAULT_CHUNK = 250_000


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def cpu_count():
    """Logical CPU count, min 1."""
    try:
        return max(1, int(os.cpu_count() or 4))
    except Exception:
        return 4


def cap_count(frac=None):
    """80%-style cap: floor(cpus * frac), at least 1."""
    frac = DEFAULT_FRAC if frac is None else frac
    try:
        frac = float(frac)
    except (TypeError, ValueError):
        frac = DEFAULT_FRAC
    frac = min(max(frac, 0.05), 1.0)
    return max(1, int(cpu_count() * frac))


def _allow_full():
    return os.environ.get("PART2_ALLOW_FULL", "").strip() in ("1", "true", "yes")


def resolve_workers(explicit=None, frac=None):
    """Worker count for ProcessPools, clamped to the CPU cap by default."""
    cap = cap_count(os.environ.get("PART2_FRAC", DEFAULT_FRAC if frac is None else frac))
    if explicit is None:
        env = os.environ.get("PART2_MAX_WORKERS", "").strip()
        explicit = int(env) if env.isdigit() else None
    if explicit is not None:
        try:
            explicit = max(1, int(explicit))
        except (TypeError, ValueError):
            return cap
        return explicit if _allow_full() else min(explicit, cap)
    return cap


def lgb_threads(explicit=None):
    """Thread count for LightGBM (CPU side only; GPU kernels stay uncapped)."""
    if explicit is None:
        env = os.environ.get("LGB_NUM_THREADS", "").strip()
        explicit = int(env) if env.lstrip("-").isdigit() else None
    cap = cap_count(os.environ.get("PART2_FRAC", DEFAULT_FRAC))
    if explicit is not None and explicit > 0:
        return explicit if _allow_full() else min(explicit, cap)
    return cap


def resolve_chunk(default=DEFAULT_CHUNK):
    """Chunk size for pair loops (RAM restraint knob)."""
    env = os.environ.get("PART2_CHUNK", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    try:
        return max(1, int(default))
    except (TypeError, ValueError):
        return DEFAULT_CHUNK


def resolve_device(explicit=None):
    """'gpu' iff explicitly requested, else 'cpu'."""
    if explicit:
        explicit = str(explicit).lower()
        if explicit in ("gpu", "cuda"):
            return "gpu"
        if explicit in ("cpu", "auto"):
            pass  # fall through to env check unless pinned to cpu
            if explicit == "cpu":
                return "cpu"
    env = os.environ.get("PART2_DEVICE", "").strip().lower()
    if env in ("gpu", "cuda") or os.environ.get("USE_GPU", "").strip() in ("1", "true", "yes"):
        return "gpu"
    return "cpu"


def lgb_device_params(device):
    """Extra LightGBM params for the requested device (GPU stays full-power)."""
    if device == "gpu":
        return {"device_type": "gpu", "gpu_device_id": 0}
    return {"device_type": "cpu"}


def apply_thread_env(frac=None):
    """Fill unset BLAS/OpenMP thread env vars with the capped count.

    Must run before numpy/scipy/lightgbm are imported for full effect; the
    ``run_part*.sh`` wrappers already export these, so this is a safety net
    for direct ``python -m part2.*`` invocations.
    """
    n = str(cap_count(os.environ.get("PART2_FRAC", DEFAULT_FRAC if frac is None else frac)))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, n)
    return n
