#!/usr/bin/env bash
# Part 2: matching model (LightGBM over Part-1 candidates).
# Run from anywhere. Requires: python 3.10+, `pip install -r requirements.txt`,
# and Part 1 already run (work/val/candidates_scored.tsv + work/test_candidates_scored.tsv
# + output/candidate_pairs.tsv).
#
# Steps (each caches its artefacts under work/part2/, so reruns skip finished work):
#   build_val     -> parse val candidates, build record stores, labels, feature matrix
#   train_model   -> LightGBM + F0.5 threshold tuned on held-out val rows
#   predict_test  -> score every test candidate, write output/matching_results.tsv, run validator
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"          # student_resource/
cd "$HERE/matching"
PY="${PYTHON:-python}"

# ---- resource budget: stay under ~80% of CPUs; GPU (opt-in) runs uncapped. ----
# Precedence: JOBS > PART2_MAX_WORKERS > 80% of nproc.
if [ -z "${JOBS:-}" ]; then
  if [ -n "${PART2_MAX_WORKERS:-}" ]; then
    JOBS="$PART2_MAX_WORKERS"
  elif command -v nproc >/dev/null 2>&1; then
    JOBS=$(( $(nproc) * 8 / 10 ))
  else
    JOBS=$("$PY" -c "import os; print(max(1, int((os.cpu_count() or 4) * 0.8)))")
  fi
fi
[ "${JOBS:-0}" -lt 1 ] && JOBS=1
export PART2_MAX_WORKERS="$JOBS"
export LGB_NUM_THREADS="${LGB_NUM_THREADS:-$JOBS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$JOBS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$JOBS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$JOBS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$JOBS}"
# GPU opt-in (uncapped by design): USE_GPU=1 bash src/run_part2.sh
# or PART2_DEVICE=gpu; resolved in part2.resources (auto > env > cpu).
echo "resource budget: JOBS=$JOBS (cpus capped at 80%), device=${PART2_DEVICE:-${USE_GPU:+gpu}(auto->cpu)}" >&2
case " $* " in *" --jobs "*) JOBS_ARG="";; *) JOBS_ARG="--jobs $JOBS";; esac

# shellcheck disable=SC2086
"$PY" -m part2.build_val     --root "$ROOT" $JOBS_ARG
# shellcheck disable=SC2086
"$PY" -m part2.train_model   --root "$ROOT" $JOBS_ARG
# shellcheck disable=SC2086
"$PY" -m part2.predict_test  --root "$ROOT" $JOBS_ARG "$@"
