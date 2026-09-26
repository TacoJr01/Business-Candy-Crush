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

"$PY" -m part2.build_val     --root "$ROOT"
"$PY" -m part2.train_model   --root "$ROOT"
"$PY" -m part2.predict_test  --root "$ROOT" "$@"
