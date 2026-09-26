#!/usr/bin/env bash
# Part 1: blocking / candidate generation.
# Run from anywhere; DATA defaults to the student_resource/dataset folder.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"          # student_resource/
DATA="${DATA:-$ROOT/dataset}"
# Extra flags for `block`, e.g. BLOCK_ARGS="--max-df 50000" for a faster, slightly lower-recall run.
BLOCK_ARGS="${BLOCK_ARGS:-}"

cargo build --release --manifest-path "$HERE/business_candy_crush/Cargo.toml"
BIN="$HERE/business_candy_crush/target/release/business_candy_crush"

# 1) Train holdout (fnv1a64(S1 id) % 10 == 0) against the FULL train S2/S3 corpus -> recall metrics.
"$BIN" block --s1 "$DATA/train/train_source1.tsv" --s2 "$DATA/train/train_source2.tsv" \
  --s3 "$DATA/train/train_source3.tsv" --out-dir "$ROOT/work/val" $BLOCK_ARGS --val-only
"$BIN" eval --scored "$ROOT/work/val/candidates_scored.tsv" --gt "$DATA/train/train_ground_truth.tsv" \
  --s1 "$DATA/train/train_source1.tsv" --misses "$ROOT/work/val/misses.tsv" | tee "$ROOT/work/val/eval.txt"

# 2) Full test set -> output/candidate_pairs.tsv (+ baseline matching_results.tsv).
"$BIN" block --s1 "$DATA/test/test_source1.tsv" --s2 "$DATA/test/test_source2.tsv" \
  --s3 "$DATA/test/test_source3.tsv" --out-dir "$ROOT/output" $BLOCK_ARGS
mv "$ROOT/output/candidates_scored.tsv" "$ROOT/work/test_candidates_scored.tsv"

cd "$ROOT" && python3 utils/validate_submission.py \
  --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
