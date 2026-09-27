# Business Candy Crush — ML Challenge 2026: Business Entity Resolution

Two-stage entity-resolution pipeline for matching noisy business records (S1 → S2+S3)
at 1.7M-record scale, optimized for macro F0.5.

- **Part 1 — Blocking (Rust):** from-scratch inverted TF-IDF index over 5 channels,
  multi-list top-k candidate generation. Pair recall **0.9687**, oracle F0.5 ceiling
  **0.9891** at ~42 candidates/S1 on a 220,441-entity train holdout.
- **Part 2 — Matching (Python/LightGBM):** 44-feature pairwise classifier with a
  global F0.5-tuned threshold. Macro F0.5 **0.9493** on held-out calibration
  (vs 0.6894 baseline).
- **Part 3 — Evaluation + Packaging:** validator PASS, pruned candidate set,
  submission zip + leaderboard upload.

## Repository layout

```
<workspace>/
├── Makefile                           # central entry point (Linux/macOS/Windows, see below)
├── .github/workflows/validate.yml     # CI: validator on a tiny fixture, every push/PR
├── tests/fixtures/ci/                 # 3-row validator fixture (tracked)
├── code/                              # this git repo
│   └── business_entity_resolution/
│       ├── src/business_candy_crush/  # Part 1: Rust blocker
│       ├── src/matching/part2/        # Part 2: Python matcher (+ resources.py budget)
│       ├── src/run_part1.sh           # end-to-end blocking (or `make part1`)
│       ├── src/run_part2.sh           # train + predict test (or `make part2`)
│       ├── README.md                  # pipeline details + CLI
│       └── requirements.txt           # Python 3.11+ (CPU default, GPU opt-in)
├── dataset/                           # from student_resource.zip (git-ignored)
│   ├── train/                         # ~2.2M S1, ~10.3M S2+S3
│   └── test/                          # 1,732,544 S1
├── utils/validate_submission.py       # tracked (rest of utils/ is git-ignored)
├── output/                            # submission files (git-ignored)
│   ├── matching_results.tsv           # THE scored submission (1 row per test S1)
│   └── candidate_pairs.tsv            # audited candidate set (smaller = better)
├── work/                              # caches: val eval, part2 stores/features (git-ignored)
├── results/part2/                     # tracked: model_meta.json, train_report.txt
├── Documentation_template.md          # filled-in submission write-up (tracked)
└── HANDOFF.md                         # Part 1/2 → Part 3 handoff notes
```

Large artefacts (`output/`, `work/`, model file) are **not committed**
(GitHub 100 MB limit) — they ship as a Release asset (`part2_outputs.zip`).
Small metrics/config under `results/part2/` **are** committed.

## Results

| Stage | Metric (macro F0.5) | Detail |
| --- | --- | --- |
| Part 1 blocking | oracle ceiling 0.9891 | recall 0.9687, ~42.3 cands/S1 (220k holdout) |
| Baseline matcher | 0.6894 | tuned blocking-score threshold (τ≈2.34) |
| Part 2 LightGBM (τ=0.73) | **0.9493** | P 0.983 / R 0.898; US 0.9647, India 0.9265 |
| Pruned candidates (total ≥ 0.8) | −0.0009 F0.5 | 72.1% kept (72.06M → 51.96M pairs), smaller = better for audit |

Full numbers: `results/part2/train_report.txt`, `results/part2/model_meta.json`,
`Documentation_template.md` §5–§6.

## Reproduce

Requirements: Rust 1.98+, Python 3.11+, `dataset/` next to `code/` (see layout above).
Windows also needs GNU Make 4.x (`choco install make`) for the `make` path.

```powershell
make setup   # one time: pip install -r requirements.txt
make all     # part1 + part2 + validate
# or per stage:
make part1
make part2
make part3   # evaluation + packaging: validate, prune (CUTOFF=0.8), submission zip
```

Script fallback (Git Bash / any POSIX shell):

```bash
pip install -r code/business_entity_resolution/requirements.txt
bash code/business_entity_resolution/src/run_part1.sh
bash code/business_entity_resolution/src/run_part2.sh
```

- `run_part1.sh`: builds index, blocks train holdout → `work/val/eval.txt`,
  blocks test → `output/candidate_pairs.tsv` + baseline `output/matching_results.tsv`.
- `run_part2.sh`: `build_val` → `train_model` → `predict_test`
  (overwrites `output/matching_results.tsv` with model decisions,
  prunes both outputs, runs the validator).
- All caches live under `work/` — safe to delete, each step is idempotent.
- Timings: Part 1 ~1 h test blocking; Part 2 predict ~45–60 min on 12-core/16 GB.

## Resource budget

CPU/RAM are capped at ~80% of the machine by default (JOBS = floor(logical CPUs × 0.8)):
Rust rayon pool (`--jobs` / `RAYON_NUM_THREADS`), Python `ProcessPool` workers
(`PART2_MAX_WORKERS`), LightGBM + BLAS threads (`LGB_NUM_THREADS`,
`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS`). Pair loops stay chunked (`PART2_CHUNK`).
GPU (LightGBM) is opt-in and runs uncapped — CPU-side work stays capped:

```powershell
make part2 DEVICE=gpu   # or PART2_DEVICE=gpu / USE_GPU=1 with the scripts
```

`PART2_ALLOW_FULL=1` lifts the CPU cap (explicit user wish).
Central defaults live in `src/matching/part2/resources.py`, enforced by the
`Makefile` and `run_part*.sh`.

## CI

`.github/workflows/validate.yml` runs `utils/validate_submission.py` (default mode
plus `--check-ids`) against the hand-made 3-row fixture in `tests/fixtures/ci/`
on every push/PR — real outputs are too big for git, so the fixture proves the
validator wiring instead. `.github/workflows/docker.yml` builds the Docker image.

## AWS CodeBuild + Docker

- `buildspec.yml` (repo root): install (Python 3.12 + Rust 1.98.1 via rustup) →
  validator fixture → `py_compile` → `cargo check`. Full `make all` needs
  `dataset/` at the build root (fetch `student_resource.zip` from S3 in
  `pre_build`; example commented in the file).
- `Dockerfile` (multi-stage: Rust builder + Python runtime): `docker build -t bcc .`,
  then mount data at run time — `dataset/`, `output/`, `work/` are never baked in:
  `docker run --rm -v "$PWD/dataset:/workspace/dataset" -v "$PWD/output:/workspace/output" -v "$PWD/work:/workspace/work" bcc make part1`.

## Validate

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
# must print PASS
```

Scoring rule (macro F0.5, singletons = 1.0 when predicted empty) is implemented
in `src/matching/part2/scores.py`.

## Submission zip

```
<team_name>_submission.zip
├── output/matching_results.tsv
├── output/candidate_pairs.tsv
├── code/business_entity_resolution/{src/,README.md,requirements.txt}
└── Documentation_template.md   (filled in)
```

Keep it free of `work/` caches and `dataset/` copies.
Leaderboard upload = `output/matching_results.tsv` via the Portal.

## Key constraints

- No external data / geocoding / APIs. No country-specific branching
  (country is an opaque hash key; France in test is unseen in train).
- One global threshold τ on model probability, tuned on untouched calibration
  entities only — do not re-tune against the leaderboard.
- Model is LightGBM (MIT, single-digit MB) — well within the ≤8B-param limit.

## Docs

- `code/business_entity_resolution/README.md` — full pipeline, CLI, blocking/matcher design.
- `Documentation_template.md` — submission write-up (§5 results, §6 conclusion).
- `HANDOFF.md` — Parts 1+2 context and remaining Part 3 checklist.
- `results/part2/train_report.txt` — calibration table + threshold sweep.
