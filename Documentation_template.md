# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Business Candy Crush  
**Team Members:** Aditya Sunil, Saloni Palo, Debaditya Malakar  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

A two-stage entity-resolution pipeline built for scale and precision: a from-scratch
Rust inverted-index blocker over four complementary TF-IDF channels (word, joined-key,
phonetic-skeleton and address) that reaches a 0.989 oracle F0.5 ceiling at ~42
candidates/entity on 1.7M test records, followed by a LightGBM matcher over 44
country-agnostic features (blocking scores, rank/context, lexical agreement, edit
distances, hub-competition signals) with the decision threshold tuned directly for
macro F0.5 on a held-out entity split.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the training data (2.2M S1, 10.3M S2+S3 records) showed:

- **Name noise:** legal-form abbreviations (Pvt/Ltd/Corp), `&` vs "and", DBA/trade
  names, word splits/joins ("Beth Chapel" vs "bethchapel.com"), word-order swaps,
  typos, and heavy Indic-script usage in Indian records — many records store the
  name in Devanagari/Bengali/Tamil/Telugu/Kannada while its match is Latin-transliterated.
- **Address noise:** street-type abbreviations, missing PIN codes, landmark references,
  municipal numbering, component reordering, leading-zero house numbers ("01018" = "1018").
- **Country:** every sampled true pair stays inside its country, but the test set adds
  `France`, unseen in training. The pipeline therefore treats `country` as an opaque
  string (hashed into blocking keys), never a fixed enum — no one-hot, no hard-coded
  country list.
- **Hubs:** generic names ("Orthopedic Care") and empty addresses make a naive
  single-ranking blocking lose true matches; and popular candidate records attract many
  S1 queries, so false-merge risk concentrates on high-frequency candidates.
- **Singletons** are 5.6% of val entities and worth a full 1.0 each under macro F0.5.

### 2.2 Solution Strategy

**Approach Type:** Blocking (Part 1, Rust) + learned pairwise classifier (Part 2, LightGBM) with a global F0.5-tuned decision threshold.  
**Core Innovation:** (a) channel-specific top-k candidate lists (combined score, name–address *agreement*, address-less name ranking, and each score group alone) so generic names can never crowd a true match out of a single ranking; (b) feeding the matcher not just per-pair text features but also *ranking context* (position within the query's candidate list) and *competition features* (how strongly other S1 entities claim the same candidate) — decisive for hub disambiguation under a precision-heavy metric.

---

## 3. Candidate Generation (Blocking)

Rust binary `src/business_candy_crush` (Part 1 code):

- **Normalisation:** Indic→Latin transliteration via one ISCII-derived table (all nine
  Indian script blocks share the layout), NFKD accent stripping, punctuation folding
  (dots/apostrophes deleted), web-name stripping (`bethchapel.com` → `bethchapel`),
  legal-form/street-type/state canonicalisation tables, zero-stripped numbers.
- **Blocking keys (5 channels, hashed with the opaque country label):** name words;
  joined-name keys (ordered + sorted); a phonetic consonant-skeleton + 4-char
  prefix/suffix; address words; address skeletons.
- **Index:** inverted index over S2+S3 with per-country IDF and per-channel L2
  normalisation; docs renumbered by (country, city-token) for cache locality;
  posting lists with df > 100k skipped.
- **Query:** per S1 record, candidate set = union of top-k lists over: the combined
  weighted score, the name–address agreement `min(name/2, addr)`, name-score among
  address-less records, and each score group alone.

- **Candidate pairs generated:** 72M pairs for 1,732,544 test S1 entities (~41.5/S1);
  9.32M for the 220,441-entity train holdout (~42.3/S1).
- **How you ensured true matches were not lost:** on the train holdout blocking captures
  96.9% of true pairs (98.3% US / 94.8% India) and retains *all* matches for 90.6% of
  entities; oracle macro-F0.5 ceiling on the candidate set is 0.989. The per-group
  top-k lists exist precisely to stop recall loss on namesakes/no-address records;
  empty output rows (S1 with zero candidates) are legal and rare.

---

## 4. Matching Model

Part 2 code: `src/matching/part2/` (Python, CPU-only). All features are recomputed
identically for val and test from the raw source files; no country, currency, or
region-specific branching is used anywhere.

**Features used (44 total, `part2/feats.py`):**
- Name features: blocking cosines (word/join/skeleton channels), token-set Jaccard and
  two-way containment over canonicalised name tokens (u32-hashed), exact normalised-name
  agreement, normalised length ratio, token counts, rapidfuzz Levenshtein similarity
  and token-set ratio on transliterated names.
- Address features: address-group cosine, token Jaccard/containment, house-number
  (numeric-token) overlap and first-number equality, postal/PIN equality, address length
  ratio, presence flags, rapidfuzz Levenshtein on addresses.
- Other: candidate-list context (rank, list length, score relative to the query's best,
  how many high/medium-score rivals the query has), and candidate competition features
  (how many S1 lists claim this candidate, its best score overall, mean score, and the
  margin of this query versus the candidate's strongest other claimant) — these give the
  model a hub-removal signal. Edit distances are computed above blocking-total 1.4
  (~25% of pairs), where the precision frontier lives; cheaper vectorised signals cover
  the tail.

**Model type:** LightGBM binary classifier (MIT licence, ~4 MB trees — far below the
8B-param limit), 5-fold-by-entity protocol: fit folds train the model, an eval fold
early-stops tree count, and the final refit + a single global decision threshold are
chosen to maximise macro F0.5 on the untouched calibration fold (~28k val entities).
Singletons need no special rule: empty predictions score correctly through the threshold.

**Threshold selection method:** direct macro-F0.5 sweep (per-entity F0.5, averaged over
all val entities incl. singletons) over tau ∈ [0.02, 0.98] on the calibration fold,
ties resolved toward the stricter threshold.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9493** on the calibration fold (27,395 val entities never
  used for fitting or early stopping) — vs **0.6894** for a tuned blocking-score
  threshold on the same entities, and 0.9891 for the perfect-matcher oracle ceiling on
  the candidate set. Breakdown: US 0.9647 (P 0.987 / R 0.924), India 0.9265 (P 0.977 /
  R 0.860); pooled P 0.983, R 0.898 at the chosen threshold tau = 0.73. The threshold
  sweep has a wide flat maximum (F0.5 ≥ 0.948 across tau ∈ [0.62, 0.77]), so the
  decision boundary is not a calibration spike.
- **Common false positives (wrong merges):** namesakes in the same city (same street,
  different business) — concentrated where the candidate's competition margin
  (`cand_comp`) is near zero — and generic DBA/trade names whose canonicalised token
  sets collide (e.g. two unrelated "Trading Co." at one landmark address).
- **Common false negatives (missed matches):** India transliteration pairs whose address
  drifted too (address channel cosine ≈ 0 and no house-number anchor); records whose
  true match was already outside the blocking recall ceiling (≈3.1% of pairs never
  appear as candidates); and long-tail pairs below the 0.8 blocking-total floor that is
  pruned before scoring (0.09 F0.5 pt).
- Top model signals by gain: address edit distance, blocking total, address token
  Jaccard/containment, the name×address agreement score, first-house-number equality and
  the hub-competition margin — i.e. precision comes from address evidence plus
  knowing when another S1 entity claims the same candidate more strongly.

---

## 6. Conclusion

A country-agnostic Rust inverted-index blocker puts a 0.989 oracle-F0.5 ceiling within
~42 candidates per entity at 1.7M-record scale, and a 44-feature LightGBM matcher —
combining the blocker's per-channel scores with lexical agreement, list-rank context and
candidate-competition features — converts that ceiling into a measured 0.949 macro F0.5
on untouched validation entities, a +26-point lift over a tuned score threshold. The
decision threshold was optimised for the exact leaderboard metric (macro F0.5,
singletons included) on a held-out fold; the final candidate file is pruned to a
blocking-total floor so the audited candidate set is exactly what the model scores
(smaller = better), at a cost of under 0.1 F0.5 point. No external data, no country
branching; the pipeline reruns end-to-end from the two shell scripts in ~2 CPU-hours.

---

## Appendix

### A. Code Artefacts

```
code/business_entity_resolution/
├── src/business_candy_crush/        # Part 1: Rust blocker (cargo build --release)
├── src/matching/part2/              # Part 2: python package
│   ├── norm.py        # country-agnostic normalisation (mirrors the Rust pipeline)
│   ├── store.py       # per-record arrays: token hashes, numbers, PINs, strings (disk-cached, mmap)
│   ├── scoresrc.py    # parallel parser for candidates_scored.tsv
│   ├── feats.py       # 44 vectorised pair features + parallel rapidfuzz
│   ├── scores.py      # exact macro-F0.5 scorer / threshold sweeps
│   ├── build_val.py   # val holdout: labels + feature matrix
│   ├── train_model.py # LightGBM + threshold calibration (writes work/part2/model.txt, train_report.txt)
│   └── predict_test.py# scores all test candidates, writes output/, runs the validator
├── src/run_part1.sh
└── src/run_part2.sh   # build_val -> train_model -> predict_test
```

Reproduce end-to-end: `bash src/run_part1.sh && bash src/run_part2.sh` from
`code/business_entity_resolution/`. Outputs land in `output/`.

### B. Additional Results

Threshold sweep on the calibration fold (27,395 S1, from `work/part2/train_report.txt`).
Wide flat maximum: F0.5 ≥ 0.948 across tau ∈ [0.62, 0.77].

| tau | macro F0.5 |
| --- | --- |
| 0.02 | 0.7697 |
| 0.07 | 0.8689 |
| 0.12 | 0.8977 |
| 0.17 | 0.9130 |
| 0.22 | 0.9228 |
| 0.27 | 0.9294 |
| 0.32 | 0.9350 |
| 0.37 | 0.9391 |
| 0.42 | 0.9425 |
| 0.47 | 0.9450 |
| 0.52 | 0.9469 |
| 0.57 | 0.9482 |
| 0.62 | 0.9489 |
| 0.67 | 0.9491 |
| 0.72 | 0.9492 |
| 0.77 | 0.9486 |
| 0.82 | 0.9466 |
| 0.87 | 0.9432 |
| 0.92 | 0.9350 |
| 0.97 | 0.9097 |

Chosen tau = 0.73 (F0.5 = 0.9493). Prune sweep (drop candidates with blocking
total < X): X=0.6 keeps 89.9% at −0.0001; X=0.8 keeps 70.5% at −0.0009
(shipped); X=1.0 keeps 50.8% at −0.0033.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
