# business_entity_resolution

Pipeline for the ML Challenge 2026 Business Entity Resolution task.

| Stage | Status | Code |
| --- | --- | --- |
| Part 1: blocking / candidate generation | done | `src/business_candy_crush/` (Rust) |
| Part 2: matching model | done | `src/matching/` (Python + LightGBM) |
| Part 3: evaluation + packaging | eval built into both stages | `business_candy_crush eval` + `part2.train_model` |

## Reproduce

Requirements: Rust 1.98+ (`brew install rust` or rustup), Python 3.11+ with
`pip install -r requirements.txt`. About 9 GB RAM peak for Part 1, ~6 GB for
Part 2; no GPU needed.
The dataset is expected at `student_resource/dataset/` (two levels above this folder).

```bash
bash src/run_part1.sh
bash src/run_part2.sh
```

Part 1 builds the blocking index and writes `output/candidate_pairs.tsv` + the
scored candidate files under `work/`. Part 2 trains the matcher on the labelled
holdout and overwrites `output/matching_results.tsv` with the model's decisions,
then runs `utils/validate_submission.py`. Every step caches its artefacts under
`work/part2/`, so a rerun only redoes what changed (delete a cache dir to force it).

This builds the binary, then:

1. **Holdout eval.** It blocks the train holdout (S1 ids with `fnv1a64(id) % 10 == 0`, about 220k entities) against the *full* train S2+S3 corpus (10.3M records). It then writes recall@k, the oracle F0.5 ceiling and candidates-per-S1 to `work/val/eval.txt`, and missed pairs to `work/val/misses.tsv`. Inspect the misses with `python3 src/inspect_misses.py 50`.
2. **Test.** It blocks the full test set and writes `output/candidate_pairs.tsv` and a baseline `output/matching_results.tsv`, then runs `utils/validate_submission.py`.

Timing on an 11-core laptop: index build about 25 s; queries about 520/s at the default `--max-df 100000`. The full test set takes about 1 hour. `BLOCK_ARGS="--max-df 50000"` roughly doubles the speed at the cost of about 0.3 pt of recall.

## CLI

```
business_candy_crush block --s1 S1.tsv --s2 S2.tsv --s3 S3.tsv --out-dir DIR
    [--k 15] [--k-both 10] [--k-noaddr 5] [--k-group 5,10,10,15]
    [--max-df 100000] [--alpha 1,0.5,0.5,1,0.3] [--tau 2.28] [--val-only] [--limit N]
business_candy_crush eval --scored DIR/candidates_scored.tsv --gt GT.tsv --s1 S1.tsv [--misses F]
```

`block` writes three files:

- `candidate_pairs.tsv`: the submission file.
- `candidates_scored.tsv`: one `id:total:name:join:fuzzy:addr` entry per candidate. These scores are the first features for Part 2.
- `matching_results.tsv`: candidates with total score >= `--tau`.

## Blocking design (Part 1)

1. **Normalisation** (`normalize.rs`):
   - Indic scripts (Devanagari, Bengali, Gujarati, Tamil, Telugu, Kannada, Malayalam, …) are transliterated to Latin with one table, since the Unicode blocks share the ISCII layout.
   - Then NFKD, accent stripping and lowercasing.
   - Dots and apostrophes are deleted ("L.L.C." → "llc").
   - Web names are un-domained ("bethchapel.com" → "bethchapel").
   - Legal forms and street types are canonicalised, and US and Indian state names are mapped to their codes.
   - Leading zeros are stripped from numbers ("01018" → "1018", "05th" → "5th").
2. **Blocking keys**, in five channels, each hashed together with the record's country label:
   - name words;
   - joined-name keys (in order and word-sorted);
   - a fuzzy name key: consonant skeleton plus 4-character prefix and suffix, robust to typos and transliteration (प्राइवेट → "prbt" = private);
   - address words;
   - address skeletons.

   Because the country string is part of every hash, the index is partitioned per country for any label, including France, which never appears in training. There is no hard-coded country list. None of the 1.04M sampled true pairs crosses countries.
3. **Index.** An inverted index over S2 and S3 with idf weighting (df computed per country) and L2 normalisation per channel. Documents are renumbered by (country, city-like token) for cache locality.
4. **Query.** For each S1 record, only posting lists with df ≤ `max_df` are traversed. The candidate set is the union of these top-k lists:
   - the combined score;
   - name–address *agreement* `min(name/2, addr)`. This stops a namesake in another city from outranking a same-address record whose name is transliterated.
   - name score among records **without** an address;
   - each score group alone (name words, joined key, fuzzy name, address).

## Holdout results (220,441 S1 entities, 763,444 true pairs)

| | pair recall | all matches found | oracle F0.5 ceiling | avg candidates / S1 |
| --- | --- | --- | --- | --- |
| ALL | 0.9687 | 0.9064 | 0.9891 | 42.3 |
| India | 0.9479 | 0.8546 | 0.9807 | 44.5 |
| US | 0.9827 | 0.9409 | 0.9947 | 40.8 |

The *oracle F0.5 ceiling* is the macro F0.5 a perfect matcher would reach on these candidates: singletons count as 1.0. The naive score-threshold matcher reaches F0.5 = 0.689 (tau = 2.28); this is the baseline for Part 2.
