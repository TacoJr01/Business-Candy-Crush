# ======================================================================
# Business Candy Crush — central Makefile (ML Challenge 2026, entity resolution)
#
# Run from the workspace root (the folder containing dataset/, code/, ...).
#
#   Windows: install GNU Make 4.x (`choco install make`), Python 3.11+ and
#            Rust 1.98+. Recipes are single portable commands (no sh-only
#            syntax), so both cmd.exe and Git Bash work as make's shell.
#   UNIX:    GNU Make + Python 3.11+ + Rust toolchain (see rust-toolchain.toml).
#
# Common uses:
#   make setup            install Python dependencies
#   make all              full pipeline: blocking + matching + validate
#   make part1            Rust blocking only (val holdout + full test)
#   make part2            Python matcher only (needs Part 1 outputs)
#   make validate         run utils/validate_submission.py (must print PASS)
#   make smoke            fast wiring check (limited val block, own out-dir)
#   make prune            re-run test inference with CUTOFF pruning
#   make clean            remove output/ + work/ caches (keeps dataset/)
#
# Knobs (command line or environment):
#   make all JOBS=8 DEVICE=gpu CUTOFF=0.8 LIMIT=5000 BLOCK_ARGS="--max-df 50000"
#   JOBS    CPU worker/thread budget (default: 80% of logical CPUs).
#   DEVICE  auto|cpu|gpu — LightGBM device (default auto; gpu is opt-in and
#           runs uncapped, CPU-side work stays under the JOBS budget).
# ======================================================================

.DEFAULT_GOAL := help

# ---------- OS detection ----------
ifeq ($(OS),Windows_NT)
DETECTED_OS := Windows
PYTHON ?= python
EXE := .exe
RM_R := powershell -NoProfile -NonInteractive -Command "Remove-Item -Recurse -Force --"
else
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
DETECTED_OS := Linux
else ifeq ($(UNAME_S),Darwin)
DETECTED_OS := macOS
else
DETECTED_OS := $(UNAME_S)
endif
PYTHON ?= python3
EXE :=
RM_R := rm -rf
endif

# ---------- resource budget (<=80% of CPUs; GPU opt-in runs full) ----------
JOBS ?= $(shell $(PYTHON) -c "import os; print(max(1, int((os.cpu_count() or 4) * 0.8)))")
export RAYON_NUM_THREADS := $(JOBS)
export PART2_MAX_WORKERS := $(JOBS)
export LGB_NUM_THREADS := $(JOBS)
export OMP_NUM_THREADS := $(JOBS)
export MKL_NUM_THREADS := $(JOBS)
export OPENBLAS_NUM_THREADS := $(JOBS)
export NUMEXPR_NUM_THREADS := $(JOBS)

DEVICE ?= auto
ifneq ($(DEVICE),auto)
export PART2_DEVICE := $(DEVICE)
endif

# ---------- layout ----------
ROOT := $(CURDIR)
PKG := code/business_entity_resolution
SRC := $(PKG)/src
MATCHING := $(SRC)/matching
MANIFEST := $(SRC)/business_candy_crush/Cargo.toml
BIN := $(SRC)/business_candy_crush/target/release/business_candy_crush$(EXE)
REQ := $(PKG)/requirements.txt
DATA ?= $(ROOT)/dataset
BLOCK_ARGS ?=
CUTOFF ?= 0.8
LIMIT ?= 5000

export PYTHONPATH := $(ROOT)/$(MATCHING)

.PHONY: help os-info setup bin build part1 part1-val part1-test eval \
	part2 part2-val part2-train part2-predict validate prune smoke \
	part3 package clean clean-cache show-report

help:
	@echo "Business Candy Crush [$(DETECTED_OS)] PYTHON=$(PYTHON) JOBS=$(JOBS) DEVICE=$(DEVICE)"
	@echo ""
	@echo "  make setup       install Python deps (pip install -r $(REQ))"
	@echo "  make all         part1 + part2 + validate"
	@echo "  make part1       Rust blocking: val holdout metrics + test candidates"
	@echo "  make part2       matcher: build_val + train_model + predict_test"
	@echo "  make validate    submission validator (expect PASS)"
	@echo "  make smoke       fast wiring check: val block limited to LIMIT=$(LIMIT)"
	@echo "  make prune       re-run test inference pruning blocking-total < CUTOFF=$(CUTOFF)"
	@echo "  make part3       evaluation + packaging: validate + prune + submission zip"
	@echo "  make clean       remove output/ and work/ (keeps dataset/)"
	@echo "  make show-report print results/part2/train_report.txt"
	@echo ""
	@echo "Knobs: JOBS=N DEVICE=auto|cpu|gpu DATA=path BLOCK_ARGS=... CUTOFF=x LIMIT=n"

os-info:
	@echo "OS=$(DETECTED_OS) PYTHON=$(PYTHON) JOBS=$(JOBS) DEVICE=$(DEVICE) BIN=$(BIN)"

setup:
	$(PYTHON) -m pip install -r "$(REQ)"
	@echo "Also needs Rust 1.98+ for Part 1 (see $(SRC)/business_candy_crush/rust-toolchain.toml)."

# ---------- Part 1: Rust blocking ----------
bin:
	cargo build --release --manifest-path "$(MANIFEST)"

build: bin

part1-val: bin
	"$(BIN)" block --s1 "$(DATA)/train/train_source1.tsv" --s2 "$(DATA)/train/train_source2.tsv" --s3 "$(DATA)/train/train_source3.tsv" --out-dir "$(ROOT)/work/val" $(BLOCK_ARGS) --jobs $(JOBS) --val-only
	"$(BIN)" eval --scored "$(ROOT)/work/val/candidates_scored.tsv" --gt "$(DATA)/train/train_ground_truth.tsv" --s1 "$(DATA)/train/train_source1.tsv" --misses "$(ROOT)/work/val/misses.tsv" > "$(ROOT)/work/val/eval.txt"
	@echo "holdout metrics -> $(ROOT)/work/val/eval.txt"

part1-test: bin
	"$(BIN)" block --s1 "$(DATA)/test/test_source1.tsv" --s2 "$(DATA)/test/test_source2.tsv" --s3 "$(DATA)/test/test_source3.tsv" --out-dir "$(ROOT)/output" $(BLOCK_ARGS) --jobs $(JOBS)
	$(PYTHON) -c "import shutil; shutil.move('$(ROOT)/output/candidates_scored.tsv', '$(ROOT)/work/test_candidates_scored.tsv')"

part1: part1-val part1-test

eval:
	"$(BIN)" eval --scored "$(ROOT)/work/val/candidates_scored.tsv" --gt "$(DATA)/train/train_ground_truth.tsv" --s1 "$(DATA)/train/train_source1.tsv" --misses "$(ROOT)/work/val/misses.tsv" > "$(ROOT)/work/val/eval.txt"

# ---------- Part 2: Python matcher (needs Part 1 outputs) ----------
part2-val:
	$(PYTHON) -m part2.build_val --root "$(ROOT)" --jobs $(JOBS)

part2-train:
	$(PYTHON) -m part2.train_model --root "$(ROOT)" --jobs $(JOBS) --device $(DEVICE)

part2-predict:
	$(PYTHON) -m part2.predict_test --root "$(ROOT)" --jobs $(JOBS)

part2: part2-val part2-train part2-predict

# ---------- evaluation / packaging ----------
all: part1 part2 validate

validate:
	$(PYTHON) "$(ROOT)/utils/validate_submission.py" --matching "$(ROOT)/output/matching_results.tsv" --candidate "$(ROOT)/output/candidate_pairs.tsv" --test-dir "$(DATA)/test"

prune:
	$(PYTHON) -m part2.predict_test --root "$(ROOT)" --jobs $(JOBS) --cand-min-total $(CUTOFF)

smoke: bin
	"$(BIN)" block --s1 "$(DATA)/train/train_source1.tsv" --s2 "$(DATA)/train/train_source2.tsv" --s3 "$(DATA)/train/train_source3.tsv" --out-dir "$(ROOT)/work/smoke" $(BLOCK_ARGS) --jobs $(JOBS) --val-only --limit $(LIMIT)
	@echo "smoke block ok -> $(ROOT)/work/smoke/ (val caches untouched)"

# ---------- Part 3: evaluation + packaging (needs Part 1+2 outputs) ----------
part3: validate prune package

package:
	$(PYTHON) "$(SRC)/package_submission.py" --root "$(ROOT)"

show-report:
	$(PYTHON) -c "print(open('results/part2/train_report.txt', encoding='utf-8').read())"

# ---------- cleaning (dataset/ is never touched) ----------
clean:
	-$(RM_R) "$(ROOT)/output" "$(ROOT)/work"

clean-cache:
	-$(RM_R) "$(ROOT)/work"
