# syntax=docker/dockerfile:1
# Business Candy Crush — reproducible pipeline image (Linux).
#
#   docker build -t bcc .
#   docker run --rm -v "$PWD/dataset:/workspace/dataset" \
#       -v "$PWD/output:/workspace/output" -v "$PWD/work:/workspace/work" \
#       bcc make part1
#   docker run --rm -v "$PWD/dataset:/workspace/dataset" \
#       -v "$PWD/output:/workspace/output" -v "$PWD/work:/workspace/work" \
#       bcc make part2 DEVICE=cpu
#
# dataset/ + output/ + work/ are mounted, never baked in (too big for git or
# layer caches). CPU threads auto-cap at 80% (Makefile JOBS + part2.resources);
# pass --gpus all + DEVICE=gpu for full-GPU LightGBM (GPU-enabled base image
# and LightGBM GPU build required).

FROM rust:slim-bookworm AS rust-builder
WORKDIR /build
# rust-toolchain.toml pins the exact toolchain (1.98.1); rustup fetches it.
COPY code/business_entity_resolution/src/business_candy_crush/ ./crate/
RUN cargo build --release --manifest-path ./crate/Cargo.toml

FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/code/business_entity_resolution/src/matching
RUN apt-get update \
    && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY code/business_entity_resolution/requirements.txt ./code/business_entity_resolution/requirements.txt
RUN pip install --no-cache-dir -r ./code/business_entity_resolution/requirements.txt
COPY code/ ./code/
COPY utils/validate_submission.py ./utils/validate_submission.py
COPY Makefile README.md Documentation_template.md HANDOFF.md buildspec.yml ./
COPY tests/fixtures/ ./tests/fixtures/
COPY --from=rust-builder /build/crate/target/release/business_candy_crush \
    ./code/business_entity_resolution/src/business_candy_crush/target/release/business_candy_crush
CMD ["make", "help"]
