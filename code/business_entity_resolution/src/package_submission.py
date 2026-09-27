"""Build the submission zip in the exact challenge layout (stdlib only, portable).

    python src/package_submission.py [--root WORKSPACE_ROOT]
                                     [--name ZIP_NAME]
                                     [--compress-level 0-9]

Layout inside the zip:
    output/matching_results.tsv
    output/candidate_pairs.tsv
    code/business_entity_resolution/src/   (excluding target/, __pycache__/, .git/, *.pyc)
    code/business_entity_resolution/README.md
    code/business_entity_resolution/requirements.txt
    Documentation_template.md

Free of work/ caches and dataset/ copies. Used by `make package` / `make part3`.
"""

import argparse
import os
import sys
import zipfile

SKIP_DIRS = {"target", "__pycache__", ".git"}

TOP_FILES = [
    "output/matching_results.tsv",
    "output/candidate_pairs.tsv",
    "code/business_entity_resolution/README.md",
    "code/business_entity_resolution/requirements.txt",
    "Documentation_template.md",
]

SRC_DIR = "code/business_entity_resolution/src"


def collect(root):
    """Yield (disk_path, arc_name) pairs in deterministic order."""
    for rel in TOP_FILES:
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            raise SystemExit(f"missing required file: {p} (run part1+part2 first)")
        yield p, rel.replace(os.sep, "/")
    base = os.path.join(root, SRC_DIR)
    if not os.path.isdir(base):
        raise SystemExit(f"missing source dir: {base}")
    for dp, dn, fn in os.walk(base):
        dn[:] = sorted(d for d in dn if d not in SKIP_DIRS)
        for f in sorted(fn):
            if f.endswith(".pyc"):
                continue
            p = os.path.join(dp, f)
            yield p, os.path.relpath(p, root).replace(os.sep, "/")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the submission zip.")
    ap.add_argument("--root", default=os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        help="workspace root (default: derived from this file's location)")
    ap.add_argument("--name", default="Business_Candy_Crush_submission.zip")
    ap.add_argument("--compress-level", type=int, default=3)
    a = ap.parse_args(argv)
    zp = os.path.join(a.root, a.name)
    entries = list(collect(a.root))
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=a.compress_level) as z:
        for src, arc in entries:
            z.write(src, arc)
            print(f"added {arc}", flush=True)
    size_mb = os.path.getsize(zp) / 1e6
    print(f"wrote {zp} ({len(entries)} entries, {size_mb:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
