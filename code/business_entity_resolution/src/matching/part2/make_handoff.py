"""Assemble part2_outputs.zip for the Part-3 handoff (run AFTER predict_test +
prune_outputs have completed).

    python -m part2.make_handoff [--cutoff 0.8]

Steps:
  1. re-run the candidate prune pass if output/candidate_pairs.tsv is unpruned
     (> 750 MB means Aditya's original is still in place);
  2. verify the validator says PASS on the final files;
  3. copy the payloads into build/part2_outputs/ in workspace layout;
  4. zip with tar.exe (bsdtar, no Python-size limits) -> part2_outputs.zip.
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

PAYLOAD = [
    "output/matching_results.tsv",
    "output/candidate_pairs.tsv",
    "work/part2/model.txt",
    "work/part2/model_meta.json",
    "work/part2/train_report.txt",
    "work/part2/val_tag.txt",
    "HANDOFF.md",
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[5]))
    ap.add_argument("--cutoff", type=float, default=0.8)
    a = ap.parse_args(argv)
    root = Path(a.root)
    t0 = time.time()

    # 1. prune pass if needed
    cand = root / "output/candidate_pairs.tsv"
    if cand.stat().st_size > 750_000_000:
        print("candidate_pairs.tsv is unpruned -> running prune_outputs", flush=True)
        rc = subprocess.run([sys.executable, "-m", "part2.prune_outputs",
                             "--root", str(root), "--cutoff", str(a.cutoff)],
                            cwd=root / "code/business_entity_resolution/src/matching").returncode
        if rc != 0:
            print("prune_outputs FAILED", flush=True)
            return 1

    # 2. validator gate
    r = subprocess.run([sys.executable, str(root / "utils/validate_submission.py"),
                        "--matching", str(root / "output/matching_results.tsv"),
                        "--candidate", str(cand),
                        "--test-dir", str(root / "dataset/test")],
                       cwd=root, capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print("VALIDATOR DID NOT PASS -> refusing to build handoff zip", flush=True)
        print(r.stderr[-1500:])
        return 1

    # 3. collect payload
    out = root / "build/part2_outputs"
    if out.exists():
        shutil.rmtree(out)
    for rel in PAYLOAD:
        src = root / rel
        if not src.exists():
            print(f"missing {rel}")
            return 1
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    vmeta = sorted((root / "work/part2").glob("val_meta_*.npz"))
    big = [v for v in vmeta if v.stat().st_size > 5_000_000]
    for v in big:
        shutil.copy2(v, out / "work/part2" / v.name)
    (out / "README.txt").write_text(
        "Business Candy Crush - Part 2 handoff artefacts.\n"
        "Unzip OVER the workspace root that contains dataset/ and the repo (layout in HANDOFF.md, section 2).\n"
        "Read HANDOFF.md (also in this zip and at the repo root) for your tasks.\n"
        "The scored submission file is output/matching_results.tsv.\n")

    # 4. zip via bsdtar (handles >2 GB, no python zipfile limits)
    zpath = root / "part2_outputs.zip"
    if zpath.exists():
        zpath.unlink()
    rc = subprocess.run(["tar.exe", "-a", "-c", "-f", str(zpath),
                         "-C", str(root / "build"), "part2_outputs"],
                        cwd=root).returncode
    if rc != 0:
        print("tar failed", flush=True)
        return rc
    print(f"wrote {zpath}  ({zpath.stat().st_size/1e6:.0f} MB, {time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
