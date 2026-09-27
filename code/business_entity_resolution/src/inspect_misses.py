"""Join work/val/misses.tsv with the raw records to eyeball blocking failures.
Usage: python3 inspect_misses.py [N]   (run from student_resource/)"""
import sys
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles break on Indic scripts
except Exception:
    pass
import sys
from collections import defaultdict

n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
miss = [l.rstrip("\n").split("\t") for l in open("work/val/misses.tsv")][1:][:n]
need = {x for row in miss for x in row[:2]}
rec = {}
for f in ("train_source1", "train_source2", "train_source3"):
    for line in open(f"dataset/train/{f}.tsv", encoding="utf-8"):
        i, _, rest = line.partition("\t")
        if i in need:
            rec[i] = rest.rstrip("\n").split("\t")
by = defaultdict(list)
for s1, m, *_ in miss:
    by[s1].append(m)
for s1, ms in by.items():
    print("S1  ", rec.get(s1))
    for m in ms:
        print("  miss", m, rec.get(m))
