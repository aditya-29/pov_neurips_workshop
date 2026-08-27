#!/usr/bin/env python
"""Write committable copies of the ASL scored files with dataset text removed.

How2Sign is CC BY-NC 4.0, research use only, not redistributable, and the
README says not to republish anything derived from it. The scored rows carry the
English translations verbatim in `ground_truth`, so committing them as-is would
redistribute the corpus text.

Every `score_*` field and the model's own output are kept, so the published
numbers remain checkable; only the upstream text and the paths that identify
source clips are dropped.
"""
import json, sys
from pathlib import Path

DROP = ("ground_truth", "source_name", "source_path", "ground_truth_path")
BASE = Path("data/asl/20260824-002004-e1225b96852f/eval")


def main():
    n_files = 0
    for tag in sorted(p.name for p in BASE.iterdir() if p.is_dir()):
        src = BASE / tag / "scored.jsonl"
        if not src.exists():
            continue
        dst = BASE / tag / "scored.redacted.jsonl"
        kept = dropped = 0
        with open(dst, "w") as out:
            for line in open(src):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                for f in DROP:
                    if f in row:
                        row[f] = None
                        dropped += 1
                row["_redacted"] = "How2Sign text removed (CC BY-NC 4.0)"
                out.write(json.dumps(row) + "\n")
                kept += 1
        print(f"  {tag:9} {kept:4} rows -> {dst.name}  ({dropped} fields nulled)")
        n_files += 1
    print(f"{n_files} file(s) redacted")


if __name__ == "__main__":
    sys.exit(main())
