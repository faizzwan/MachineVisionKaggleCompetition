"""Operator review and override.

    python review.py                                  # parts waiting for a human
    python review.py all                              # every part with its final decision
    python review.py decide <part_id> PASS --by Sam   # override (any part, any decision)
"""

import argparse
import sys
from pathlib import Path

from cell import audit
from cell.config import CellConfig

RUNTIME = Path(CellConfig.runtime_dir)


def show(parts: list[dict]) -> None:
    if not parts:
        print("nothing to show")
    for p in parts:
        by = f"  (override by {p['override']['operator']})" if p["override"] else ""
        print(f"{p['part_id']}  {Path(p['image_path']).name:28s} {p['final_decision']:7s}{by}  {p['reason']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("all")
    d = sub.add_parser("decide")
    d.add_argument("part_id")
    d.add_argument("decision", choices=["PASS", "REJECT"])
    d.add_argument("--by", required=True, help="operator name, recorded in the audit trail")
    d.add_argument("--note", default="")
    args = ap.parse_args()

    parts = audit.load_parts(RUNTIME)
    if args.cmd == "decide":
        if args.part_id not in {p["part_id"] for p in parts}:
            print(f"unknown part {args.part_id}")
            return 1
        audit.record_override(RUNTIME, args.part_id, args.decision, args.by, args.note)
        print(f"{args.part_id} -> {args.decision} (by {args.by})")
    elif args.cmd == "all":
        show(parts)
    else:
        show([p for p in parts if p["final_decision"] == "REVIEW"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
