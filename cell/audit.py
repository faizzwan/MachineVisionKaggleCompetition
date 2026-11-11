"""Append-only audit trail. Every part, every AI answer, every operator override."""

import json
import time
from dataclasses import asdict
from pathlib import Path

from .tree import PartRecord


def _append(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit_path(runtime: Path) -> Path:
    return Path(runtime) / "audit.jsonl"


def overrides_path(runtime: Path) -> Path:
    return Path(runtime) / "overrides.jsonl"


def record_part(runtime: Path, part: PartRecord, model_name: str) -> dict:
    entry = asdict(part)
    entry["image_path"] = str(part.image_path)
    entry["decision"] = part.decision.value if part.decision else None
    entry["lane"] = part.lane or "held (line stopped before actuation)"
    entry["model"] = model_name
    entry["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _append(audit_path(runtime), entry)
    return entry


def record_override(runtime: Path, part_id: str, decision: str, operator: str, note: str = "") -> dict:
    entry = {
        "part_id": part_id,
        "decision": decision,
        "operator": operator,
        "note": note,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _append(overrides_path(runtime), entry)
    return entry


def load_parts(runtime: Path) -> list[dict]:
    """Audit entries with the latest operator override (if any) applied as `final_decision`."""
    overrides = {o["part_id"]: o for o in _read(overrides_path(runtime))}
    parts = _read(audit_path(runtime))
    for p in parts:
        o = overrides.get(p["part_id"])
        p["override"] = o
        p["final_decision"] = o["decision"] if o else p["decision"]
    return parts
