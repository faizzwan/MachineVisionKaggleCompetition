"""Run the real cell (Gemma 3 4B via local Ollama) over the samples and record evidence.
Usage: python gemma_run.py <project_dir>
Writes <project>/results/gemma_samples_run.json and .md
"""

import json
import platform
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

proj = Path(sys.argv[1])
sys.path.insert(0, str(proj))

import requests

from cell.config import CellConfig
from cell.sources import FolderSource
from cell.tree import Cell, build_tree
from cell.vlm import OllamaVLM

# Ground truth from make_samples.py (the images, not the file names, are what Gemma sees).
TRUTH = {
    "good_01": "PASS", "good_02": "PASS", "garbage_01": "PASS", "timeout_01": "PASS",
    "missing_screw_01": "REJECT (screw top-right)", "missing_screw_02": "REJECT (screw bottom-right)",
    "crack_01": "REJECT (surface center)", "dark_01": "REVIEW (bad photo)", "blurry_01": "REVIEW (bad photo)",
}


class TimedVLM(OllamaVLM):
    calls: list = []

    def inspect(self, image_path, region):
        t0 = time.monotonic()
        try:
            raw = super().inspect(image_path, region)
            self.calls.append({"image": Path(image_path).name, "region": region.name, "s": round(time.monotonic() - t0, 2), "raw": raw})
            return raw
        except Exception as e:
            self.calls.append({"image": Path(image_path).name, "region": region.name, "s": round(time.monotonic() - t0, 2), "error": str(e)})
            raise


runtime = proj / "results" / "runtime_gemma"
shutil.rmtree(runtime, ignore_errors=True)
TIMEOUT = float(sys.argv[2]) if len(sys.argv) > 2 else CellConfig.vlm_timeout_s
ONLY = sys.argv[3].split(",") if len(sys.argv) > 3 else None  # e.g. missing_screw_01,crack_01
cfg = replace(CellConfig(), runtime_dir=str(runtime), vlm_timeout_s=TIMEOUT)
version = requests.get(f"{cfg.ollama_url}/api/version", timeout=5).json()["version"]
vlm = TimedVLM(cfg)
cell = Cell(cfg, FolderSource(proj / "samples"), vlm)
if ONLY:
    cell.source.images = [p for p in cell.source.images if p.stem in ONLY]
tree = build_tree(cell)

warm_s = round(vlm.warm_up(), 1)
print(f"warm-up: {warm_s}s", flush=True)
parts = []
t_start = time.monotonic()
while cell.stopped is None:
    cell.part = None
    tree.tick()
    p = cell.part
    if p is None:
        continue
    row = {
        "image": p.image_path.name,
        "expected": TRUTH.get(p.image_path.stem, "?"),
        "decision": p.decision.value,
        "reason": p.reason,
        "decided_by": p.decided_by,
        "ai_seconds": p.vlm_latency_s,
        "regions": [{"region": r.region, "answer": r.answer, "description": r.description} for r in p.regions],
        "raw": p.vlm_raw,
    }
    parts.append(row)
    print(f"{row['image']:22s} {row['decision']:7s} expected {row['expected']:28s} {p.vlm_latency_s}s  {p.reason}", flush=True)

ok_calls = [c["s"] for c in vlm.calls if "raw" in c]
summary = {
    "model": cfg.model,
    "ollama_version": version,
    "machine": f"{platform.processor()} / {platform.system()} {platform.release()}",
    "timeout_s_per_closeup": TIMEOUT,
    "parts_run": ONLY or "all samples",
    "warm_up_seconds": warm_s,
    "total_minutes": round((time.monotonic() - t_start) / 60, 1),
    "model_calls": len(vlm.calls),
    "seconds_per_call_median": sorted(ok_calls)[len(ok_calls) // 2] if ok_calls else None,
    "seconds_per_call_min": min(ok_calls) if ok_calls else None,
    "seconds_per_call_max": max(ok_calls) if ok_calls else None,
}
out = proj / "results"
out.mkdir(exist_ok=True)
(out / "gemma_samples_run.json").write_text(json.dumps({"summary": summary, "parts": parts, "calls": vlm.calls}, indent=2), encoding="utf-8")

md = [f"# Real Gemma run on the samples ({time.strftime('%Y-%m-%d')})", "",
      "| | |", "|---|---|"] + [f"| {k} | {v} |" for k, v in summary.items()] + [
      "", "| Image | Expected | Cell decision | Reason |", "|---|---|---|---|"]
for r in parts:
    md.append(f"| {r['image']} | {r['expected']} | {r['decision']} | {r['reason'].replace('|', '/')} |")
md += ["", "## Every model answer", ""]
for c in vlm.calls:
    text = c.get("raw", c.get("error", "")).replace("\n", " / ")
    md.append(f"- **{c['image']} · {c['region']}** ({c['s']} s): {text}")
(out / "gemma_samples_run.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
