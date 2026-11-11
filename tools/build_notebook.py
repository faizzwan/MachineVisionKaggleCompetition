"""Build a self-contained Kaggle notebook from the project sources.

Usage: python build_notebook.py <project_dir> <out.ipynb>

Every project file is embedded with %%writefile, so the notebook never drifts from the code
and runs on Kaggle with no dataset attached (only `pip install py_trees`).
"""

import sys
from pathlib import Path

import nbformat as nbf

proj = Path(sys.argv[1])
out = Path(sys.argv[2])

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


def writefile(rel):
    body = (proj / rel).read_text(encoding="utf-8")
    code(f"%%writefile {rel}\n{body}")


md("""
# Trusted Inspection Cell: a behaviour tree that decides, a local VLM that looks

**Challenge idea:** #2 *Hybrid Logic & VLM Orchestrator* applied to #6 *Camera-based quality & part inspection*, with the
#3 fault-recovery idea in the form of a circuit breaker and a human-review fallback.

An offline part-inspection cell. A `py_trees` **behaviour tree** runs the line: e-stop, image gate, circuit breaker,
decision rules, diverter. A local **vision-language model** (Gemma 3 4B through Ollama) answers one narrow question about
one close-up the tree chose. The model never moves anything. Each answer goes through a deterministic validator,
fixed rules turn validated answers into PASS / REJECT / REVIEW, and every failure falls back to a human.

| Judging area | Where to look in this notebook |
|---|---|
| 1. Deterministic logic & AI handoffs (30) | section 3 (tree + data/control flow), 4 (decisions with traces), 5a (validator), 5d (override) |
| 2. On-device / offline (25) | section 8 (topology, connectivity drops, live local model), 5c (model server down) |
| 3. Impact & execution (25) | sections 4 to 6 (pipeline, failure handling, audit trail, tests), 8 (latency) |
| 4. Writeup & trade-offs (20) | sections 7 and 8 (why this model, prompt design, topology and control structure) |

**Read this first, because it affects how to read the results.** Sections 4 to 7 run with `MockVLM`, a stand-in model that
reads its answer from the *file name*. It is not an inspector. It exists so that every path through the tree
(pass, reject, bad image, garbage answer, timeout, e-stop, dead model) can be exercised in seconds and checked by tests.
Section 8 runs the real model when an Ollama server is reachable, and says so when it is not.
""")

md("## 1. Setup")
code("""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "py_trees>=2.2", "requests", "pytest"], check=False)
import os
os.makedirs("cell", exist_ok=True)
os.makedirs("tests", exist_ok=True)
open("cell/__init__.py", "w").close()
""")

md("""
## 2. The code

The project files are written out below so the notebook is self-contained. They are the same files as in the project repo.
`cell/spec.py` (what a good part is and where to look), `cell/validator.py` (what the AI may say) and `cell/tree.py`
(who decides what) are the three worth reading.
""")
for rel in [
    "cell/__init__.py",
    "cell/config.py",
    "cell/spec.py",
    "cell/validator.py",
    "cell/vlm.py",
    "cell/quality.py",
    "cell/sources.py",
    "cell/overlay.py",
    "cell/tree.py",
    "cell/audit.py",
    "make_samples.py",
    "tests/test_cell.py",
]:
    if rel == "cell/__init__.py":
        continue  # created empty in the setup cell
    writefile(rel)

md("""
## 3. The behaviour tree and the exact flow of data and control

The tree is the deterministic part. The only AI step is `Inspect regions with VLM`.

| Step | Owner | Input | Output | On failure |
|---|---|---|---|---|
| E-stop clear? | rules | `runtime/ESTOP` file | continue / halt | line halts |
| Acquire part | rules | camera frame or file | image path | line stops (no parts) |
| Image quality OK? | rules | image | brightness, sharpness | human review, **AI not called** |
| Model healthy? | rules | failure counter | go / skip | human review (degraded mode) |
| **Handoff:** crop + prompt | rules | spec region | one close-up + one fixed question | n/a |
| Inspect region | **AI** (Gemma 3 4B, localhost) | close-up | raw text, **untrusted** | timeout/down: human review |
| **Handback:** validate | rules | raw text | `RegionResult` (allowed word + description) | invalid: human review |
| Apply decision rules | rules | validated results | PASS / REJECT / REVIEW + reason | n/a |
| E-stop clear? (pre-actuation) | rules | ESTOP file | continue / hold | decision logged, gate **not** moved |
| Actuate diverter | rules | decision | lane | n/a |
| Audit | rules | everything above | `audit.jsonl` line | n/a |
""")
code("""
import py_trees
from dataclasses import replace
from pathlib import Path

from cell.config import CellConfig
from cell.tree import Cell, build_tree
from cell.sources import FolderSource
from cell.vlm import MockVLM

import make_samples
make_samples.main(Path("samples"))

_demo = Cell(CellConfig(), FolderSource(Path("samples")), MockVLM())
print(py_trees.display.unicode_tree(build_tree(_demo).root))
""")

md("""
Synthetic test parts: a metal plate with four screws. `good_*` are fine, `missing_screw_*` and `crack_*` are defective,
`dark_*` and `blurry_*` are bad photos, and `garbage_*` / `timeout_*` make the (mock) model misbehave.
""")
code("""
import matplotlib.pyplot as plt
from PIL import Image

files = sorted(Path("samples").glob("*.png"))
fig, axes = plt.subplots(1, len(files), figsize=(2.2 * len(files), 2.6))
for ax, f in zip(axes, files):
    ax.imshow(Image.open(f)); ax.set_title(f.stem, fontsize=8); ax.axis("off")
plt.tight_layout(); plt.show()
""")

md("""
## 4. Run the line (mock model)

One tick of the tree is one part. Every decision carries the reason, which tree leaves ran, and the lane the diverter chose.
""")
code("""
import shutil, pandas as pd
from cell import audit

shutil.rmtree("runtime", ignore_errors=True)
cfg = replace(CellConfig(), runtime_dir="runtime")
cell = Cell(cfg, FolderSource(Path("samples")), MockVLM())
tree = build_tree(cell)

parts = []
while cell.stopped is None:
    cell.part = None
    tree.tick()
    if cell.part is not None:
        parts.append(cell.part)
        audit.record_part(cell.runtime, cell.part, "mock")

pd.set_option("display.max_colwidth", 90)
pd.DataFrame({
    "image": [p.image_path.name for p in parts],
    "decision": [p.decision.value for p in parts],
    "lane": [p.lane for p in parts],
    "decided by": [p.decided_by for p in parts],
    "reason": [p.reason for p in parts],
})
""")

md("""
Every AI answer and every decision lands in an append-only audit log. Below, the leaves visited for three kinds of part:
the AI path for a good part, an early stop for a defect, and a bad photo that never reaches the model.
""")
code("""
by_name = {p.image_path.stem: p for p in parts}
for name in ["good_01", "missing_screw_01", "dark_01"]:
    p = by_name[name]
    print(f"{name:18s} {p.decision.value:7s} " + " -> ".join(p.trace))
    print(f"{'':18s} AI calls: {len(p.vlm_raw)}\\n")
""")

md("""
### Where the AI looked

Green: the model said OK. Red: the model found a defect. Grey: not checked (a defect stops the check early).
Because the *spec* picks the regions, an operator can always see what the AI was asked and where.
""")
code("""
from cell.overlay import draw_regions
from cell.spec import DEFECT_ANSWER

def outcome(p):
    return {r.region: r.answer == DEFECT_ANSWER[r.kind] for r in p.regions}

show = [p for p in parts if p.regions]
fig, axes = plt.subplots(1, len(show), figsize=(2.8 * len(show), 3.2))
for ax, p in zip(axes, show):
    ax.imshow(draw_regions(p.image_path, outcome(p)))
    ax.set_title(f"{p.image_path.stem}\\n{p.decision.value}", fontsize=8); ax.axis("off")
plt.tight_layout(); plt.show()
""")

md("""
## 5. What can go wrong, and what the cell does about it

### 5a. The model's answer is never trusted as-is
The validator needs a description *and* exactly one allowed word for that kind of region. Anything else is rejected and the
part goes to a human.
""")
code("""
from cell.spec import PART_SPEC
from cell.validator import ValidationError, parse_region_answer

screw = next(r for r in PART_SPEC if r.kind == "screw")
surface = next(r for r in PART_SPEC if r.kind == "surface")
cases = [
    ("good answer",            "A screw head is installed.\\nANSWER: SCREW", screw),
    ("empty reply",            "", screw),
    ("no ANSWER line",         "The part is fine.", screw),
    ("word not allowed",       "A screw.\\nANSWER: MAYBE", screw),
    ("wrong region vocabulary","A clean surface.\\nANSWER: SCREW", surface),
    ("answer, no evidence",    "ANSWER: SCREW", screw),
]
rows = []
for label, raw, region in cases:
    try:
        r = parse_region_answer(raw, region)
        rows.append((label, "ACCEPTED", f"{r.answer}: {r.description}"))
    except ValidationError as e:
        rows.append((label, "REJECTED -> human", str(e)))
pd.DataFrame(rows, columns=["case", "validator", "detail"])
""")

md("""
### 5b. E-stop during the slow AI step
The e-stop is checked again just before the diverter. The decision is logged, but the gate does not move.
""")
code("""
class EStopWhileThinking(MockVLM):
    def inspect(self, image_path, region):
        c.estop_file.parent.mkdir(parents=True, exist_ok=True)
        c.estop_file.touch()          # operator hits e-stop while the model is busy
        return super().inspect(image_path, region)

c = Cell(replace(CellConfig(), runtime_dir="runtime_estop"), FolderSource(Path("samples")), EStopWhileThinking())
c.source.images = [Path("samples/crack_01.png")]
t = build_tree(c); c.part = None; t.tick()
print("decision:", c.part.decision.value, "| lane:", c.part.lane, "(None = diverter never moved)")
shutil.rmtree("runtime_estop", ignore_errors=True)
""")

md("""
### 5c. Dead model: the circuit breaker
A failing model can cost a full timeout per part. After 3 failures in a row the cell stops waiting on it,
sends parts to human review, and probes the model every 5th part.
""")
code("""
calls = []
class DeadModel(MockVLM):
    def inspect(self, image_path, region):
        calls.append(image_path)
        return MockVLM.inspect(self, Path("samples/timeout_01.png"), region)

c = Cell(replace(CellConfig(), runtime_dir="runtime_cb", max_consecutive_vlm_failures=2, probe_every_n_parts=3),
         FolderSource(Path("samples")), DeadModel())
c.source.images = [Path("samples/good_01.png")] * 9
t = build_tree(c)
ps = []
while c.stopped is None:
    c.part = None; t.tick()
    if c.part is not None: ps.append(c.part)
print(f"{len(ps)} parts, {len(calls)} model calls (naive loop would make {len(ps)})")
print("decisions:", {p.decision.value for p in ps})
print("part 3 reason:", ps[2].reason)
shutil.rmtree("runtime_cb", ignore_errors=True)
""")

md("""
### 5d. Operator override
The AI-path decision is never overwritten. The override is a separate record with the operator's name.
""")
code("""
garbage = next(p for p in audit.load_parts(Path("runtime")) if "garbage" in p["image_path"])
print("before:", garbage["final_decision"], "|", garbage["reason"])
audit.record_override(Path("runtime"), garbage["part_id"], "PASS", "Sam", "checked by hand")
after = next(p for p in audit.load_parts(Path("runtime")) if p["part_id"] == garbage["part_id"])
print("after: ", after["final_decision"], "| AI path said:", after["decision"], "| override by:", after["override"]["operator"])
""")

md("## 6. Safety tests\n\nThe project's test suite, run against the code above.")
code("""
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--color=no", "-p", "no:cacheprovider", "--rootdir", ".", "tests"],
                   capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."})
print(r.stdout[-1500:], r.stderr[-500:])
""")

md("""
## 7. Why the AI only sees close-ups (what we learned about the model)

These come from the author's runs of Gemma 3 4B on a laptop CPU through Ollama, on the project's synthetic parts. They are
recorded observations, not something this notebook re-runs, and the part images are synthetic, so treat them as
design evidence rather than a benchmark.

| Approach | Result with Gemma 3 4B on CPU |
|---|---|
| Whole part, "is it defective?" with confidence | Said a part with a **missing screw** was fine at **95% confidence**; missed the crack |
| Whole part, check each corner | Said all 4 screws present on every image |
| Close-up per corner, forced JSON answer | Same answer for every crop, right or wrong |
| **Close-up per region, describe first, then answer** | **Used in the final cell; measured below** |

So: self-reported confidence is not used anywhere, the deterministic side decides where to look, and the model must describe
before it answers.

### Measured run of the final cell (real Gemma 3 4B via local Ollama, 2026-11-12)

Hardware: Intel laptop, Windows 11, 7.7 GB RAM, CPU only. Limit: 600 s per close-up. Full log: `results/gemma_samples_run.md`.

| Part | Expected | Cell decision | Model's words at the deciding region |
|---|---|---|---|
| good_01 | PASS | **PASS** (9/9 regions OK) | "smooth and uniform with no visible lines, cracks, scratches, or marks" |
| missing_screw_01 | REJECT top-right | **REJECT at screw top-right** (2 close-ups) | "a circular, black hole in the center" |
| crack_01 | REJECT | **REJECT at surface top** | "several fine lines and scratches" |

3/3 decisions correct; 16/16 answers passed the validator; 10/10 screw answers correct. The crack was flagged in the
region holding its upper tip rather than the centre, so treat that one as a correct decision with weak evidence. Close-ups
took 12-238 s (median 138 s) because this laptop is short of RAM. With the default 120 s limit, every call timed out, and
the cell sent all 9 parts to human review via the circuit breaker: the fault path, working on real hardware.
""")

md("""
### Trade-offs

| Choice | Why | What we gave up |
|---|---|---|
| Gemma 3 4B (open weights, Ollama) | Takes images, runs quantised in about 3.3 GB on an 8 GB laptop CPU, no cloud, explains its answer in words | Slower than a trained detector |
| VLM, not a trained CNN | No labelled defect set; a new part is a spec edit; plain-English reasons | Throughput; accuracy must be proven per part |
| Behaviour tree, not a state machine | The Selector gives "AI path, else human" structurally; reads as a diagram; e-stop re-checked every tick; standard in robot stacks | One extra library |
| Model confidence ignored | It was meaningless (95% on a wrong answer) | Decisions rest on fixed rules only |

## 8. Deployment: on-device, offline

| Node | Runs | Network |
|---|---|---|
| Inspection workstation | behaviour tree, Gemma 3 4B via Ollama, dashboard server | model at `localhost` only |
| Operator tablet (optional) | browser on the dashboard | local LAN only |

**Why fully local, not hybrid:** the model makes one narrow judgement on every part. A cloud call would add latency and a
Wi-Fi dependency to that decision, and part images can be customer IP. **Connectivity drops:** plant Wi-Fi down changes
nothing; a tablet disconnect leaves the line running with all its safety behaviour, because the dashboard is a separate
process; a model server that dies trips the circuit breaker shown in 5c.

### Real model run

If an Ollama server with `gemma3:4b` is reachable, this cell runs the same tree with the real model on the missing-screw
part (2 close-ups). The full measured run is in section 7. On Kaggle there is normally no Ollama, so it skips cleanly. Nothing here calls a
hosted API: the only network address is `localhost`.
""")
code("""
import requests
from cell.vlm import OllamaVLM

# 600 s per close-up: on an 8 GB CPU-only laptop a new close-up can take minutes.
cfg = replace(CellConfig(), runtime_dir="runtime_live", vlm_timeout_s=600)
try:
    tags = requests.get(f"{cfg.ollama_url}/api/tags", timeout=2).json()
    have = any(m["name"].startswith(cfg.model) for m in tags.get("models", []))
except Exception:
    have = False

if not have:
    print(f"No local Ollama with {cfg.model} found: skipping the live run.")
    print("To run it on your machine:  ollama pull gemma3:4b   then re-run this notebook locally.")
else:
    vlm = OllamaVLM(cfg)
    live = Cell(cfg, FolderSource(Path("samples")), vlm)
    live.source.images = [Path("samples/missing_screw_01.png")]  # 2 close-ups: quick
    print(f"warm-up: {vlm.warm_up():.0f}s")
    lt = build_tree(live)
    while live.stopped is None:
        live.part = None; lt.tick()
        if live.part is not None:
            p = live.part
            print(f"{p.image_path.name:22s} {p.decision.value:7s} {p.vlm_latency_s}s  {p.reason}")
            for r in p.regions:
                print(f"    {r.region:18s} {r.answer:7s} {r.description}")
    shutil.rmtree("runtime_live", ignore_errors=True)
""")

md("""
## 9. Limits and what comes next

* Regions are fractions of the frame, so the cell assumes a fixed camera and a jig. The coordinates in `cell/spec.py` fit the
  synthetic 512 px parts and must be re-fitted to real photos.
* The sample parts are synthetic. Results on real parts have not been measured here.
* 12-238 s per close-up on an 8 GB CPU-only laptop. More RAM or a GPU brings that to seconds; the tree and rules do not change.
* The real-model run covers 3 synthetic parts (16 close-ups): evidence the pipeline works, not an accuracy benchmark.
* The diverter is simulated. `ActuateDiverter` is the one leaf to swap for a PLC write or robot motion routine.
* `MockVLM` is not an inspector. It only proves the control logic.

The operator dashboard (`dashboard.py`, Streamlit) shows the live part, the review queue with override buttons, the history,
and a software e-stop. It runs as a separate process and shares only files with the line, so a dashboard crash cannot
affect the line's safety behaviour.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
