"""Render gallery images for the Kaggle writeup from the real project code (mock model).
Usage: python make_media.py <project_dir>"""

import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

proj = Path(sys.argv[1])
sys.path.insert(0, str(proj))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

import make_samples
from cell.config import CellConfig
from cell.overlay import draw_regions
from cell.sources import FolderSource
from cell.spec import DEFECT_ANSWER
from cell.tree import Cell, build_tree
from cell.vlm import MockVLM

media = proj / "media"
media.mkdir(exist_ok=True)
tmp = Path(tempfile.mkdtemp())
make_samples.main(tmp / "samples")
cfg = replace(CellConfig(), runtime_dir=str(tmp / "runtime"))
cell = Cell(cfg, FolderSource(tmp / "samples"), MockVLM())
tree = build_tree(cell)
parts = {}
while cell.stopped is None:
    cell.part = None
    tree.tick()
    if cell.part is not None:
        parts[cell.part.image_path.stem] = cell.part

COL = {"PASS": "#16a34a", "REJECT": "#dc2626", "REVIEW": "#d97706"}

# ---- 1. overlays: where the AI looked, and what the rules decided -----------------------
show = [
    ("good_01", "Good part"),
    ("missing_screw_02", "Missing screw"),
    ("crack_01", "Crack"),
    ("dark_01", "Dark photo"),
    ("garbage_01", "Model rambles"),
]
fig, axes = plt.subplots(1, len(show), figsize=(3 * len(show), 3.9))
for ax, (stem, title) in zip(axes, show):
    p = parts[stem]
    oc = {r.region: r.answer == DEFECT_ANSWER[r.kind] for r in p.regions}
    ax.imshow(draw_regions(p.image_path, oc))
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(f"{p.decision.value}  ->  {p.lane}\n{len(p.vlm_raw)} AI call(s)", fontsize=9, color=COL[p.decision.value])
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Where the AI looked and what the fixed rules decided (mock model; green = OK, red = defect, grey = not checked)", fontsize=10)
plt.tight_layout()
fig.savefig(media / "1_overlays.png", dpi=150)
plt.close(fig)

# ---- 2. who decides what: the tree ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")
DET, AI, HUM = "#dbeafe", "#fde68a", "#fed7aa"


def box(x, y, w, h, text, fc, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#334155", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, fontweight="bold" if bold else "normal")


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#475569"))


box(0.2, 5.4, 9.6, 0.55, "Inspection cycle  (Sequence)", "#e2e8f0", True)
steps = [
    (0.2, "E-stop\nclear?", DET), (1.45, "Acquire\npart", DET), (2.7, "Classify (Selector)", "#e2e8f0"),
    (7.0, "E-stop clear?\n(pre-actuation)", DET), (8.5, "Actuate\ndiverter", DET),
]
for x, t, c in steps:
    w = 3.9 if t.startswith("Classify") else (1.4 if x > 8 else 1.15 if x < 2 else 1.5)
    box(x, 4.2, w, 0.75, t, c, t.startswith("Classify"))
    arrow(x + w / 2, 5.4, x + w / 2, 4.95)
box(2.7, 2.4, 3.0, 1.4, "AI path  (Sequence)", "#e2e8f0", True)
arrow(4.2, 4.2, 4.2, 3.8)
box(6.0, 2.4, 1.9, 1.4, "Route to\nhuman review\n(fallback)", HUM)
arrow(6.5, 4.2, 6.9, 3.8)
leaf = [("Image quality OK?", DET), ("Model healthy?\n(circuit breaker)", DET), ("Inspect regions\nwith VLM", AI), ("Apply decision\nrules", DET)]
for i, (t, c) in enumerate(leaf):
    x = 0.2 + i * 2.45
    box(x, 0.9, 2.25, 0.85, t, c, c == AI)
    arrow(4.2, 2.4, x + 1.12, 1.75)
ax.text(0.2, 0.35, "Blue: deterministic.  Yellow: the only AI step (Gemma 3 4B via Ollama, local).  Orange: human.", fontsize=9)
ax.text(0.2, 0.05, "Any failure on the AI path falls through to a human. The AI never touches the diverter.", fontsize=9)
plt.tight_layout()
fig.savefig(media / "2_tree.png", dpi=150)
plt.close(fig)

shutil.rmtree(tmp, ignore_errors=True)
print("wrote", sorted(p.name for p in media.iterdir()))
