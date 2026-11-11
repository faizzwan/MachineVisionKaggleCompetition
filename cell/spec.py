"""The part spec: what a good part looks like and where to look.

Positions are fractions of the image, so they assume the part sits in a fixed jig under a
fixed camera, as it would in a real inspection cell. The deterministic side decides WHERE
to look; the AI only answers a narrow question about each close-up.

Why close-ups: on a whole-part image, Gemma 3 4B said a plate with a missing screw was
fine at 95% confidence and missed a crack. On close-ups of each region it got every
screw right and flagged the crack.
"""

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

# Each region kind has exactly one "OK" answer and one "defect" answer.
OK_ANSWER = {"screw": "SCREW", "surface": "CLEAN"}
DEFECT_ANSWER = {"screw": "EMPTY", "surface": "DAMAGED"}
DEFECT_NAME = {"screw": "missing screw", "surface": "surface damage"}


@dataclass(frozen=True)
class Region:
    name: str
    kind: str  # "screw" or "surface"
    box: tuple[float, float, float, float]  # left, top, right, bottom as fractions of width/height


def _box(left: int, top: int, right: int, bottom: int, size: int = 512) -> tuple[float, float, float, float]:
    return (left / size, top / size, right / size, bottom / size)


def _screw(name: str, x: int, y: int) -> Region:
    return Region(name, "screw", _box(x - 60, y - 60, x + 60, y + 60))


# Screws first: a missing screw is the most common defect and ends the check early.
PART_SPEC = (
    _screw("screw top-left", 120, 120),
    _screw("screw top-right", 392, 120),
    _screw("screw bottom-left", 120, 392),
    _screw("screw bottom-right", 392, 392),
    Region("surface top", "surface", _box(160, 70, 352, 190)),
    Region("surface left", "surface", _box(70, 160, 190, 352)),
    Region("surface center", "surface", _box(160, 160, 352, 352)),
    Region("surface right", "surface", _box(322, 160, 442, 352)),
    Region("surface bottom", "surface", _box(160, 322, 352, 442)),
)

SCREW_CROP_PX = 360  # screw crops are upscaled; the model reads small features better


def crop(image_path: Path, region: Region) -> bytes:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    left, top, right, bottom = region.box
    out = img.crop((round(left * w), round(top * h), round(right * w), round(bottom * h)))
    if region.kind == "screw":
        out = out.resize((SCREW_CROP_PX, SCREW_CROP_PX))
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()
