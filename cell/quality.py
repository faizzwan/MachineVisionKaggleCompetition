"""Image quality gate. Cheap, deterministic, and runs before the AI: a dark or blurry
photo gets a human, not a guess."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import CellConfig

ANALYSIS_SIZE = 512  # measure at a fixed size so thresholds don't depend on camera resolution


@dataclass(frozen=True)
class ImageQuality:
    brightness: float  # mean grey level, 0..255
    sharpness: float  # variance of the Laplacian; low means blurry


def measure(image_path: Path) -> ImageQuality:
    img = Image.open(image_path).convert("L")
    img.thumbnail((ANALYSIS_SIZE, ANALYSIS_SIZE))
    g = np.asarray(img, dtype=np.float64)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return ImageQuality(brightness=float(g.mean()), sharpness=float(lap.var()))


def problem(q: ImageQuality, cfg: CellConfig) -> str | None:
    """Return why the image is unusable, or None if it is fine."""
    if q.brightness < cfg.min_brightness:
        return f"image too dark (brightness {q.brightness:.0f} < {cfg.min_brightness:.0f})"
    if q.brightness > cfg.max_brightness:
        return f"image overexposed (brightness {q.brightness:.0f} > {cfg.max_brightness:.0f})"
    if q.sharpness < cfg.min_sharpness:
        return f"image too blurry (sharpness {q.sharpness:.0f} < {cfg.min_sharpness:.0f})"
    return None
