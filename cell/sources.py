"""Where part images come from: a folder (tests, datasets, demos) or a live webcam."""

import time
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


class FolderSource:
    def __init__(self, folder: Path):
        self.images = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        self._i = 0

    def next_image(self) -> Path | None:
        if self._i >= len(self.images):
            return None
        self._i += 1
        return self.images[self._i - 1]


class WebcamSource:
    """Grabs a frame every `interval_s` seconds and saves it, so the audit log can point at it."""

    def __init__(self, capture_dir: Path, device: int = 0, interval_s: float = 3.0, max_parts: int | None = None):
        import cv2  # optional dependency, only needed for live capture

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open webcam {device}")
        self.dir = Path(capture_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.interval_s = interval_s
        self.max_parts = max_parts
        self._n = 0
        self._last = 0.0

    def next_image(self) -> Path | None:
        if self.max_parts is not None and self._n >= self.max_parts:
            return None
        time.sleep(max(0.0, self.interval_s - (time.monotonic() - self._last)))
        ok, frame = self.cap.read()
        self._last = time.monotonic()
        if not ok:
            return None
        self._n += 1
        path = self.dir / f"capture_{time.strftime('%Y%m%d-%H%M%S')}_{self._n:04d}.jpg"
        self.cv2.imwrite(str(path), frame)
        return path
