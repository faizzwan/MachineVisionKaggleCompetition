"""Generate synthetic part images (a plate with four screws) for dry runs and tests.
The file names drive MockVLM; with a real model they are just pictures."""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SCREWS = [(120, 120), (392, 120), (120, 392), (392, 392)]


def plate(missing: int | None = None, crack: bool = False, background: int = 200) -> Image.Image:
    img = Image.new("RGB", (512, 512), (background,) * 3)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((60, 60, 452, 452), radius=24, fill=(150, 155, 165), outline=(70, 70, 80), width=4)
    for i, (x, y) in enumerate(SCREWS):
        if i == missing:
            d.ellipse((x - 14, y - 14, x + 14, y + 14), fill=(30, 30, 30))  # empty hole
            continue
        d.ellipse((x - 26, y - 26, x + 26, y + 26), fill=(205, 205, 210), outline=(60, 60, 60), width=3)
        d.line((x - 16, y, x + 16, y), fill=(60, 60, 60), width=5)
        d.line((x, y - 16, x, y + 16), fill=(60, 60, 60), width=5)
    if crack:
        d.line([(200, 180), (230, 230), (215, 270), (260, 330), (250, 380)], fill=(40, 40, 40), width=4)
    return img


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    images = {
        "good_01.png": plate(),
        "good_02.png": plate(background=180),
        "missing_screw_01.png": plate(missing=1),
        "missing_screw_02.png": plate(missing=3),
        "crack_01.png": plate(crack=True),
        "garbage_01.png": plate(),
        "timeout_01.png": plate(),
        "dark_01.png": Image.eval(plate(), lambda v: v // 8),
        "blurry_01.png": plate().filter(ImageFilter.GaussianBlur(10)),
    }
    for name, img in images.items():
        img.save(out / name)
    print(f"wrote {len(images)} images to {out}/")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("samples"))
