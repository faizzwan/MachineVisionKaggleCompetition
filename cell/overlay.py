"""Draw the part spec on an image so an operator can see exactly where the AI looked."""

from pathlib import Path

from PIL import Image, ImageDraw

from .spec import PART_SPEC, Region

OK_COLOUR = (22, 163, 74)
DEFECT_COLOUR = (220, 38, 38)
UNCHECKED_COLOUR = (150, 150, 150)


def draw_regions(image_path: Path, outcome: dict[str, bool], spec: tuple[Region, ...] = PART_SPEC) -> Image.Image:
    """`outcome` maps region name -> True if the AI found a defect there, False if OK.
    Regions missing from it were not checked and are drawn grey."""
    img = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(img)
    w, h = img.size
    width = max(2, round(min(w, h) / 160))
    for region in spec:
        left, top, right, bottom = region.box
        if region.name not in outcome:
            colour = UNCHECKED_COLOUR
        else:
            colour = DEFECT_COLOUR if outcome[region.name] else OK_COLOUR
        d.rectangle((left * w, top * h, right * w, bottom * h), outline=colour, width=width)
    return img
