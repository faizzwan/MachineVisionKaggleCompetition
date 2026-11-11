"""Vision-language model clients. They only return raw text; trusting it is the validator's job."""

import base64
from pathlib import Path

import requests

from .config import CellConfig
from .spec import DEFECT_ANSWER, OK_ANSWER, Region, crop

# "Describe first, then answer" matters. With a forced JSON schema, Gemma 3 4B gave the
# same answer for every crop; asked to describe first, it read each crop correctly.
PROMPTS = {
    "screw": f"""This is a close-up of one screw position on a metal plate.
First, describe in one sentence exactly what you see.
Then, on a new last line, write exactly one of:
ANSWER: {OK_ANSWER['screw']}   (a screw head is installed)
ANSWER: {DEFECT_ANSWER['screw']}   (the hole has no screw)""",
    "surface": f"""This is a close-up of a metal plate surface on a manufacturing line.
First, describe the surface in one sentence: is it smooth and uniform, or are there any lines, cracks, scratches or marks on it?
Then, on a new last line, write exactly one of:
ANSWER: {OK_ANSWER['surface']}
ANSWER: {DEFECT_ANSWER['surface']}""",
}


class VLMError(Exception):
    """The model could not produce an answer at all (timeout, server down, HTTP error)."""


class OllamaVLM:
    def __init__(self, cfg: CellConfig):
        self.cfg = cfg

    @property
    def name(self) -> str:
        return f"ollama:{self.cfg.model}"

    def warm_up(self, timeout_s: float = 900.0) -> float:
        """Load the model and run its vision encoder once before the line starts.

        On a CPU laptop the first image call took ~158 s (the vision path warming up) and
        later calls ~10-20 s. Without this, the first parts hit the per-region timeout and
        trip the circuit breaker. Returns the seconds it took; raises VLMError on failure."""
        import io
        import time

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (150, 155, 165)).save(buf, "PNG")
        body = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": "Describe this image in five words.",
                          "images": [base64.b64encode(buf.getvalue()).decode()]}],
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0},
        }
        t0 = time.monotonic()
        try:
            r = requests.post(f"{self.cfg.ollama_url}/api/chat", json=body, timeout=timeout_s)
            r.raise_for_status()
        except requests.RequestException as e:
            raise VLMError(f"model warm-up failed: {e}")
        return time.monotonic() - t0

    def inspect(self, image_path: Path, region: Region) -> str:
        image_b64 = base64.b64encode(crop(image_path, region)).decode()
        body = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": PROMPTS[region.kind], "images": [image_b64]}],
            "stream": False,
            "keep_alive": "30m",  # avoid reloading the model between parts
            "options": {"temperature": 0},
        }
        try:
            r = requests.post(
                f"{self.cfg.ollama_url}/api/chat", json=body, timeout=self.cfg.vlm_timeout_s
            )
            r.raise_for_status()
            return r.json()["message"]["content"]
        except requests.Timeout:
            raise VLMError(f"model did not answer within {self.cfg.vlm_timeout_s:.0f}s")
        except requests.ConnectionError:
            raise VLMError("model server unreachable")
        except (requests.HTTPError, KeyError, ValueError) as e:
            raise VLMError(f"bad response from model server: {e}")


class MockVLM:
    """Stand-in model for tests and dry runs. It reads the answer from the file name,
    so it is NOT an inspector; it exists to exercise every path through the tree.

      missing_screw_01  -> top-right screw hole is empty  (matches make_samples.py)
      missing_screw_02  -> bottom-right screw hole is empty
      missing_screw_*   -> any other number: top-right
      crack_*           -> centre of the surface is damaged
      garbage_*         -> rambling answer with no ANSWER line
      timeout_*         -> raises VLMError
      anything else     -> every region OK
    """

    name = "mock"

    def inspect(self, image_path: Path, region: Region) -> str:
        stem = Path(image_path).stem
        if stem.startswith("timeout"):
            raise VLMError("model did not answer (simulated timeout)")
        if stem.startswith("garbage"):
            return "Sure! The part looks mostly fine I think."
        empty_hole = "screw bottom-right" if stem.startswith("missing_screw_02") else "screw top-right"
        defect = (stem.startswith("missing_screw") and region.name == empty_hole) or (
            stem.startswith("crack") and region.name == "surface center"
        )
        if defect:
            return f"I see a defect here.\nANSWER: {DEFECT_ANSWER[region.kind]}"
        return f"Looks normal.\nANSWER: {OK_ANSWER[region.kind]}"
