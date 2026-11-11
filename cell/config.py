"""Every threshold the cell uses, in one place, so an engineer can audit and tune them."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CellConfig:
    # Local model served by Ollama. Nothing leaves this machine.
    model: str = "gemma3:4b"
    ollama_url: str = "http://localhost:11434"
    # Per region. Measured on an 8 GB CPU-only laptop: 12-238 s per close-up (it swaps), so
    # use `run.py --timeout 600` there. With enough RAM or a GPU, 120 s is ample.
    vlm_timeout_s: float = 120.0

    # Image quality gate, checked before the AI ever sees the image.
    min_brightness: float = 40.0
    max_brightness: float = 225.0
    min_sharpness: float = 30.0

    # Circuit breaker: after this many AI failures in a row the cell runs in
    # degraded mode (every part goes to human review) and only probes the
    # model again every `probe_every_n_parts` parts.
    max_consecutive_vlm_failures: int = 3
    probe_every_n_parts: int = 5

    runtime_dir: str = "runtime"
