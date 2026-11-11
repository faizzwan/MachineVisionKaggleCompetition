"""The behaviour tree that runs the inspection cell.

The tree is the deterministic part of the system: it owns the e-stop, the image gate,
the circuit breaker, the part spec, the decision rules and the actuator. The AI only
answers narrow questions about close-ups the tree chooses, and every answer must pass
the validator before it counts. Every way the AI path can fail falls through to
"Route to human review", so the line never acts on a guess.

    Inspection cycle                      (Sequence)
    ├── E-stop clear?
    ├── Acquire part
    ├── Classify                          (Selector: first child that succeeds)
    │   ├── AI path                       (Sequence)
    │   │   ├── Image quality OK?
    │   │   ├── Model healthy?            (circuit breaker)
    │   │   ├── Inspect regions with VLM  (one validated answer per spec region)
    │   │   └── Apply decision rules
    │   └── Route to human review
    ├── E-stop clear? (pre-actuation)
    └── Actuate diverter
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import py_trees
from py_trees.common import Status

from . import quality
from .config import CellConfig
from .quality import ImageQuality
from .spec import PART_SPEC, Region
from .validator import Decision, RegionResult, ValidationError, decide, parse_region_answer
from .vlm import VLMError

LANES = {
    Decision.PASS: "pass conveyor",
    Decision.REJECT: "reject bin",
    Decision.REVIEW: "human review shelf",
}


@dataclass
class PartRecord:
    part_id: str
    image_path: Path
    quality: ImageQuality | None = None
    vlm_raw: dict[str, str] = field(default_factory=dict)  # region name -> raw model text
    vlm_latency_s: float | None = None
    regions: list[RegionResult] = field(default_factory=list)  # validated answers
    decision: Decision | None = None
    reason: str = ""
    decided_by: str = ""  # "ai+rules" or "fallback"
    lane: str | None = None
    trace: list[str] = field(default_factory=list)  # leaves visited, in order


class Cell:
    """Shared state the tree's leaves read and write: one part at a time, plus line state."""

    def __init__(self, cfg: CellConfig, source, vlm, spec: tuple[Region, ...] = PART_SPEC):
        self.cfg = cfg
        self.source = source
        self.vlm = vlm
        self.spec = spec
        self.runtime = Path(cfg.runtime_dir)
        self.run_id = time.strftime("%Y%m%d-%H%M%S")
        self.part: PartRecord | None = None
        self.parts_seen = 0
        self.stopped: str | None = None  # why the line stopped, if it did
        self.consecutive_vlm_failures = 0
        self.parts_while_degraded = 0

    @property
    def estop_file(self) -> Path:
        return self.runtime / "ESTOP"

    @property
    def degraded(self) -> bool:
        return self.consecutive_vlm_failures >= self.cfg.max_consecutive_vlm_failures


class Leaf(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, cell: Cell):
        super().__init__(name)
        self.cell = cell

    def update(self) -> Status:
        if self.cell.part is not None:
            self.cell.part.trace.append(self.name)
        return self.step()

    def step(self) -> Status:
        raise NotImplementedError


class EStopClear(Leaf):
    def step(self):
        if self.cell.estop_file.exists():
            self.cell.stopped = "operator e-stop engaged"
            return Status.FAILURE
        return Status.SUCCESS


class AcquirePart(Leaf):
    def step(self):
        path = self.cell.source.next_image()
        if path is None:
            self.cell.stopped = "no more parts"
            return Status.FAILURE
        self.cell.parts_seen += 1
        self.cell.part = PartRecord(f"{self.cell.run_id}-{self.cell.parts_seen:04d}", Path(path))
        self.cell.part.trace.append(self.name)
        return Status.SUCCESS


class ImageQualityOK(Leaf):
    def step(self):
        part = self.cell.part
        part.quality = quality.measure(part.image_path)
        issue = quality.problem(part.quality, self.cell.cfg)
        if issue:
            part.reason = issue
            return Status.FAILURE
        return Status.SUCCESS


class ModelHealthy(Leaf):
    """Circuit breaker. After repeated AI failures, stop waiting on the model for every
    part (each failure can cost a full timeout) and only probe it now and then."""

    def step(self):
        cell = self.cell
        if not cell.degraded:
            return Status.SUCCESS
        cell.parts_while_degraded += 1
        if cell.parts_while_degraded % cell.cfg.probe_every_n_parts == 0:
            return Status.SUCCESS  # probe: give the model one chance to prove it is back
        cell.part.reason = (
            f"degraded mode: AI failed {cell.consecutive_vlm_failures} times in a row, skipping it"
        )
        return Status.FAILURE


class InspectRegions(Leaf):
    """Ask the VLM about each region in the part spec. Each answer is validated before
    the next question; one invalid answer sends the whole part to a human."""

    def step(self):
        cell, part = self.cell, self.cell.part
        t0 = time.monotonic()
        try:
            for region in cell.spec:
                raw = cell.vlm.inspect(part.image_path, region)
                part.vlm_raw[region.name] = raw
                result = parse_region_answer(raw, region)
                part.regions.append(result)
                if result.is_defect:
                    break  # one confirmed defect is enough to reject; don't spend time on the rest
        except VLMError as e:
            part.reason = f"AI unavailable: {e}"
            cell.consecutive_vlm_failures += 1
            return Status.FAILURE
        except ValidationError as e:
            part.reason = f"AI answer rejected: {e}"
            cell.consecutive_vlm_failures += 1
            return Status.FAILURE
        finally:
            part.vlm_latency_s = round(time.monotonic() - t0, 2)
        cell.consecutive_vlm_failures = 0
        cell.parts_while_degraded = 0
        return Status.SUCCESS


class ApplyDecisionRules(Leaf):
    def step(self):
        part = self.cell.part
        part.decision, part.reason = decide(part.regions, self.cell.spec)
        part.decided_by = "ai+rules"
        return Status.SUCCESS


class RouteToHumanReview(Leaf):
    def step(self):
        part = self.cell.part
        part.decision = Decision.REVIEW
        part.decided_by = "fallback"
        part.reason = part.reason or "AI path did not produce a decision"
        return Status.SUCCESS


class ActuateDiverter(Leaf):
    """Simulated diverter gate. Swap this leaf for a PLC write or robot motion routine."""

    def step(self):
        part = self.cell.part
        part.lane = LANES[part.decision]
        return Status.SUCCESS


def build_tree(cell: Cell) -> py_trees.trees.BehaviourTree:
    ai_path = py_trees.composites.Sequence(
        "AI path",
        memory=True,
        children=[
            ImageQualityOK("Image quality OK?", cell),
            ModelHealthy("Model healthy?", cell),
            InspectRegions("Inspect regions with VLM", cell),
            ApplyDecisionRules("Apply decision rules", cell),
        ],
    )
    classify = py_trees.composites.Selector(
        "Classify",
        memory=True,
        children=[ai_path, RouteToHumanReview("Route to human review", cell)],
    )
    root = py_trees.composites.Sequence(
        "Inspection cycle",
        memory=True,
        children=[
            EStopClear("E-stop clear?", cell),
            AcquirePart("Acquire part", cell),
            classify,
            EStopClear("E-stop clear? (pre-actuation)", cell),
            ActuateDiverter("Actuate diverter", cell),
        ],
    )
    return py_trees.trees.BehaviourTree(root)
