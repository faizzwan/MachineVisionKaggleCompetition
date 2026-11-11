import json
from dataclasses import replace

import pytest

import make_samples
from cell import audit
from cell.config import CellConfig
from cell.sources import FolderSource
from cell.spec import PART_SPEC
from cell.tree import Cell, build_tree
from cell.validator import Decision, RegionResult, ValidationError, decide, parse_region_answer
from cell.vlm import MockVLM, OllamaVLM

SCREW = next(r for r in PART_SPEC if r.kind == "screw")
SURFACE = next(r for r in PART_SPEC if r.kind == "surface")


@pytest.fixture
def samples(tmp_path):
    make_samples.main(tmp_path / "samples")
    return tmp_path / "samples"


def make_cell(tmp_path, folder, vlm=None, **cfg_overrides):
    cfg = replace(CellConfig(), runtime_dir=str(tmp_path / "runtime"), **cfg_overrides)
    cell = Cell(cfg, FolderSource(folder), vlm or MockVLM())
    return cell, build_tree(cell)


def run_all(cell, tree):
    parts = []
    while cell.stopped is None:
        cell.part = None
        tree.tick()
        if cell.part is not None:
            parts.append(cell.part)
    return parts


# --- validator: the AI's answer is never trusted as-is ---

@pytest.mark.parametrize(
    "raw, region, msg",
    [
        ("", SCREW, "empty answer"),
        ("The part is fine.", SCREW, "no ANSWER line"),
        ("A screw.\nANSWER: MAYBE", SCREW, "expected one of"),
        ("A clean surface.\nANSWER: SCREW", SURFACE, "expected one of"),  # wrong vocabulary for region
        ("ANSWER: SCREW", SCREW, "no description"),
    ],
)
def test_validator_rejects_bad_answers(raw, region, msg):
    with pytest.raises(ValidationError, match=msg):
        parse_region_answer(raw, region)


@pytest.mark.parametrize("last_line", ["ANSWER: EMPTY", "**ANSWER:** empty", "ANSWER: EMPTY   (the hole has no screw)"])
def test_validator_accepts_answer_line_variants(last_line):
    r = parse_region_answer(f"A solid black circle.\n{last_line}", SCREW)
    assert r.answer == "EMPTY" and r.is_defect and r.description == "A solid black circle."


def test_pass_needs_every_region_reject_needs_one():
    ok = [RegionResult(r.name, r.kind, "SCREW" if r.kind == "screw" else "CLEAN", "fine") for r in PART_SPEC]
    assert decide(ok, PART_SPEC)[0] is Decision.PASS
    assert decide(ok[:-1], PART_SPEC)[0] is Decision.REVIEW
    assert decide([RegionResult(SCREW.name, "screw", "EMPTY", "a hole")], PART_SPEC)[0] is Decision.REJECT


# --- the tree ---

def test_every_sample_gets_the_expected_decision(tmp_path, samples):
    parts = run_all(*make_cell(tmp_path, samples))
    got = {p.image_path.stem: p.decision for p in parts}
    assert got == {
        "blurry_01": Decision.REVIEW,
        "crack_01": Decision.REJECT,
        "dark_01": Decision.REVIEW,
        "garbage_01": Decision.REVIEW,
        "good_01": Decision.PASS,
        "good_02": Decision.PASS,
        "missing_screw_01": Decision.REJECT,
        "missing_screw_02": Decision.REJECT,
        "timeout_01": Decision.REVIEW,
    }
    assert all(p.lane for p in parts)


def test_defect_stops_the_check_early(tmp_path, samples):
    parts = run_all(*make_cell(tmp_path, samples))
    missing = next(p for p in parts if p.image_path.stem == "missing_screw_01")
    assert [r.region for r in missing.regions] == ["screw top-left", "screw top-right"]
    good = next(p for p in parts if p.image_path.stem == "good_01")
    assert len(good.regions) == len(PART_SPEC)


def test_mock_flags_the_screw_that_is_actually_missing(tmp_path, samples):
    parts = run_all(*make_cell(tmp_path, samples))
    second = next(p for p in parts if p.image_path.stem == "missing_screw_02")
    assert second.regions[-1].region == "screw bottom-right"  # make_samples removes screw 3


def test_bad_image_never_reaches_the_ai(tmp_path, samples):
    parts = run_all(*make_cell(tmp_path, samples))
    dark = next(p for p in parts if p.image_path.stem == "dark_01")
    assert "Inspect regions with VLM" not in dark.trace
    assert dark.vlm_raw == {}


def test_estop_halts_before_any_part(tmp_path, samples):
    cell, tree = make_cell(tmp_path, samples)
    cell.estop_file.parent.mkdir(parents=True)
    cell.estop_file.touch()
    assert run_all(cell, tree) == []
    assert cell.stopped == "operator e-stop engaged"


def test_estop_during_inspection_blocks_actuation(tmp_path, samples):
    class EStopWhileThinking(MockVLM):
        def inspect(self, image_path, region):
            cell.estop_file.parent.mkdir(parents=True, exist_ok=True)
            cell.estop_file.touch()  # operator hits e-stop while the model is busy
            return super().inspect(image_path, region)

    cell, tree = make_cell(tmp_path, samples, vlm=EStopWhileThinking())
    cell.source.images = [p for p in cell.source.images if p.stem == "crack_01"]
    [part] = run_all(cell, tree)
    assert part.decision is Decision.REJECT
    assert part.lane is None  # decided, but the diverter never moved


def test_circuit_breaker_stops_calling_a_dead_model(tmp_path, samples):
    calls = []

    class DeadModel(MockVLM):
        def inspect(self, image_path, region):
            calls.append(image_path)
            return MockVLM.inspect(self, samples / "timeout_01.png", region)

    cell, tree = make_cell(
        tmp_path, samples, vlm=DeadModel(), max_consecutive_vlm_failures=2, probe_every_n_parts=3
    )
    cell.source.images = [samples / "good_01.png"] * 9
    parts = run_all(cell, tree)
    assert all(p.decision is Decision.REVIEW for p in parts)
    # 2 failures trip the breaker; of the remaining 7 parts, every 3rd is a probe.
    assert len(calls) == 2 + 2
    assert "degraded mode" in parts[2].reason


def test_model_server_down_routes_to_review(tmp_path, samples):
    vlm = OllamaVLM(replace(CellConfig(), ollama_url="http://127.0.0.1:9", vlm_timeout_s=2))
    cell, tree = make_cell(tmp_path, samples, vlm=vlm)
    cell.source.images = [samples / "good_01.png"]
    [part] = run_all(cell, tree)
    assert part.decision is Decision.REVIEW
    # Linux refuses the connection at once ("unreachable"); Windows silently drops it until
    # the timeout. Either way the part must go to a human, not to the line.
    assert "unreachable" in part.reason or "did not answer" in part.reason


# --- audit + operator override ---

def test_override_changes_final_decision_and_is_logged(tmp_path, samples):
    cell, tree = make_cell(tmp_path, samples)
    for p in run_all(cell, tree):
        audit.record_part(cell.runtime, p, "mock")
    garbage = next(p for p in audit.load_parts(cell.runtime) if "garbage" in p["image_path"])
    assert garbage["final_decision"] == "REVIEW"

    audit.record_override(cell.runtime, garbage["part_id"], "PASS", "Sam", "checked by hand")
    after = next(p for p in audit.load_parts(cell.runtime) if p["part_id"] == garbage["part_id"])
    assert after["final_decision"] == "PASS"
    assert after["decision"] == "REVIEW"  # original AI-path decision is kept
    assert after["override"]["operator"] == "Sam"
    lines = audit.audit_path(cell.runtime).read_text().splitlines()
    assert all(json.loads(line)["model"] == "mock" for line in lines)
