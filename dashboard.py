"""Operator dashboard.

    .venv/bin/streamlit run dashboard.py

It runs as a separate process from the line (run.py): it only reads the audit log and
writes operator actions (overrides, e-stop). If the dashboard crashes, the line keeps its
safety behaviour; if the line crashes, the dashboard still shows what happened.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from cell import audit
from cell.config import CellConfig
from cell.overlay import draw_regions
from cell.spec import DEFECT_ANSWER

RUNTIME = Path(CellConfig.runtime_dir)
ESTOP = RUNTIME / "ESTOP"
BADGE_COLOUR = {"PASS": "green", "REJECT": "red", "REVIEW": "orange"}

st.set_page_config(page_title="Inspection Cell", layout="wide")


def operator() -> str:
    return st.session_state.get("operator", "").strip()


def outcome(part: dict) -> dict[str, bool]:
    return {r["region"]: r["answer"] == DEFECT_ANSWER[r["kind"]] for r in part["regions"]}


def badge(decision: str) -> None:
    st.badge(decision, color=BADGE_COLOUR.get(decision, "gray"))


def override_buttons(part: dict, key: str) -> None:
    name = operator()
    left, right = st.columns(2)
    for col, decision in ((left, "PASS"), (right, "REJECT")):
        if col.button(f"Override: {decision}", key=f"{key}-{decision}", disabled=not name, width="stretch"):
            audit.record_override(RUNTIME, part["part_id"], decision, name)
            st.rerun()
    if not name:
        st.caption("Enter your name in the sidebar to override.")


def part_detail(part: dict, key: str) -> None:
    image_col, info_col = st.columns([2, 3])
    with image_col:
        path = Path(part["image_path"])
        if path.exists():
            st.image(draw_regions(path, outcome(part)), width="stretch")
        else:
            st.warning(f"image not found: {path}")
        st.caption("Green: AI saw it OK · Red: AI found a defect · Grey: not checked")
    with info_col:
        st.markdown(f"**{path.name}** · `{part['part_id']}` · {part['timestamp']}")
        badge(part["final_decision"])
        if part["override"]:
            o = part["override"]
            st.markdown(f"Overridden by **{o['operator']}** at {o['timestamp']} (AI path said {part['decision']})")
        st.markdown(f"**Why:** {part['reason']}")
        meta = f"Decided by: {part['decided_by']} · Model: {part['model']}"
        if part["vlm_latency_s"] is not None:
            meta += f" · AI time: {part['vlm_latency_s']:.0f}s"
        st.caption(meta)
        if part["regions"]:
            st.dataframe(
                pd.DataFrame(part["regions"])[["region", "answer", "description"]],
                hide_index=True,
                width="stretch",
            )
        st.caption("Behaviour tree path: " + " → ".join(part["trace"]))
        override_buttons(part, key)


with st.sidebar:
    st.header("Operator")
    st.text_input("Your name", key="operator", placeholder="required for overrides")
    st.divider()
    if ESTOP.exists():
        st.error("E-STOP ENGAGED: line halted")
        if st.button("Clear e-stop", width="stretch"):
            ESTOP.unlink(missing_ok=True)
            st.rerun()
        st.caption("Clearing does not restart the line. Restart run.py deliberately.")
    else:
        if st.button("E-STOP", type="primary", width="stretch"):
            RUNTIME.mkdir(parents=True, exist_ok=True)
            ESTOP.touch()
            st.rerun()
        st.caption("Halts before the next part and before any actuation.")


@st.fragment(run_every="2s")
def live() -> None:
    parts = audit.load_parts(RUNTIME)
    st.title("Trusted Inspection Cell")
    if ESTOP.exists():
        st.error("E-STOP ENGAGED")

    finals = [p["final_decision"] for p in parts]
    cols = st.columns(5)
    cols[0].metric("Parts", len(parts))
    cols[1].metric("Pass", finals.count("PASS"))
    cols[2].metric("Reject", finals.count("REJECT"))
    cols[3].metric("Waiting for review", finals.count("REVIEW"))
    cols[4].metric("Operator overrides", sum(1 for p in parts if p["override"]))

    if not parts:
        st.info("No parts yet. Start the line: `.venv/bin/python run.py --source samples`")
        return

    # A segmented control (not st.tabs): the selected view is kept in session state under a
    # fixed key, so an override or the 2 s refresh never throws the operator back to the
    # first tab. (st.tabs reset whenever a label changed, and the queue count is in a label.)
    waiting_count = finals.count("REVIEW")
    view = st.segmented_control(
        "View",
        ["Latest part", "Review queue", "History"],
        default="Latest part",
        required=True,
        key="view",
        label_visibility="collapsed",
    )
    if view == "Review queue":
        st.caption(f"{waiting_count} part(s) waiting for a human decision")

    if view == "Latest part":
        part_detail(parts[-1], "latest")
    elif view == "Review queue":
        waiting = [p for p in parts if p["final_decision"] == "REVIEW"]
        if not waiting:
            st.success("Nothing waiting for review.")
        for p in reversed(waiting):
            with st.container(border=True):
                part_detail(p, f"queue-{p['part_id']}")
    else:
        table = pd.DataFrame(
            {
                "part": [p["part_id"] for p in parts],
                "time": [p["timestamp"] for p in parts],
                "image": [Path(p["image_path"]).name for p in parts],
                "AI path": [p["decision"] for p in parts],
                "final": [p["final_decision"] for p in parts],
                "override by": [p["override"]["operator"] if p["override"] else "" for p in parts],
                "reason": [p["reason"] for p in parts],
            }
        ).iloc[::-1]
        st.dataframe(table, hide_index=True, width="stretch")
        chosen = st.selectbox("Inspect a part", table["part"], key="history-pick")
        part_detail(next(p for p in parts if p["part_id"] == chosen), "history")


live()
