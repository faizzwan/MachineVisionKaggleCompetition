# Project context

Entry for the Kaggle hackathon "Build with AI - Smart Automation, Robotics, and Vision"
(GDG Windsor): https://www.kaggle.com/competitions/build-with-ai-smart-automation-robotics-and-vision
Deadline: about 2026-11-15. Check the competition page for the exact time.

Judged, not leaderboard. Rubric (100): deterministic logic + AI handoffs 30, on-device/offline 25,
industrial impact + code 25, writeup/notebook/video 20. Submission = Kaggle Writeup with media
gallery, public notebook, public video, project link.

## What this is
An offline part-inspection cell: a py_trees behaviour tree runs the line, Gemma 3 4B (local,
via Ollama) answers narrow questions about close-up crops, a deterministic validator and
decision rules turn answers into PASS / REJECT / REVIEW. Streamlit dashboard for operators
(override, e-stop). See README.md for the tree and failure table.

## Decisions already made (don't undo without a reason)
- Whole-image prompts to Gemma 3 4B are unreliable: it passed a missing-screw part at 95%
  confidence and missed a crack. So the spec (cell/spec.py) picks fixed regions and the
  model is asked about one crop at a time.
- Forced JSON output made the model give the same answer for every crop. Prompts are
  "describe first, then a final `ANSWER: WORD` line"; the validator requires both.
- Model confidence is not used anywhere; it was meaningless.
- One defect -> REJECT (check stops early). PASS needs every region checked and OK.
- Dashboard is a separate process from run.py; they share only files in runtime/
  (audit.jsonl, overrides.jsonl, ESTOP).
- Region positions assume a fixed camera + jig. Coordinates in spec.py are for the
  synthetic 512px samples and must be re-fit to real photos.

## Environment
- Python venv in .venv (created with `python3 -m venv --without-pip .venv` and
  `pip3 --python .venv/bin/python install -r requirements.txt streamlit` because
  python3-venv was not installed).
- Ollama + `ollama pull gemma3:4b`. CPU only, 7.7 GB RAM: 12-238 s per crop (median 138 s), up to 9 crops per part.
- Tests: `.venv/bin/python -m pytest` (18 tests, use MockVLM, no model needed).
  On Windows use `.venv\Scripts\python.exe`; launch.json's `.venv/bin/streamlit` is Linux-only.

## Done (2026-11-11)
- Dashboard override bug fixed: views are an `st.segmented_control` with key "view" (tabs reset
  when the queue count in the label changed). Checked with AppTest, not yet by eye in a browser.
- MockVLM now flags the screw that is actually missing in missing_screw_02 (bottom-right).
- notebook/trusted_inspection_cell.ipynb: self-contained (embeds the code with %%writefile), executed
  offline with the mock; the live-Gemma cell skips itself when no Ollama is reachable.
  Regenerate it if cell/*.py changes, otherwise it drifts.
- WRITEUP.md, VIDEO_SCRIPT.md, media/ (gallery images rendered from the real code, mock model).

## Real Gemma run (2026-11-12, Windows laptop, 7.7 GB RAM, CPU only, Ollama 0.40.16)
- good_01 PASS, missing_screw_01 REJECT at top-right, crack_01 REJECT at "surface top" (crack tip is
  in that box; description vague). 3/3 decisions right, 16/16 answers valid. Log: results/gemma_samples_run.md.
- Close-ups took 12-238 s (median 138 s): the laptop swaps. With the default 120 s every call timed out
  and the circuit breaker sent all 9 parts to review. Use `run.py --timeout 600` on this machine.
- Added OllamaVLM.warm_up() (run.py calls it at start-up) and run.py --timeout.
- tools/ holds build_notebook.py, make_media.py, gemma_run.py (scripts that regenerate notebook/media/results).

## Next steps
1. Photograph real parts (screws in cardboard/wood, fixed camera, taped outline),
   put them in photos/, re-fit region boxes in cell/spec.py, run with real Gemma.
2. Record the video (VIDEO_SCRIPT.md), publish the notebook and repo, fill the three links in WRITEUP.md.
