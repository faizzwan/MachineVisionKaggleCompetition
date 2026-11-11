# Trusted Inspection Cell

An offline part-inspection cell where a **behaviour tree** runs the line and a local
**vision-language model (Gemma 3 4B via Ollama)** answers narrow questions about close-ups
the tree chooses. The AI never moves anything: each answer must pass a deterministic
validator, fixed rules turn the answers into a decision, and every failure mode falls back
to a human.

## How control flows

```
Inspection cycle                      (Sequence)
├── E-stop clear?                     deterministic: halts the line
├── Acquire part                      camera or folder
├── Classify                          (Selector)
│   ├── AI path                       (Sequence)
│   │   ├── Image quality OK?         deterministic: dark/blurry -> human, AI never sees it
│   │   ├── Model healthy?            circuit breaker: stop waiting on a failing model
│   │   ├── Inspect regions with VLM  Gemma, local, one close-up per spec region  <- the only AI step
│   │   └── Apply decision rules      deterministic
│   └── Route to human review         fallback for every failure above
├── E-stop clear? (pre-actuation)     re-checked after the slow AI step
└── Actuate diverter                  pass conveyor / reject bin / review shelf
```

**Who decides what**

| Deterministic (rules) | AI (Gemma) |
|---|---|
| Where to look: the part spec lists 4 screw positions and 5 surface areas ([cell/spec.py](cell/spec.py)) | Is there a screw in this close-up? Is this surface clean? |
| Whether the answer is usable: must describe what it saw, then give one allowed word | Describing what it sees, in words the operator reads |
| The decision: one defect -> REJECT; every region checked and OK -> PASS | |
| E-stop, image quality, timeouts, circuit breaker, actuation | |

## What we learned about the model (and why the design looks like this)

| Approach | Result with Gemma 3 4B on CPU |
|---|---|
| Whole part, "is it defective?" with confidence | Said a part with a **missing screw** was fine at **95% confidence**; missed the crack |
| Whole part, check each corner | Said all 4 screws present on every image |
| Close-up per corner, forced JSON answer | Same answer for every crop, right or wrong |
| **Close-up per region, describe first, then answer** | **Used in the final cell; measured below** |

So: self-reported confidence is not used anywhere, the deterministic side decides where to
look, and the model must describe before it answers.

Measured on 2026-11-12 (Windows laptop, 7.7 GB RAM, CPU only): good part PASS, missing screw REJECT,
crack REJECT, with 16/16 answers valid. Close-ups took 12-238 s on that machine because it is short of RAM, so use
`run.py --timeout 600` there. Up to 9 close-ups per part; a defect stops the check early. More RAM or a GPU
brings this to seconds. Full log: [results/gemma_samples_run.md](results/gemma_samples_run.md).

## What can go wrong

| Failure | What the cell does |
|---|---|
| Image too dark, bright or blurry | Human review, AI not called |
| Model timeout or server down | Human review; after 3 in a row, degraded mode (AI skipped, probed every 5 parts) |
| Answer with no description, no ANSWER line, or a word not allowed for that region | Answer rejected, human review |
| Operator hits e-stop mid-inspection | Decision is logged, diverter never moves |
| Operator disagrees | `review.py decide` override, logged with their name; the AI's decision is kept |

All thresholds live in [cell/config.py](cell/config.py).

## Run it

```bash
.venv/bin/python make_samples.py                 # synthetic test parts
.venv/bin/python run.py --source samples --mock  # dry run, no model needed
.venv/bin/python run.py --source samples         # real run with local Gemma
.venv/bin/streamlit run dashboard.py             # operator dashboard: live view, review queue, overrides, e-stop
.venv/bin/python review.py                       # same review queue, from the terminal
.venv/bin/python -m pytest                       # safety behaviour tests
```

On Windows replace `.venv/bin/python` with `.venv\Scripts\python.exe`.

Also in the repo: [WRITEUP.md](WRITEUP.md) (Kaggle writeup draft), [VIDEO_SCRIPT.md](VIDEO_SCRIPT.md),
[notebook/trusted_inspection_cell.ipynb](notebook/trusted_inspection_cell.ipynb) (self-contained, runs offline with the mock model).

E-stop: `touch runtime/ESTOP` (delete it to clear). Audit trail: `runtime/audit.jsonl`.

Model setup (one time): install [Ollama](https://ollama.com), then `ollama pull gemma3:4b`.

Camera setup: region positions assume the part sits in the same place in every frame
(a jig or printed outline under a fixed camera).
