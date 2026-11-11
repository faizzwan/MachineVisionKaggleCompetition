# Trusted Inspection Cell

**Subtitle:** A behaviour tree runs the line. A local vision-language model only looks. Fixed rules decide, and every failure goes to a human.

**Track:** Part B, Idea 6 (Camera-Based Quality & Part Inspection), built as Idea 2 (Hybrid Logic & VLM Orchestrator):
deterministic control with a narrow VLM handoff, plus Idea 3-style fault recovery (circuit breaker, human fallback).

**Links:** project repo: https://github.com/faizzwan/MachineVisionKaggleCompetition · public notebook: `<ADD LINK>` · video: `<ADD LINK>`

---

## The problem

End-of-line visual checks for missing fasteners and surface damage are still often done by eye. It is slow and tiring,
and misses go up late in a shift. A missing screw that ships becomes a field failure. Fixed-rule machine vision needs
a labelled defect set and re-tuning for every new part. A vision-language model can read a new part from a plain description,
but it cannot be trusted with the line. We saw Gemma 3 4B call a part with a missing screw "fine" at 95% confidence.

This project keeps the model's eyes and takes away its hands.

---

## 1. Architecture: deterministic logic and AI handoffs

### Who decides what

| Must never fail: deterministic | Open-ended perception: AI (Gemma 3 4B, local) |
|---|---|
| E-stop, checked before each part **and again just before actuation** | "Is there a screw in this close-up?" |
| Image quality gate (dark, overexposed, blurry: the AI is never called) | "Is this surface clean or damaged?" |
| Where to look: the part spec lists 4 screw positions and 5 surface areas | A one-sentence description of what it saw, shown to the operator |
| Which words are allowed and whether an answer is usable (validator) | |
| PASS / REJECT / REVIEW decision rules, timeouts, circuit breaker | |
| Diverter actuation and the audit log | |

### The control program (a `py_trees` behaviour tree)

```
Inspection cycle                      (Sequence)
├── E-stop clear?
├── Acquire part
├── Classify                          (Selector: first child that succeeds)
│   ├── AI path                       (Sequence)
│   │   ├── Image quality OK?         deterministic
│   │   ├── Model healthy?            circuit breaker
│   │   ├── Inspect regions with VLM  <- HANDOFF to AI, the only AI step
│   │   └── Apply decision rules      <- HANDBACK: deterministic from here on
│   └── Route to human review         fallback for every failure above
├── E-stop clear? (pre-actuation)
└── Actuate diverter                  pass conveyor / reject bin / review shelf
```

### The handoff and handback contract

* **Handoff (tree to AI):** the tree picks one region from the spec, crops it, and sends the model one fixed question.
  The model gets no say in where to look and no access to the line.
* **What may come back:** free text. It is treated as untrusted until the validator accepts it. It must hold a description
  and end with `ANSWER: <word>`, where the word is one of two allowed for that region (`SCREW`/`EMPTY` or `CLEAN`/`DAMAGED`).
  An empty reply, a missing ANSWER line, a word not allowed for that region, or an answer with no description is rejected.
* **Handback (AI to tree):** only a validated `RegionResult` crosses back. Fixed rules decide. One confirmed defect
  gives REJECT, because a false reject only costs a re-check. PASS needs every region checked and OK, because a false
  pass ships a defect. Anything else gives REVIEW.
* **The model never touches the actuator.** If the AI path fails for any reason, the Selector falls through to human review.

### Verifiability and override

* **Inspect:** the whole program is the tree printout above, and every threshold sits in one file (`cell/config.py`).
  Each part records the exact leaves it visited, e.g. `Acquire part → Image quality OK? → Route to human review`.
* **Verify:** the dashboard draws the checked regions on the image (green OK, red defect, grey not checked) next to the
  model's own words for each region. An append-only audit log keeps every raw model reply, decision and lane.
* **Override:** an operator can override any part from the dashboard or the CLI. The override is a separate record with
  their name. The AI-path decision is kept, never overwritten.
* **Stop:** a software e-stop in the dashboard halts the line before the next part and before the gate moves. It
  supplements the hardware e-stop and does not replace it, since it is not safety-rated.

---

## 2. Shop-floor deployment: on-device and offline

**Topology: single workstation, with an optional operator handheld on the local network.**

| Node | Runs | Network |
|---|---|---|
| Inspection workstation (laptop / IPC with camera) | Behaviour tree (`run.py`), Gemma 3 4B via Ollama, dashboard server | Model at `localhost:11434` only |
| Operator tablet or phone (optional) | Web browser pointed at the dashboard | Local LAN or Wi-Fi only (`streamlit run dashboard.py --server.address 0.0.0.0`) |

* **Zero external dependencies at run time.** The only network address in the code is `localhost`. Images, model replies
  and the audit log never leave the plant.
* **Why fully local and not hybrid:** a cloud call adds latency and a dependency on plant Wi-Fi to a decision taken on every
  part, and part images can be customer IP. The judgement the model makes is narrow (one close-up, one question), so a
  4B open model is enough and there is nothing worth offloading.
* **What happens when things drop:**

| Failure | What the cell does |
|---|---|
| Plant Wi-Fi or internet down | Nothing changes: nothing in the inspection path uses the network |
| Operator tablet disconnects | The line keeps running with all its safety behaviour; the dashboard is a separate process that shares only files |
| Model server slow or down | That part goes to human review. After 3 failures in a row: degraded mode (AI skipped, everything to review, model probed every 5th part, auto-recovers) |
| Image too dark, bright or blurry | Human review, AI never called |
| Invalid model answer | Rejected by the validator, human review |
| E-stop during the slow AI step | Decision is logged, diverter never moves |

---

## 3. Industrial impact and technical execution

* **Bottleneck addressed:** fastener presence and surface damage, among the most common end-of-line escapes. A new part is
  a new spec (regions plus allowed words), not a new training set.
* **Pipeline robustness:** image gate, region crops with screw crops upscaled for small features, a strict validator,
  early stop on the first defect, a circuit breaker, and pre-actuation e-stop.
* **Latency, measured honestly:** on our 8 GB laptop with no GPU, a close-up took **12 to 238 s, median 138 s**. That is
  slow because the machine is short of RAM and swaps the model to disk. A passing part took 11 minutes and a missing screw
  27 s, since the check stops at the first defect. This is too slow for a production line on this hardware. The fix is
  hardware (16 GB RAM or a GPU or edge accelerator), not logic: the tree, validator and rules do not change. Ollama keeps
  the model loaded between parts (`keep_alive`), and `run.py` warms the model up before the first part.
* **Usability for the worker:** a single screen with the latest part, a review queue with one-click PASS/REJECT override
  (name required), history, and a big e-stop button. Every reason is in plain English, in the form
  `missing screw at screw bottom-right: AI saw "<the model's own description>"`.
* **Code:** about 1,100 lines, modular (`spec`, `validator`, `vlm`, `quality`, `tree`, `audit`, `overlay`, `sources`).
  18 tests cover the validator, rules, every sample path, e-stop before and during inspection, the circuit breaker,
  an unreachable model server, and the override trail. They run in seconds with a mock model.

---

## 4. Trade-offs and why

| Choice | Why | What we gave up |
|---|---|---|
| **Gemma 3 4B** (open weights, via Ollama) | Takes images, runs quantised in about 3.3 GB on an 8 GB laptop CPU, no licence or cloud cost, explains its answer in words | Slower than a trained detector; needs the narrow-question design to be reliable |
| **VLM, not a trained CNN detector** | No labelled defect dataset; a new part is a spec edit; plain-English reasons for the operator | Lower throughput; accuracy must be proven per part type |
| **Close-ups, describe-then-answer** | See the table below: it was the only prompt design that worked | Up to 9 model calls per part |
| **Behaviour tree, not a state machine** | The Selector gives "AI path, else human" fallback structurally; the tree reads as a diagram an engineer can sign off; e-stop is re-checked on every tick; BTs are standard in robot stacks (ROS 2 / Nav2) | One more library (`py_trees`) |
| **Single device** | Fewest failure points; works with no network at all | No load sharing across cells (a later step is one Ollama edge server per line) |
| **Model confidence ignored** | It was meaningless: 95% on a wrong answer | Decisions rely on fixed rules only |

### What we learned about the model

Prompt-design experiments with Gemma 3 4B on a laptop CPU through Ollama, on synthetic parts:

| Approach | Result |
|---|---|
| Whole part, "is it defective?" with confidence | Missing screw called fine at 95% confidence; crack missed |
| Whole part, check each corner | All 4 screws "present" on every image |
| Close-up per corner, forced JSON answer | Same answer for every crop, right or wrong |
| **Close-up per region, describe first, then answer** | **Used in the final cell; measured below** |

### Measured run of the final cell (real Gemma 3 4B, fully offline)

Run on 2026-11-12. Setup: `gemma3:4b` in Ollama 0.40.16 on an Intel laptop (Windows 11, 7.7 GB RAM, CPU only),
600 s limit per close-up. Every model answer and timing is in `results/gemma_samples_run.md`.

| Part | Expected | Cell decision | Model's own words at the deciding region |
|---|---|---|---|
| `good_01` | PASS | **PASS**, all 9 regions OK | "smooth and uniform with no visible lines, cracks, scratches, or marks" |
| `missing_screw_01` | REJECT, top-right | **REJECT at screw top-right** after 2 close-ups | "a circular, black hole in the center" |
| `crack_01` | REJECT | **REJECT at surface top** | "several fine lines and scratches" |

* **3/3 decisions correct, 16/16 model answers passed the validator, 10/10 screw answers correct, 0 false defects on the
  good part.**
* Honest caveat on the crack: the model flagged "surface top", not "surface center" where most of the crack is. The crack's
  upper tip does lie inside the surface-top box, so the call is defensible, but the description is vague. We count it as a
  correct decision with weak evidence.
* Small sample, synthetic parts: this shows the pipeline working end to end on real model output. It is not an accuracy
  benchmark.

### The same hardware also exercised the fault path, live

Our first two real runs used the default 120 s limit, and every new close-up on this low-RAM laptop took longer than that.
The cell behaved as designed on real hardware: each timeout went to human review. After 3 in a row the circuit breaker
tripped and the rest of the parts went straight to review without calling the model. **9/9 parts went to a human, and none
was passed or rejected on a guess.** We then made the limit a command-line setting (`run.py --timeout`) and added a model
warm-up at start-up.

---

## Limits (stated plainly)

* The sample parts are **synthetic**, and the real-model run covers 3 parts (16 close-ups). We have not measured accuracy on
  real photographs.
* On an 8 GB CPU-only laptop a close-up takes minutes. Production use needs more RAM or an accelerator.
* Regions are fractions of the frame, so the cell assumes a fixed camera and a jig. They must be re-fitted for real parts.
* The diverter is simulated. `ActuateDiverter` is the single leaf to replace with a PLC write or robot motion routine.
* The software e-stop is not a safety-rated stop.

## Run it

```bash
python make_samples.py
python run.py --source samples --mock     # dry run, no model needed
python run.py --source samples            # real run with local Gemma (ollama pull gemma3:4b)
python run.py --source samples --timeout 600   # on a low-RAM CPU laptop
streamlit run dashboard.py                # operator dashboard (add --server.address 0.0.0.0 for a LAN tablet)
python -m pytest                          # safety behaviour tests
```
