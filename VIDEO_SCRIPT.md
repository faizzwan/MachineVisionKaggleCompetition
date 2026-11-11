# Demo video script (target about 3:00; check the competition page for the limit)

The rubric asks the video to show the system **working end to end, including how it responds to edge cases, faults and
unexpected inputs**. Every fault row below is a real behaviour of the code, not a slide.

Screen recording plus voice-over, at 1080p. Shots marked **LIVE** need Ollama running `gemma3:4b`. On the 8 GB laptop a
close-up took 12-238 s, so record with `python run.py --source samples --timeout 600`, close other apps first, and speed up
the waits in editing with "sped up ×N" on screen. The missing-screw part is the quickest live shot: 2 close-ups.
The fault shot is easy to get for real: run with the default 120 s limit on this laptop and the timeouts and circuit
breaker happen on their own, which is exactly what our first real run did.

Setup: `python make_samples.py`, delete `runtime/`, start `streamlit run dashboard.py --server.address 0.0.0.0`, open the
dashboard on a phone too, and keep a second terminal ready for `run.py`.

| Time | Rubric | On screen | Voice-over |
|---|---|---|---|
| 0:00-0:15 | Impact | Title, then a photo or sketch of manual end-of-line inspection | "Missing screws and cracks still get caught by eye. A vision model can read a part, but it can't be trusted with the line." |
| 0:15-0:40 | Architecture | `media/2_tree.png`, highlight the yellow node | "So the model is a sensor and a behaviour tree is in charge. One step talks to the AI. Everything else is fixed logic an engineer can sign off." |
| 0:40-1:00 | Architecture | Zoom into `cell/vlm.py` prompt and `cell/validator.py` | "Handoff: one close-up, one question. Handback: only an answer that passes the validator, with a description and one allowed word, gets to the rules." |
| 1:00-1:10 | Offline | **Turn Wi-Fi off** on camera (taskbar), then start the run | "Wi-Fi is off. The model runs on this laptop." |
| 1:10-1:40 | End to end | **LIVE** `python run.py --source samples`: good part to PASS, missing screw to REJECT; dashboard shows the red box on the hole and the model's words | "Good part: nine regions checked, pass. Missing screw: the model describes the empty hole, the rules reject it, the check stops early." |
| 1:40-2:00 | Edge cases | Dark and blurry parts go to review with zero AI calls; the garbage-answer part is rejected by the validator | "Bad photo: the AI is never asked. A rambling answer gets rejected. Both go to a human." |
| 2:00-2:20 | Fault | **LIVE** quit Ollama mid-run (tray icon, then Quit): parts go to review, then "degraded mode"; restart Ollama and the probe recovers | "Kill the model server. The line doesn't hang and doesn't guess. After three failures it stops waiting, sends parts to review, and probes until the model is back." |
| 2:20-2:35 | Fault | Press the dashboard E-STOP **on the phone** during a slow AI step; the terminal shows the decision but the lane is "held" | "E-stop from the operator's phone. It's checked again right before the gate, so the decision is logged and nothing moves. The hardware e-stop is still the real one." |
| 2:35-2:50 | Override | Type a name, Override: PASS in the review queue; History shows "AI path: REVIEW, final: PASS, by Sam" | "The operator has the last word, and it's on the record with their name." |
| 2:50-3:00 | Execution | `pytest` green (18 passed), then the limits slide | "Eighteen safety tests run in seconds. Limits: synthetic parts, fixed camera, simulated diverter." |

Do not narrate numbers you have not re-measured on the recording machine.
