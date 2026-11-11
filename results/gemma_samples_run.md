# Real Gemma run on the samples (2026-11-12)

| | |
|---|---|
| model | gemma3:4b |
| ollama_version | 0.40.16 |
| machine | Intel64 Family 6 Model 154 Stepping 4, GenuineIntel / Windows 11 |
| timeout_s_per_closeup | 600.0 |
| parts_run | ['missing_screw_01', 'crack_01', 'good_01'] |
| warm_up_seconds | 66.3 |
| total_minutes | 27.0 |
| model_calls | 16 |
| seconds_per_call_median | 138.21 |
| seconds_per_call_min | 11.8 |
| seconds_per_call_max | 238.46 |

| Image | Expected | Cell decision | Reason |
|---|---|---|---|
| crack_01.png | REJECT (surface center) | REJECT | surface damage at surface top: AI saw "The metal plate surface exhibits several fine lines and scratches, indicating wear and use on a manufacturing line." |
| good_01.png | PASS | PASS | all 9 regions checked and OK |
| missing_screw_01.png | REJECT (screw top-right) | REJECT | missing screw at screw top-right: AI saw "I see a gray metal plate with a circular, black hole in the center." |

## Every model answer

- **crack_01.png · screw top-left** (138.21 s): I see a gray metal plate with a white plus sign (+) symbol representing a screw head installed in a circular hole. /  / ANSWER: SCREW   (a screw head is installed)
- **crack_01.png · screw top-right** (238.46 s): I see a gray metal plate with a black plus sign (+) symbol representing a screw head positioned in a circular recess. /  / ANSWER: SCREW   (a screw head is installed)
- **crack_01.png · screw bottom-left** (224.53 s): I see a gray, circular button with a black plus sign (+) inside, likely representing a screw hole. / ANSWER: SCREW   (a screw head is installed)
- **crack_01.png · screw bottom-right** (183.18 s): I see a gray metal plate with a black plus sign (+) symbol representing a screw head installed in a circular hole. /  / ANSWER: SCREW   (a screw head is installed)
- **crack_01.png · surface top** (146.43 s): The metal plate surface exhibits several fine lines and scratches, indicating wear and use on a manufacturing line. / ANSWER: DAMAGED
- **good_01.png · screw top-left** (12.86 s): I see a gray metal plate with a white plus sign (+) symbol representing a screw head positioned in a circular outline. / ANSWER: SCREW   (a screw head is installed)
- **good_01.png · screw top-right** (12.42 s): I see a gray metal plate with a circular icon featuring a plus sign (+) within a black circle. / ANSWER: SCREW   (a screw head is installed)
- **good_01.png · screw bottom-left** (19.89 s): I see a gray, circular button with a black plus sign (+) centered within it. / ANSWER: SCREW   (a screw head is installed)
- **good_01.png · screw bottom-right** (15.65 s): I see a gray metal plate with a black plus sign (+) symbol representing a screw head installed in a circular hole. /  / ANSWER: SCREW   (a screw head is installed)
- **good_01.png · surface top** (148.96 s): The metal plate surface appears smooth and uniform with no visible lines, cracks, scratches, or marks. / ANSWER: CLEAN
- **good_01.png · surface left** (201.86 s): The metal plate surface appears smooth and uniform with no visible lines, cracks, scratches, or marks. / ANSWER: CLEAN
- **good_01.png · surface center** (211.39 s): The metal plate surface appears smooth and uniform with no visible lines, cracks, scratches, or marks. / ANSWER: CLEAN
- **good_01.png · surface right** (14.22 s): The metal plate surface appears smooth and uniform with no visible lines, cracks, scratches, or marks. / ANSWER: CLEAN
- **good_01.png · surface bottom** (20.36 s): The metal plate surface appears smooth and uniform with no visible lines, cracks, scratches, or marks. / ANSWER: CLEAN
- **missing_screw_01.png · screw top-left** (14.87 s): I see a gray metal plate with a white plus sign (+) symbol representing a screw head positioned in a circular outline. / ANSWER: SCREW   (a screw head is installed)
- **missing_screw_01.png · screw top-right** (11.8 s): I see a gray metal plate with a circular, black hole in the center. / ANSWER: EMPTY
