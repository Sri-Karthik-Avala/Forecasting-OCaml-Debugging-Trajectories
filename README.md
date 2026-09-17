# Forecasting OCaml Debugging Trajectories

| | |
| --- | --- |
| Final rank | not ranked |
| Domain | NLP |
| Difficulty | Medium |
| Scoring | ↑ Higher is better |
| Compute | CPU |
| Challenge status | Accepted / closed |
| Solutions submitted | 2 |
| Last submission | 2026-08-08 |

## Problem statement

### Overview

Build a **CPU-only NLP system for proactive OCaml debugging support**.

Each row represents one point in a programmer's revision process. You receive up to three chronological attempts at the same OCaml definition. Every attempt contains:

- an anonymized OCaml program;
- a sanitized evaluator diagnostic;
- a broad diagnostic family;
- the number of immediately repeated evaluations compressed into that state.

The final item in the trace is the current erroneous attempt. An **evaluator-clean state** is a later top-level evaluation for which no error was recorded; it does not certify that the complete homework or program is semantically correct. Your system must forecast two things that an IDE or teaching-support queue could use before showing an intervention:

1. **retry survival** — whether the programmer will produce at least 1, 2, and 4 additional erroneous evaluator states before reaching the next evaluator-clean version;
2. **repair footprint** — how the eventual edit mass will be distributed across four consecutive regions of the current program.

The goal is to estimate debugging difficulty and likely repair location from code, diagnostics, and recent revision dynamics. A useful model can help choose between immediate assistance, a lightweight hint, and deferred intervention, while also selecting which part of the current definition deserves attention.

Identifiers, constructors, custom operators, strings, characters, and numeric literals are consistently pseudonymized within a row. The aliases are not stable across rows. Comments, source paths, source-location offsets, course labels, assignment labels, and personal identifiers are absent. Common OCaml syntax and standard-library vocabulary are retained so that type and control-flow structure remain learnable.

Training and test learner groups are disjoint. This evaluates transfer to new programmers instead of memorization of an individual's editing habits. Case identifiers and row order are non-semantic.

Your complete solution must run on CPU and finish within **1.5 hours**, including training, validation, inference, and writing the submission. Up to **62 GB RAM** is available.

### Dataset

```
dataset/public/

├── train.csv

├── test.csv

└── sample_submission.csv
```

The prepared release contains exactly **1800 training rows** and **650 test rows**. JSON-valued cells use compact JSON serialization.

`train.csv`

```
case_id,learner_group,trace,risk_1,risk_2,risk_4,repair_profile
```

Columns:

- `case_id`

   - Data type: string
   - Opaque identifier unique within `train.csv`.
- `learner_group`

   - Data type: string
   - Opaque group shared by rows from one learner. Use it for grouped validation.
- `trace`

   - Data type: JSON array of 1–3 objects
   - Attempts are ordered from oldest to newest. Each object has exactly these fields:

      - `program`: whitespace-normalized, anonymized OCaml text;
      - `diagnostic`: sanitized diagnostic text, possibly empty for an evaluator-clean earlier state;
      - `error_family`: one of `name`, `syntax`, `typing`, `other`, or `clear`;
      - `repeat_count`: integer from 1 through 9 giving the number of consecutive evaluations represented by this collapsed trace item. The count is local to that state: repeated evaluations of the same program with the same evaluator outcome are merged and increase `repeat_count`; when the program or evaluator outcome changes, a new trace item starts with its own count. Therefore two adjacent trace items may contain different program text even when the later item has `repeat_count > 1`.
   - The last trace item is always the current erroneous state.
- `risk_1`

   - Data type: integer in `{0,1}`
   - `1` when at least one additional distinct erroneous evaluator state occurs before the next evaluator-clean state.
- `risk_2`

   - Data type: integer in `{0,1}`
   - `1` when at least two additional distinct erroneous evaluator states occur before the next evaluator-clean state.
- `risk_4`

   - Data type: integer in `{0,1}`
   - `1` when at least four additional distinct erroneous evaluator states occur before the next evaluator-clean state.

The horizon labels always satisfy:

```
risk_1 >= risk_2 >= risk_4
```

- `repair_profile`

   - Data type: JSON array of four numbers in `[0,1]` summing to `1`
   - If the current program contains `n` lexical tokens numbered `j = 0, ..., n-1`, token `j` belongs to region `min(3, floor(4 × j / n))`. This divides the token order into four consecutive, nearly equal-position regions:

```
[start, early-middle, late-middle, end]
```

- The target is the normalized distribution of lexical insertion, deletion, and replacement mass between the current attempt and its first later evaluator-clean state. A concentrated edit in the beginning of the definition therefore produces most mass in the first component. This is a distribution, not a single location label.

A shortened `trace` example is:

```
[

  {

    "program":"let rec v1 v0 = match v0 with [ ] -> N | v2 :: v3 -> v2 + v1 v3",

    "diagnostic":"Error: This expression has type int but an expression was expected of type int list",

    "error_family":"typing",

    "repeat_count":1

  },

  {

    "program":"let rec v1 v0 = match v0 with [ ] -> N | v2 :: v3 -> v2 :: v1 v3",

    "diagnostic":"Error: This expression has type int but an expression was expected of type int list",

    "error_family":"typing",

    "repeat_count":2

  }

]
```

In this example, the second program differs from the first, but `repeat_count:2` means that the second program state itself was evaluated twice consecutively before the trajectory continued; it does not mean that it matches the preceding trace item.

Aliases such as `v0`, `v1`, `C0`, `'t0`, and masked literals carry no meaning across different rows.

`test.csv`

```
case_id,learner_group,trace
```

The feature columns have the same meanings as in `train.csv`; the four targets are omitted. Every `case_id` in `test.csv` must appear exactly once in the submission. No `learner_group` occurs in both training and test data.

`sample_submission.csv`

```
case_id,risk_1,risk_2,risk_4,repair_profile
```

The sample predicts `0.5` for every retry horizon and the uniform profile `[0.25,0.25,0.25,0.25]`. It is format-valid and intentionally has a score of exactly `0.0` under the skill-normalized metric.

### Submission format

Write predictions to:

```
working/submission.csv
```

The CSV must contain exactly these columns in this order:

```
case_id,risk_1,risk_2,risk_4,repair_profile
```

Requirements:

- include every test `case_id` exactly once;
- do not include additional rows or columns;
- each risk must be a finite number in `[0,1]`;
- each row must satisfy `risk_1 >= risk_2 >= risk_4`;
- `repair_profile` must be a JSON array of exactly four finite numbers in `[0,1]` whose sum differs from `1` by no more than `0.000001`;
- do not submit labels, explanations, code, or confidence intervals in any other field.

Rows may appear in any order.

Example:

```
case_id,risk_1,risk_2,risk_4,repair_profile
case_39e725eba51f8474c3d8,0.71,0.46,0.18,"[0.58,0.22,0.14,0.06]"
case_39e725eba51f8474c3d9,0.71,0.46,0.18,"[0.58,0.22,0.14,0.06]"
```

Malformed submissions are rejected rather than clipped or partially scored. Rejection includes wrong columns, missing or extra IDs, duplicate IDs, non-finite values, probabilities outside `[0,1]`, non-monotone retry risks, malformed JSON, a profile of the wrong length, or a profile that does not sum to one.

### Evaluation

The score rewards **calibrated horizon forecasting** and **repair-region forecasting**, with equal top-level influence from those two target families. All averaging is macro-averaged by `learner_group`, so a learner with many rows cannot dominate the result.

### 1. Retry-horizon skill

For learner group `g` and horizon `h ∈ {1,2,4}`, let `p_i,h` be the submitted probability and `y_i,h` the binary target. Define group Brier error:

```
B_g,h = mean over rows i in g of (p_i,h - y_i,h)^2
```

Convert it to skill relative to the neutral probability `0.5`:

```
H_g,h = clip(1 - B_g,h / 0.25, 0, 1)
```

Then macro-average across the `G` test learner groups:

```
H_h = (1 / G) × sum over groups g of H_g,h
```

A constant `0.5` forecast has squared error `0.25` for every binary outcome and therefore receives zero retry skill. Perfect probabilities receive skill `1`.

### 2. Repair-profile skill

For one row, let:

```
p = submitted four-region distribution

q = true four-region distribution

u = [0.25, 0.25, 0.25, 0.25]
```

Define:

```
E_i  = sum from k=1 to 4 of (p_k - q_k)^2

E0_i = sum from k=1 to 4 of (u_k - q_k)^2

R_i  = clip(1 - E_i / E0_i, 0, 1)
```

Every test target is non-uniform, so `E0_i` is strictly positive. First average `R_i` within each learner group, then average those group values:

```
R = learner-macro mean of R_i
```

The uniform profile receives zero skill on every row. An exact profile receives skill `1`.

### 3. Final score

First combine the three nested horizon skills into one survival-curve skill:

```
S = (H_1 × H_2 × H_4)^(1/3)
```

This treats the three horizons as evaluations of one survival distribution rather than counting them as three independent task families. Next combine survival forecasting and repair localization symmetrically:

```
A = (S + R) / 2

G = sqrt(S × R)

score = 100 × (0.25 × A + 0.75 × G)
```

### Not allowed

- External APIs or hosted inference services.
- External novice-programmer telemetry, OCaml error-to-fix pair collections, assignment solutions, course repositories, parallel versions of the challenge cases, or manually constructed test labels.
- De-anonymizing learner groups or linking public rows to external records.
- Manual labeling or transcription of test rows.
- Hard-coding predictions by `case_id`.
- Using case identifiers, row order, CSV serialization, alias numbering, or evaluator behavior as substitutes for modeling the public trace.
