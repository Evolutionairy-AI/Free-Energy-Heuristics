# Frontier feasibility read — H1 off the small-model floor

Scored 672 cells (448 high, 224 low) from 2 frontier models, conditions C1–C4, 2 reps. Scoring identical to calibration (is_cbd / gold_match, pool v0.3).

**H1 prediction:** high-regime cbd-correctness falls from assigned-SHORT (C1) to assigned-LONG (C2–C4); low-regime accuracy stays flat. The difference-in-differences (high drop − low drop) is the H1 effect; it should be **> 0**.


## claude-sonnet-4-5-20250929

| regime | C1 | C2 | C3 | C4 | short(C1) | long(C2–4) | short→long drop |
|---|---|---|---|---|---|---|---|
| high (cbd-correct) | 0.464 | 0.375 | 0.393 | 0.304 | 0.464 | 0.357 | **+0.107** [-0.018, +0.238] |
| low (accuracy) | 1.0 | 1.0 | 1.0 | 1.0 | 1.000 | 1.000 | +0.000 |

- **H1 difference-in-differences (high drop − low drop): +0.107 [-0.024, +0.232]** (CI includes 0)
- floor items (mean cbd<0.10): 9/28 ; ceiling (>0.90): 4/28
- realized steps by condition (C1 compliance): C1=4.7 C2=4.0 C3=3.8 C4=6.3

## gpt-4o-2024-11-20

| regime | C1 | C2 | C3 | C4 | short(C1) | long(C2–4) | short→long drop |
|---|---|---|---|---|---|---|---|
| high (cbd-correct) | 0.786 | 0.768 | 0.643 | 0.643 | 0.786 | 0.685 | **+0.101** [+0.006, +0.208] |
| low (accuracy) | 1.0 | 0.964 | 1.0 | 1.0 | 1.000 | 0.988 | +0.012 |

- **H1 difference-in-differences (high drop − low drop): +0.089 [-0.006, +0.202]** (CI includes 0)
- floor items (mean cbd<0.10): 3/28 ; ceiling (>0.90): 16/28
- realized steps by condition (C1 compliance): C1=0.0 C2=3.3 C3=12.7 C4=19.4

## Pooled (both frontier models)

- high-regime short→long drop: **+0.104** [+0.021, +0.188] (short 0.625 → long 0.521)
- low-regime short→long drop: +0.006 (short 1.000 → long 0.994)
- **H1 difference-in-differences: +0.098 [+0.018, +0.182]** (SUPPORTS H1)

## Comparison to the local calibration

- Local panel (3× 7–8B): high short→long drop ≈ 7 pp (clean C1→C4); many K2/K4 items floored; registered realized-steps model returned the WRONG sign (the reason for Amendment 2).
- Frontier read: see DiD above. If the frontier DiD CI excludes 0 with fewer floored items, H1 holds off the floor → the central claim is not a small-model artifact.
