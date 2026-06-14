# Calibration Analysis — FEH H1 effect size & σ_g (real local-model data)

> Pre-data feasibility estimate, **not** the confirmatory test. Regime = item category (K = high, R/A = low). See `Calibration_protocol.md`.

**Data**: 1890 scored responses — 1260 high-regime, 630 low-regime — across 3 models (llama3.1:8b, mistral:7b-instruct, qwen2.5:7b).

**Convergence**: R̂_max = 1.010, ESS_min = 987 → ⚠️ NOT converged.

## Headline numbers

| quantity | value |
|---|---|
| Pr(β3 < 0) | **0.189** |
| β3 median (95% CI) | +0.115 [-0.139, +0.370] |
| β1 median (95% CI) | -0.208 [-0.623, +0.241] |
| low-regime slope (median β1) | -0.208 |
| high-regime slope (median β1+β3) | -0.091 |
| **robust implied high-regime pp-drop** | **+5.2 pp** [-18.4, +27.0] |
| registered min→max pp-drop | +9.7 pp |
| **σ_g (model-slope SD)** | **0.168** [0.009, 0.980] |
| calibration verdict (registered rule) | inconclusive |
| calibration verdict (amended rule) | inconclusive |

*Verdicts are descriptive here — the registered gate applies to the full panel, not this 3-model 7–8 B calibration.*

## cbd-correctness by CoT condition (the H1 signature)

*H1 predicts high-regime cbd-correctness falls C1→C5; low-regime accuracy should not.*

| condition | high-regime cbd-correct | low-regime accuracy |
|---|---|---|
| C1 | 0.71 | 0.659 |
| C2 | 0.583 | 0.706 |
| C3 | 0.583 | 0.69 |
| C4 | 0.639 | 0.667 |
| C5 | 0.583 | 0.69 |

## Per-model high-regime step slope (median g_m)

| model | median slope contribution g_m |
|---|---|
| llama3.1:8b | -0.011 |
| mistral:7b-instruct | -0.040 |
| qwen2.5:7b | +0.056 |

## Effect heterogeneity across the 28 high-regime items

- **21** items show the predicted cbd-collapse (C5 < C1 − 0.10)
- **2** items floored (mean cbd-correct < 0.10 — model never recognized cbd)
- **3** items ceilinged (mean cbd-correct > 0.90 — robustly recognized cbd)

| item | C1 | C2 | C3 | C4 | C5 | mean | direction |
|---|---|---|---|---|---|---|---|
| K1-001 | 0.78 | 0.67 | 0.67 | 0.89 | 0.78 | 0.76 | flat |
| K1-002 | 0.67 | 0.67 | 0.56 | 0.56 | 0.56 | 0.60 | drop |
| K1-003 | 1.00 | 0.89 | 1.00 | 1.00 | 0.78 | 0.93 | drop |
| K1-004 | 0.89 | 0.56 | 0.56 | 0.78 | 0.67 | 0.69 | drop |
| K1-006 | 0.67 | 0.78 | 0.56 | 0.67 | 0.67 | 0.67 | flat |
| K1-007 | 0.89 | 0.78 | 0.78 | 0.78 | 0.67 | 0.78 | drop |
| K1-008 | 1.00 | 0.89 | 0.89 | 0.78 | 0.89 | 0.89 | drop |
| K1-009 | 0.78 | 0.56 | 0.56 | 0.78 | 0.56 | 0.64 | drop |
| K1-010 | 0.89 | 0.56 | 0.67 | 0.67 | 0.67 | 0.69 | drop |
| K1-011 | 1.00 | 0.89 | 0.78 | 1.00 | 0.78 | 0.89 | drop |
| K1-012 | 1.00 | 0.89 | 0.89 | 0.89 | 0.89 | 0.91 | drop |
| K1-013 | 0.89 | 0.89 | 0.89 | 0.78 | 0.89 | 0.87 | flat |
| K1-014 | 0.67 | 0.67 | 0.67 | 0.56 | 0.67 | 0.64 | flat |
| K1-015 | 0.78 | 0.56 | 0.67 | 0.67 | 0.67 | 0.67 | drop |
| K1-016 | 0.89 | 0.78 | 0.78 | 0.78 | 0.78 | 0.80 | drop |
| K1-017 | 0.89 | 0.78 | 0.56 | 0.78 | 0.44 | 0.69 | drop |
| K1-018 | 0.78 | 0.67 | 0.67 | 0.78 | 1.00 | 0.78 | rise |
| K1-019 | 1.00 | 0.89 | 1.00 | 0.89 | 0.89 | 0.93 | drop |
| K1-020 | 0.67 | 0.78 | 0.67 | 0.67 | 0.67 | 0.69 | flat |
| K2-003 | 0.56 | 0.33 | 0.22 | 0.44 | 0.33 | 0.38 | drop |
| K2-005 | 0.33 | 0.00 | 0.00 | 0.11 | 0.00 | 0.09 | drop |
| K2-006 | 0.00 | 0.11 | 0.33 | 0.11 | 0.22 | 0.16 | rise |
| K2-008 | 0.22 | 0.33 | 0.33 | 0.56 | 0.11 | 0.31 | drop |
| K2-012 | 0.56 | 0.33 | 0.22 | 0.56 | 0.44 | 0.42 | drop |
| K2-015 | 0.44 | 0.33 | 0.22 | 0.44 | 0.22 | 0.33 | drop |
| K2-016 | 0.78 | 0.33 | 0.78 | 0.78 | 0.56 | 0.64 | drop |
| K4-002 | 0.22 | 0.00 | 0.00 | 0.11 | 0.00 | 0.07 | drop |
| K4-003 | 0.67 | 0.44 | 0.44 | 0.11 | 0.56 | 0.44 | drop |

