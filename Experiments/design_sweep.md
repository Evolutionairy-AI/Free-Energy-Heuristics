# Design Sweep — More Items vs Power (amended gate, floor 8pp)

> **⚠️ Calibration caveat (added post-hoc).** This run used `s_coef = -0.25/1.271`,
> which makes the quartile **effective β3 ≈ −0.50** (a STRONG effect) with realistic
> item-level effect heterogeneity — NOT the −0.25 modest target. So the absolute
> power numbers below are NOT the modest-target power (that remains ~0.47, per
> power_curve.md). **What IS valid here is the comparative, paired conclusion:
> widening regime bins to add items does not improve power** (quartile ≥ tercile ≈
> continuous > median). Item heterogeneity also appears to depress power
> substantially even at a strong average effect (0.55 here vs ~0.90 in the
> homogeneous power_curve) — a separate concern worth modeling an item random
> slope for. Code corrected to `/(2×1.271)`; re-run for clean modest-target absolutes.

Continuous-truth DGP (item reversal scales with regime score, so wider bins add weaker-effect items). Four analyses compared on the same data each rep.

**Monte-Carlo reps**: 20  |  **n_reps (replications)**: 3  |  **sampler**: {'draws': 600, 'tune': 600, 'chains': 2}  |  **2026-05-29 14:45**

## Power at the target effect (and Type I under the null)

| analysis | items/bin | total obs | **power** | Type I |
|---|---|---|---|---|
| quartile | 20 | 3000 | **0.55** | None |
| tercile | 26 | 3900 | **0.50** | None |
| median | 39 | 5850 | **0.35** | 0.00 |
| continuous | 79 | 5925 | **0.50** | 0.00 |

*Power = confirm rate at the target effect; Type I = confirm rate under the null (b3-scaling = 0).*

## Read

- Best analysis: **quartile** (power 0.55). Quartile anchor: 0.55 (should ≈0.47 from power_curve).

- **No analysis reached 0.80** at 3 replications. More items helps but isn't sufficient alone; consider combining with more replications or the model-structure change.

