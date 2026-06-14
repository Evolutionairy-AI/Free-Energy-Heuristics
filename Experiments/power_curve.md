# Power Curve — Amended Confirmatory Gate

Compares the pre-registered ORIGINAL gate (Pr(b3<0 ∧ |b3|>|b1|)>0.95 ∧ pp-drop>10 ∧ low-slope≥0) against the proposed AMENDED gate (Pr(b3<0)>0.95 ∧ robust pp-drop>FLOOR) over a grid of true interaction sizes. 30 reps/point at the full design (5×40×5×3=3000 obs). Sampler: {'draws': 600, 'tune': 600, 'chains': 2}.

**Generated**: 2026-05-29 12:20

## Confirmation rate by true effect size

| true b3 | true pp-drop | orig gate | amended f=6 | amended f=8 | amended f=10 | mean Pr(b3<0) |
|---|---|---|---|---|---|---|
| +0.00 | -6.5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.613 |
| -0.15 | +3.3 | 0.00 | 0.30 | 0.23 | 0.20 | 0.916 |
| -0.25 | +9.7 | 0.03 | 0.47 | 0.33 | 0.33 | 0.983 |
| -0.40 | +19.3 | 0.13 | 0.90 | 0.87 | 0.87 | 1.000 |

*Row b3=+0.00 is the null → confirmation rate there is the Type I error rate. The target effect is b3=-0.25.*

## Recommendation

**Adopt magnitude floor = 6 pp.** At this floor the amended gate has **power = 0.47** at the target effect (b3=-0.25) and **Type I = 0.00** under the null. (no floor hit 0.80 power; best Type-I-valid floor shown.)

Compare: the original full-reversal gate has power 0.03 at the same target effect.

## Proposed amended primary test (for OSF addendum)

> **H1 (amended).** The interaction coefficient β3 in eq. 6.1 is negative: more reasoning degrades accuracy in the high-regime bin relative to the low-regime bin.
>
> **Confirmation:** Pr(β3 < 0 | data) > 0.95 AND the posterior median robust implied accuracy drop exceeds 6.0 percentage points (high-regime, 10th→90th percentile step range, anchored at the empirical high-regime base rate).
>
> **Reported as effect sizes (no longer gating):** the full-reversal probability Pr(β3<0 ∧ |β3|>|β1|), the within-bin slopes, and the low-regime slope sign.
>
> **Falsification:** Pr(β3 ≥ 0 | data) > 0.95, or Pr(β3<0)>0.95 with robust implied drop < 3 pp (directional but negligible).

