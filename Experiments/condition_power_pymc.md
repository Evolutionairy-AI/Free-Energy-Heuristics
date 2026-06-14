# Condition-ITT power — full-PyMC confirmation (#52)

> Registered Bayesian gate (eq. 6.1', Pr(b3<0)>0.95 AND robust short->long pp-drop>6) vs the IRLS proxy, PAIRED on identical calibration-anchored simulated data.

**Design**: 5 models x 31+12 items x reps=5 x short:long 1:4 = 5375 obs/sim. Sampler: draws=500, tune=500, chains=4. 40 sims/cell.

**High-regime short base rates** (resampled measured C1): mean=0.681, floored=1, ceiled=4.

| true pp | power (Bayesian gate) | power (IRLS proxy) | median Pr(b3<0) | median pp-drop | median b3 | non-conv | s/fit |
|---|---|---|---|---|---|---|---|
| 0.0 | **0.05** | 0.05 | 0.607 | 0.9 | -0.043 | 40/40 | 104.4 |
| 7.0 | **0.82** | 0.82 | 0.995 | 9.8 | -0.426 | 40/40 | 98.6 |
| 10.0 | **1.00** | 1.00 | 1.000 | 15.9 | -0.676 | 40/40 | 99.1 |

Notes:
- true pp = TRUE mean high-regime short->long accuracy drop in the DGP.
- measured calibration effect range: ~7pp (clean C1->C4) to ~13pp (full C1->C5); the registered design is locked at reps=5.
- Bayesian gate = the ACTUAL registered gate (random model intercept+slope over 5 clusters, item random intercept, N(0,1) shrinkage on b3). IRLS proxy = condition_power.py's fixed-effects Wald gate, re-applied on the SAME data with the SAME empirical short-rate anchor.
- Type I error = the power row at true pp = 0.0 (should be small).
- non-conv = sims with R-hat >= 1.01 or ESS <= 400 (registered convergence gate); power is over all sims (a real run re-samples a non-converged fit, so excluding them would only raise power).
