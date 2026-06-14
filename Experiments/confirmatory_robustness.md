# Confirmatory Robustness Pass — FEH H1

Post-confirmation supporting analyses. Does NOT alter the gated verdict.

**Primary ITT export** (reproduces verdict: ✅): β3=-0.692, Pr(β3<0)=1.0000, robust drop=17.30pp [7.68,25.55].

**R7 realized-steps export**: β3=+0.416 [+0.271,+0.560], Pr(β3<0)=0.000.

**IV-R7 (instrumented)**: per-step interaction -0.526 [-0.857,-0.201], Pr<0=0.999. Wald/2SLS IV; instrument = assigned-length contrast. Corrects the endogeneity that flipped the naive R7 sign. Sign-consistent with the primary ITT confirms the per-step interaction is negative once realized steps are instrumented.


## R5 — per-model ITT (β3, item REs only)

| model | n | β3 median | 95% CrI | Pr(β3<0) | robust drop pp |
|---|---|---|---|---|---|
| claude-sonnet-4-5-20250929 | 1125 | -0.797 | [-2.085, +0.479] | 0.884 | +6.1 |
| gpt-4o-2024-11-20 | 1125 | -0.889 | [-2.145, +0.392] | 0.913 | +21.6 |
| mistral:7b-instruct | 1125 | +0.517 | [-0.357, +1.395] | 0.122 | +0.9 |
| phi3.5 | 1125 | +0.979 | [+0.187, +1.788] | 0.009 | +6.3 |
| qwen2.5:14b | 1125 | -2.227 | [-3.286, -1.167] | 1.000 | +53.1 |
| qwen2.5:32b | 1125 | -2.923 | [-4.050, -1.810] | 1.000 | +22.8 |
| qwen2.5:7b | 1125 | -2.859 | [-3.806, -1.935] | 1.000 | +30.3 |

## R6 — + item random slope on length
β3=-0.468 [-1.030,+0.150], Pr(β3<0)=0.935 (R̂_b3=1.0009).


## R1 — paragraph-level step segmentation (secondary)
β3=-0.391 [-0.537,-0.247], Pr(β3<0)=1.000. (Compare sentence-level R7.)


## F7.1 per-condition cluster-bootstrap CIs (cluster=item)

| condition | high mean [95%] | low mean [95%] |
|---|---|---|
| C1 | 0.771 [0.690,0.846] | 0.837 [0.718,0.943] |
| C2 | 0.665 [0.574,0.752] | 0.800 [0.688,0.896] |
| C3 | 0.647 [0.560,0.731] | 0.816 [0.729,0.898] |
| C4 | 0.644 [0.562,0.721] | 0.804 [0.724,0.867] |
| C5 | 0.621 [0.537,0.706] | 0.857 [0.786,0.922] |

## DEFERRED (NOT computed here)

- **R2_continuous_regime_binning**: needs assembled per-item regime score
- **R3_drop_calibration_component**: needs assembled per-item regime score
- **R4_heldout_calibration_items**: needs assembled per-item regime score
- **regime_score_vs_category_alignment**: needs assembled per-item regime score
- **note**: Regime is category-assigned in the executed run; the continuous regime-score variants + alignment are a dedicated regime-score pass, NOT computed here. Do not report these as run.
