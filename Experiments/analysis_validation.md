# Pre-Data Validation of the Confirmatory Analysis Pipeline

Validates `confirmatory_analysis.py` (pre-reg eq. 6.1) by parameter recovery and simulation-based operating characteristics on synthetic data at the full planned design (5 models x 40 items x 5 conditions x 3 reps).

**Sampler**: preregistered (4 chains, 2000+2000)  
**Generated**: 2026-05-29 10:28

## (A) Parameter recovery + decision-rule verdicts

**Verdicts correct in all scenarios:** ❌ NO  
**Converged (R̂<1.01, ESS>400) in all scenarios:** ✅ yes

| scenario | verdict | expected | ok | Pr(H1) | b3 median (true) | pp-drop | R̂max | ESSmin |
|---|---|---|---|---|---|---|---|---|
| h1_strong | confirmed | confirmed | ✅ | 0.978 | -0.706 (-0.60) | 40.0 | 1.000 | 1611 |
| h1_boundary | falsified | confirmed | ❌ | 0.601 | -0.298 (-0.25) | 3.4 | 1.000 | 2024 |
| null | falsified | not-confirmed | ✅ | 0.001 | +0.134 (+0.00) | -30.2 | 1.000 | 1827 |
| reversed | falsified | falsified | ✅ | 0.000 | +0.645 (+0.60) | -83.3 | 1.000 | 2022 |

### Coverage of true coefficients (95% CrI)

**h1_strong** — b0=+0.25[-0.22,+0.70]✓, b1=+0.33[-0.05,+0.71]✓, b2=-0.64[-0.96,-0.31]✓, b3=-0.71[-0.87,-0.55]✓, sigma_a=+0.34[+0.16,+0.98]✓, sigma_g=+0.34[+0.16,+0.88]✓, sigma_u=+0.46[+0.34,+0.62]✓

**h1_boundary** — b0=-0.41[-0.92,+0.10]✗(true +0.40), b1=+0.27[-0.03,+0.57]✓, b2=-0.29[-0.62,+0.05]✓, b3=-0.30[-0.46,-0.15]✓, sigma_a=+0.44[+0.22,+1.12]✓, sigma_g=+0.23[+0.09,+0.67]✓, sigma_u=+0.48[+0.35,+0.64]✓

**null** — b0=+0.23[-0.56,+1.01]✓, b1=+0.13[-0.10,+0.37]✓, b2=-0.15[-0.51,+0.20]✓, b3=+0.13[-0.02,+0.29]✓, sigma_a=+0.72[+0.40,+1.55]✓, sigma_g=+0.17[+0.03,+0.56]✓, sigma_u=+0.51[+0.39,+0.69]✓

**reversed** — b0=+0.14[-0.23,+0.52]✓, b1=+0.40[+0.16,+0.63]✓, b2=-0.61[-0.91,-0.32]✓, b3=+0.64[+0.47,+0.82]✓, sigma_a=+0.27[+0.12,+0.78]✓, sigma_g=+0.17[+0.03,+0.54]✓, sigma_u=+0.39[+0.28,+0.54]✓

## (B) Operating characteristics (simulation-based power)

| scenario | reps | confirm rate | falsify rate | inconcl. | mean Pr(H1) | interpretation |
|---|---|---|---|---|---|---|
| h1_boundary | 30 | 0.07 | 0.17 | 0.77 | 0.635 | power = 0.07 |
| null | 30 | 0.00 | 0.10 | 0.90 | 0.086 | false-confirm = 0.00 |

## Verdict

**ATTENTION** — at least one scenario failed recovery, convergence, or returned an unexpected verdict. Inspect the table above before launching data collection.
