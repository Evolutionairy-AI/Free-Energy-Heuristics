# Confirmatory Analysis — FEH H1 (registered eq. 6.1' ITT)

**Data**: 7875 scored cells (0 gen errors), 7 models, 45 items.

**Sampler**: official fit = 4000×4 @ target_accept=0.99 (post-hoc draw increase from pre-registered 2000×4 @ 0.95 for convergence only; verdict identical, model/gate unchanged).

**Convergence**: R̂_max=1.0056, ESS_min=1692 → ✅ converged (all R̂<1.01).

## PRIMARY — eq. 6.1' assigned-length ITT (the registered gate)

| quantity | value |
|---|---|
| **Pr(β3 < 0 \| data)** | **1.0000** (gate: >0.95) |
| **robust short→long pp-drop** | **+17.30** [+7.68, +25.55] (gate: >6) |
| β3 median | -0.692 |
| β1 median (low-regime length slope) | -0.183 |
| Pr(β3<0 ∧ \|β3\|>\|β1\|) (descriptive) | 0.9054 |

### VERDICT (primary): **CONFIRMED**

- Directional gate Pr(β3<0)>0.95: ✅ PASS (1.0000)
- Magnitude gate robust pp-drop>6: ✅ PASS (+17.30 pp)

## SECONDARY — R7 realized-steps (eq. 6.1, reported not gated)

- verdict=falsified; Pr(β3<0)=0.000; β3=+0.416 [+0.271,+0.560]; pp-drop=+16.5

## Descriptive (the H1 signature)

| condition | high cbd-correct | low accuracy |
|---|---|---|
| C1 | 0.7714 | 0.8367 |
| C2 | 0.6654 | 0.8 |
| C3 | 0.647 | 0.8163 |
| C4 | 0.6442 | 0.8041 |
| C5 | 0.6212 | 0.8571 |

High-regime short→long drop **12.7 pp** vs low-regime **1.73 pp** → difference-in-differences **10.97 pp**.

| model | n | high drop pp | low drop pp |
|---|---|---|---|
| claude-sonnet-4-5-20250929 | 1125 | 2.58 | 0.0 |
| gpt-4o-2024-11-20 | 1125 | 10.48 | 1.07 |
| mistral:7b-instruct | 1125 | 0.32 | 8.93 |
| phi3.5 | 1125 | 3.55 | 23.93 |
| qwen2.5:14b | 1125 | 37.9 | 1.43 |
| qwen2.5:32b | 1125 | 15.48 | -8.93 |
| qwen2.5:7b | 1125 | 18.55 | -14.29 |

