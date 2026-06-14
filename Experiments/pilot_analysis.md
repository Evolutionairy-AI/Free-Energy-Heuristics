# §4.9 Pilot Analysis — FEH-79 v0.3, Mistral-7B-Instruct

**Cells**: 150. **Extraction success**: 150/150 = 100.0%

## Pilot success criteria (per §4.9.2)

- **P1** (K mean Knightian-signal > R+A mean, rank-order on composite of σ_b + cbd_rate; K3 open-ended excluded): ⚠️ fail (K=0.367, R+A=0.4)
- **P2** (step-count κ ≥ 0.7): out-of-scope for this script — requires human coding
- **P3** (extract ≥ 90%): ✅ pass (100.0%)
- **P4** (≥1 K item non-monotone σ_b trajectory): ✅ pass (3 of 8)

## Per-frame summary

| frame | cat | K-signal | mean σ_b | mean cbd-rate | mean steps C1→C5 | acc |
|-------|-----|----------|----------|---------------|------------------|-----|
| A-003 | aleatory | **0.33** | 0.33 | 0.00 | 1→5→12→16→4 | 0.67 *(open-ended; sig_b not applicable)* |
| K1-001 | non-recurrent- | **1.00** | 0.00 | 1.00 | 0→4→10→15→2 | - |
| K1-005 | non-recurrent- | **0.20** | 0.20 | 0.20 | 1→7→11→19→11 | - |
| K2-005 | novel-syntheti | **0.07** | 0.07 | 0.07 | 0→1→8→11→2 | - |
| K2-006 | novel-syntheti | **0.40** | 0.20 | 0.40 | 0→2→10→7→5 | - |
| K3-001 | open-ended-dil | **0.67** | 0.67 | 0.00 | 1→4→10→16→12 | - *(open-ended; sig_b not applicable)* |
| K3-005 | open-ended-dil | **0.67** | 0.67 | 0.27 | 1→5→8→17→7 | - *(open-ended; sig_b not applicable)* |
| K4-003 | strategic-unce | **0.20** | 0.07 | 0.20 | 1→5→6→11→6 | - |
| K4-004 | strategic-unce | **0.33** | 0.33 | 0.00 | 0→7→12→22→12 | - |
| R-001 | reference | **0.47** | 0.47 | 0.00 | 0→2→12→14→5 | 0.07 *(open-ended; sig_b not applicable)* |

## CBD-rate trajectory across conditions (K items only)

*Theorem 2.6.1's prediction: under meta-uncertainty, more reasoning may push the model out of cbd-recognition into substantive confabulation. Drop in cbd-rate from C1 to C5 = directional evidence for the regime shift.*

| frame | C1 | C2 | C3 | C4 | C5 | trajectory |
|-------|----|----|----|----|----|------------|
| K1-001 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ≈ |
| K1-005 | 0.67 | 0.33 | 0.00 | 0.00 | 0.00 | ↓ (drop) |
| K2-005 | 0.00 | 0.00 | 0.00 | 0.33 | 0.00 | ≈ |
| K2-006 | 0.00 | 0.00 | 0.67 | 0.67 | 0.67 | ↑ (rise) |
| K3-001 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | ≈ |
| K3-005 | 0.33 | 0.33 | 0.33 | 0.00 | 0.33 | ≈ |
| K4-003 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | ↓ (drop) |
| K4-004 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | ≈ |

*σ_b = categorical disagreement = 1 - max-frequency over the 3 replications per (item, condition) cell. Higher = more cross-seed variance, the §3.2 signature (b) of meta-uncertainty.*

*Pilot is descriptive only per §4.9.3. The confirmatory H1 test is conducted on the full data set, not the pilot.*
