# K3 Training-Data-Sparsity Pre-Screen — FEH-79 K2 items (pool v0.2)

Search engine: Mojeek (https://www.mojeek.com/search), exact-quoted phrases. (DuckDuckGo HTML was attempted first but now serves an anti-bot page in response to scripted GET/POST.)

K3 verdict thresholds (first-page result count): 0 → K3-pass-clean, 1-4 → K3-pass-marginal, ≥5 → K3-fail-contaminated.

Note: K3-fail-contaminated does not necessarily mean training-data contamination — many coined names collide with real proper nouns (e.g., 'Karsk' is a Norwegian place name, 'Korlin' is a surname). Each contaminated item should be audited by inspecting the returned titles below.

**Verdict summary**:
- **K3-pass-clean**: 15 / 19
- **K3-pass-marginal**: 4 / 19
- **K3-fail-contaminated**: 0 / 19
- **K3-error**: 0 / 19


## Per-item results

| frame | verdict | max-filtered | max-raw | queries |
|-------|---------|--------------|---------|---------|
| K2-001 | K3-pass-clean | 0 | 0 | `"Quogard" sintering; "Quogard people"` |
| K2-002 | K3-pass-clean | 0 | 0 | `"Vasrenian" threadcounting; "Vasrenian people"` |
| K2-003 | K3-pass-clean | 0 | 10 | `"anticipative formalism"` |
| K2-004 | K3-pass-clean | 0 | 2 | `"zorinite"` |
| K2-005 | K3-pass-clean | 0 | 10 | `"Drelvian-Lindner"; "Drelvian-Lindner interferometer"; "Institute for Photonic S` |
| K2-006 | K3-pass-marginal | 2 | 10 | `"drandology"; "fnobel"` |
| K2-008 | K3-pass-clean | 0 | 10 | `"Vermex" game; "game of Vermex"` |
| K2-009 | K3-pass-clean | 0 | 10 | `"planet Karsk"; "Karskian"` |
| K2-010 | K3-pass-marginal | 4 | 10 | `"Lassic Mathematicians"; "Federation of Lassic"` |
| K2-011 | K3-pass-marginal | 1 | 10 | `"Yethra people"; "Yethra"` |
| K2-012 | K3-pass-clean | 0 | 0 | `"tirstent"` |
| K2-013 | K3-pass-clean | 0 | 0 | `"Prendic" anticipatory autonomy; "anticipatory autonomy" Prendic` |
| K2-014 | K3-pass-clean | 0 | 0 | `"Physarum consilium"` |
| K2-015 | K3-pass-marginal | 2 | 10 | `"Frobenian field theory"` |
| K2-016 | K3-pass-clean | 0 | 0 | `"Volnari" trust quotient; "Volnari civic"` |
| K2-017 | K3-pass-clean | 0 | 0 | `"Larcot" phantom debt; "Larcot legal tradition"` |
| K2-018 | K3-pass-clean | 0 | 4 | `"tenebrith"` |
| K2-019 | K3-pass-clean | 0 | 10 | `"Korlin people" memory; "Korlin people"` |
| K2-020 | K3-pass-clean | 0 | 10 | `"Iridian Council"` |
