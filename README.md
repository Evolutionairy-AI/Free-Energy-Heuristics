# Free Energy Heuristics

**Fast-and-Frugal Cognition as Active Inference Under Uncertain Precision**

Alex Bogdan — EvolutionAIry AI Inc., Toronto, Canada

<!-- ARXIV: replace the line below with the badge/link once the arXiv ID is assigned, e.g.
     **Preprint:** [arXiv:2506.XXXXX](https://arxiv.org/abs/2506.XXXXX) -->
**Preprint:** arXiv:XXXX.XXXXX — *to be assigned (placeholder; will be updated on submission).*

This repository provides public access to the code, benchmark, experimental data,
and analysis behind the *Free Energy Heuristics* (FEH) paper. The confirmatory
study was pre-registered on OSF ([osf.io/9dvzb](https://osf.io/9dvzb)) before any
data were collected.

---

## The claim

Contemporary practice treats inference-time compute as a free dial: more
chain-of-thought is assumed to be neutral at worst and helpful at best. FEH
predicts the opposite in a specific, bounded regime. Under genuine uncertainty
about prior precision — the *Knightian* regime, where the question admits no
determinate answer — the expected free energy of multi-cue inference is U-shaped,
and its minimum slides toward **fewer** cues as meta-uncertainty grows. The
operational prediction for language models (**H1**) is that, on properly
constructed Knightian items, **more instructed reasoning lowers accuracy**, and
nowhere else.

Formally, H1 is a negative interaction coefficient `β₃ < 0` between assigned
reasoning length and regime, estimated by a pre-registered hierarchical Bayesian
logistic model (eq. 6.1′, intent-to-treat on assigned length).

## The result

The confirmatory run collected **7,875 responses with zero generation errors**
(7 models × 45 items × 5 length conditions × 5 replications). Against the
pre-registered decision gate (`Pr(β₃<0) > 0.95` **and** robust implied accuracy
drop `> 6` percentage points):

| quantity | value | gate |
|---|---|---|
| `Pr(β₃ < 0 \| data)` | **1.0000** | > 0.95 ✅ |
| robust short→long accuracy drop | **+17.30 pp** [7.68, 25.55] | > 6 ✅ |
| β₃ posterior median | −0.692 | — |
| convergence | R̂_max = 1.0056, ESS_min = 1692 | < 1.01 ✅ |

**Verdict: H1 CONFIRMED.**

The effect is **not universal**, and the paper does not claim a universal law.
Refit per model, the interaction is decisive in the three Qwen systems,
directional-but-inconclusive in the two frontier models (GPT-4o, Claude), and
reversed only in Phi-3.5 (with Mistral inconclusive). See §7.4 and Figure 7.4.

## Panel

Seven models, pinned snapshots where applicable: `phi3.5`, `mistral:7b-instruct`,
`qwen2.5:7b`, `qwen2.5:14b`, `qwen2.5:32b` (local, via Ollama) and
`claude-sonnet-4-5-20250929`, `gpt-4o-2024-11-20` (frontier, via API).

---

## Repository layout

```
FEH_Manuscript_v0.2.{md,docx}     The paper (canonical assembled draft)
Appendix_A1_binary_toy_model.md   Binary toy model (Beta-mixture closed form)
Appendix_A2_A3_gaussian_gamma.md  Gaussian-Gamma proofs incl. Prop. A.2.3 (mean-field accuracy)
Appendix_A4_exact_TTB_identity.md Take-the-best exact identity (Theorem 2.7.4)

Preregistration_full_empirical_*  Pre-registration chain + Amendments 1 & 2
feh79_item_pool_v0.*.yaml         The FEH benchmark item pool (v0.4 = confirmatory run)
frame_template.yaml               Item schema
prompt_templates.md               Exact text of the 5 length conditions (C1–C5)
grading_rubric.md                 5-point coherence rubric
step_counter.py                   Reasoning-step counting pipeline

verify_*.py, binary_toy_*.py      Theory verification scripts (Chapter 2 + appendices)
Figures/{PNG,PDF,SVG}/            All six paper figures

Experiments/
  confirmatory_runner.py          The registered run harness (resumable)
  confirmatory_responses.json     ── THE confirmatory dataset (7,875 cells) ──
  confirmatory_run.log            Full run log
  confirmatory_analyze.py         Registered analysis: eq. 6.1′ ITT + R7 secondary
  confirmatory_analysis.py        Model + gate machinery
  confirmatory_analysis_results.{json,md}   H1 verdict (the result above)
  confirmatory_robustness.py      Robustness battery (R1, R5, R6, IV-R7, bootstrap)
  confirmatory_robustness.{json,md}, confirmatory_posterior_draws.json
  fig_f*.py                       Figure-generation scripts (data read live, nothing hand-typed)

  calibration_*, frontier_*, pilot_*    Pre-data feasibility & power phases
  cross_model_check.py, k3_floor_check.py   Benchmark Knightian-ness validation (K2, K3)
  power_curve.py, design_sweep.py, condition_power_pymc.py   Power analysis
```

## Reproducing the analysis

The analysis runs on the committed data; no model calls or API keys are needed.

```bash
cd Experiments
python confirmatory_analyze.py        # refits eq. 6.1′ → confirmatory_analysis_results.{json,md}
python confirmatory_robustness.py     # robustness battery + posterior draws
```

Re-running the **data collection** (`confirmatory_runner.py`) requires the model
backends: Ollama for the local panel and provider API keys for the frontier
models. Keys are read at runtime from `Experiments/API_KEYS/` (not included — see
below) and are never written into any output. With keys absent, the analysis and
figure scripts still run end-to-end against the committed data.

Python deps: `numpy`, `pymc`, `arviz`, `matplotlib`, `pyyaml` (analysis);
`ollama` / provider SDKs only for re-collection.

## Not included

- **API keys** (`Experiments/API_KEYS/`) — credentials, excluded by `.gitignore`.
- Superseded chapter drafts, round-trip artifacts, and personal review copies —
  the paper is published here only as the final assembled `FEH_Manuscript_v0.2`.

## License

Copyright 2026 Alex Bogdan / EvolutionAIry AI Inc. Released under the
[Apache License 2.0](LICENSE) — free use, modification, and redistribution
with attribution and an explicit patent grant. See [`LICENSE`](LICENSE) for
the full terms.
