# Confirmatory Run Readiness Audit — FEH H1

**Date:** 2026-05-31
**Scope:** Read-only audit of the FEH experiment code to assess readiness for the registered confirmatory data-collection run. No code was modified, nothing was launched, no API/Ollama calls were made. (This report is the only file written.)
**Target run (pre-reg v0.4):** 7 models × 79 items × 5 conditions (C1–C5) × 5 reps = **13,825 cells**. 5 local (Ollama): Phi-3.5-mini (3.8B), Mistral-7B-Instruct, Qwen2.5-7B, Qwen2.5-14B, Qwen2.5-32B; 2 frontier (API): Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`), GPT-4o (`gpt-4o-2024-11-20`). Pool: `feh79_item_pool_v0.4.yaml`. Est. 45–75h local compute.

**BOTTOM LINE: NOT READY — multiple launch blockers.** There is **no confirmatory data-collection runner** (only `confirmatory_analysis.py`, which is analysis). **No runner has resume/checkpoint logic** (fatal for a 45–75h job). Every runner/scorer reads the **wrong pool version (v0.3)**. The local runner covers **only 3 models — and the wrong 3** (includes non-registered `llama3.1:8b`; missing Phi-3.5, Qwen-14B, Qwen-32B). The runners are hardcoded to the **42-item calibration subset**, and **no code selects the registered 79-frame set** (the on-disk v0.4 pool actually holds **139** frames — see C).

---

## A. RUNNER COVERAGE

**No unified 7-model confirmatory runner exists, and no confirmatory runner of any kind exists.** The only file named "confirmatory" is `confirmatory_analysis.py`, which is the *analysis* pipeline (PyMC fit of eq. 6.1) — it never calls a model and expects to read `confirmatory_responses.json` (a file that does not yet exist). Glob of `Experiments\` for `confirmatory_run*` / `confirmatory_runner.py` → **not present**. The only `*run*.py` files are `calibration_runner.py` and `pilot_runner.py`.

Data collection is **split** across two scripts, each of which would have to be promoted/forked into the confirmatory runner:

### Local (Ollama) — `calibration_runner.py:64`
```python
MODELS = ["mistral:7b-instruct", "qwen2.5:7b", "llama3.1:8b"]
```
- Only **3** local models, and **the wrong set** (see D): includes `llama3.1:8b` (NOT registered) and is **missing `phi3.5:3.8b`, `qwen2.5:14b`, `qwen2.5:32b`**.
- Items are **hardcoded to the 42-item calibration subset**, not 79:
  - `calibration_runner.py:67-73` `HIGH_ITEMS` = 28 cbd-scorable Knightian items.
  - `calibration_runner.py:76-79` `LOW_ITEMS` = 14 easy controls (deliberately avoids R-001=47×83, which floors 7B models). **Total 42.**
- Conditions C1–C5 are iterated (`calibration_runner.py:168`). Reps via `--reps=` (**default 3**).
- Loop order (`:187-196`): model → frame → condition → rep (model-outer so each model loads VRAM once).
- **Confirmed against actual data:** `calibration_responses.json` = **1,890 cells = 3 models × 42 items × 5 conditions × 3 reps** (reps 1/2/3). So the calibration ran at **3 reps, not 5.**

### Frontier (API) — model IDs from `cross_model_check.py:51-52`, used by `frontier_read.py`
```python
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"   # cross_model_check.py:51
OPENAI_MODEL = "gpt-4o-2024-11-20"            # cross_model_check.py:52
```
`frontier_read.py:144-150`:
```python
MODELS = {
    "claude": ("claude", cmc.CLAUDE_MODEL, "anthropic", call_claude),
    "gpt4o":  ("gpt4",  cmc.OPENAI_MODEL,  "openai",   call_openai),
}
DEFAULT_MODELS = ["claude", "gpt4o"]
DEFAULT_CONDS  = ["C1", "C2", "C3", "C4"]   # C5 dropped (Amendment 2 §6)
DEFAULT_REPS   = 2
```
- Both frontier tags are **correct / registered.**
- Items again default to the 42-item subset (`frontier_read.py:162-169` `_frames()` → `HIGH_ITEMS`+`LOW_ITEMS` imported from `calibration_runner`).
- **Default conditions are C1–C4 (C5 dropped); default reps = 2.** Both differ from the registered 5-condition × 5-rep design and would have to be overridden.

**Net:** the registered design needs a new/forked runner that (a) uses the 5 correct local tags + 2 frontier tags, (b) iterates the registered 79-frame set, (c) runs 5 conditions × 5 reps, and (d) writes one unified `confirmatory_responses.json`.

---

## B. CHECKPOINTING / RESUME — **LAUNCH BLOCKER**

**No runner has idempotent resume. Neither detects already-completed cells; both initialize an empty list and overwrite the results file from scratch.** For a 45–75h run this is the single most dangerous gap. Grep of the runners for `resume|skip|already|done_keys|completed|checkpoint|existing` returns **nothing**.

`calibration_runner.py:182,194-203`:
```python
cells: list = []
...
        cells.append(run_cell(model_id, item, regime, condition, rep, log))
        if len(cells) % 10 == 0:   # crash-safe snapshot
            RESULTS_PATH.write_text(json.dumps(cells, indent=2), encoding="utf-8")
...
RESULTS_PATH.write_text(json.dumps(cells, indent=2), encoding="utf-8")
```
The "crash-safe snapshot" only protects the *current* process: it dumps the in-memory `cells` list. On **restart**, `cells` is re-initialized to `[]`, and the first snapshot **overwrites** `calibration_responses.json` with a near-empty list. There is no read of the existing file, no per-cell key check, no skip.

`frontier_read.py:282,293-300` is identical in spirit:
```python
cells: list = []
...
        cells.append(run_cell(mkey, item, regime, cond, rep, log))
        if mode == "run" and len(cells) % 10 == 0:
            RESULTS_PATH.write_text(json.dumps(cells, indent=2), encoding="utf-8")
...
RESULTS_PATH.write_text(json.dumps(cells, indent=2), encoding="utf-8")
```
Same overwrite-from-empty on restart. Both runners also open the log with `open(LOG_PATH, "w", ...)` (`calibration_runner.py:184`, `frontier_read.py:284`), truncating prior logs each run.

**Verdict: BLOCKER.** Any crash, OOM, Ollama hiccup, reboot, or API outage during a 45–75h run destroys all prior cells. Resume-by-cell-key (load the existing JSON, skip any (model, frame_id, condition, rep) already present, append new) must be added before launch. (Note: `cell_seed()` is deterministic per (model, frame, condition, rep), `pilot_runner.py:83-86`, so a clean resume key already exists — it just isn't used.)

---

## C. POOL VERSION — **BLOCKER (every reference is v0.3) + 79-vs-139 frame discrepancy**

**Every hardcoded pool reference in the codebase points to an OLD version. Not one `.py` file references v0.4.** The confirmatory run is registered against v0.4.

| file:line | reference |
|---|---|
| `pilot_runner.py:41` | `POOL_PATH = REPO_ROOT / "feh79_item_pool_v0.3.yaml"` |
| `calibration_analysis.py:44` | `POOL_PATH = REPO_ROOT / "feh79_item_pool_v0.3.yaml"` |
| `cross_model_check.py:46` | `POOL_PATH = REPO_ROOT / "feh79_item_pool_v0.3.yaml"` |
| `cross_model_check.py:9` (docstring) | `feh79_item_pool_v0.1.yaml` |
| `pilot_analysis.py:32` | `POOL_PATH = REPO_ROOT / "feh79_item_pool_v0.3.yaml"` |
| `k3_floor_check.py:48` | `POOL_PATH = REPO_ROOT / "feh79_item_pool_v0.2.yaml"` |
| `k3_floor_check.py:11` (docstring) | `feh79_item_pool_v0.2.yaml` |

Critical inheritance chain: `calibration_runner.py` and `frontier_read.py` both call `load_pool()` imported from `pilot_runner` (`calibration_runner.py:37-49`, `frontier_read.py:57-65`), whose `POOL_PATH` is **v0.3** (`pilot_runner.py:41`, used by `load_pool` at `:125-128`). **So both runners load v0.3 today.**

**The scorers:**
- `calibration_analysis.py` reads the pool **only for low-regime gold lookup** (`score_cells:182` `pool.get(fid,{}).get("gold_answer")`) and via its own `POOL_PATH` at `:44` (also **v0.3**). High-regime scoring is pool-free (regex on the answer, `:80-84,:179-180`).
- `confirmatory_analysis.py` (the registered scorer) does **not read the pool at all** — it ingests `confirmatory_responses.json` and derives everything from the records.

**79-vs-139 frame discrepancy (IMPORTANT).** I parsed the YAMLs (multi-document streams; must use `yaml.safe_load_all`, which the code does):
- `feh79_item_pool_v0.4.yaml` contains **139 frames**, not 79: K1=20, K2=18, K3=20, K4=19 (=77 Knightian), R=22 (reference), A=10 (aleatory), **CB=30 (new "calibration-baseline" category, added in v0.4)**. v0.3 had 109 frames (the same set minus the 30 CB).
- The registered "79-frame confirmatory pool" is **not** a file — it is defined in pre-reg §3.4 as a **regime-score quartile pre-screen**: top-quartile high-regime (~20 items) + bottom-quartile low-regime (~20) for the primary analysis, with the middle ~39 retained for robustness (≈79 total), the CB items being calibration baselines. `Calibration_protocol.md:17-20` states the 42-item calibration subset is "drawn from the 79-frame confirmatory pool (top + bottom regime quartiles)."
- **No code performs this 79-frame quartile selection.** The runners hardcode the 42-item calibration subset; nothing builds the registered 79-frame target from the 139-frame pool. This selection step is **missing** and must exist (or the 79 must be enumerated) before a "79-item" confirmatory run can be defined.

**Data-side good news:** v0.4 items have keys `frame_id, category, question, answer_format, answer_choices, gold_answer, grading_method, synonyms, metadata`. The gold field is **`gold_answer`** (matches what `calibration_analysis.py:182` reads) and the question field is **`question`** (matches `build_user_prompt` at `pilot_runner.py:89-95` and `cross_model_check.py:139`). All 28 HIGH + 14 LOW calibration frame_ids exist in v0.4 (0 missing). So pointing the runners at v0.4 is mechanically safe for the existing 42; the open issue is defining the registered 79.

---

## D. OLLAMA TAGS

Registered local models (pre-reg v0.4 §3.3, `Preregistration_full_empirical_v0.4.md:92-100`):
M1 Phi-3.5-mini-instruct (3.8B), M2 Mistral-7B-Instruct-v0.3, M3 Qwen-2.5-7B, M4 Qwen-2.5-14B (4-bit AWQ), M5 Qwen-2.5-32B (4-bit AWQ).

What `calibration_runner.py:64` actually uses:
```python
MODELS = ["mistral:7b-instruct", "qwen2.5:7b", "llama3.1:8b"]
```

| Registered model | Expected Ollama tag | In code? |
|---|---|---|
| Phi-3.5-mini (3.8B) | (e.g. `phi3.5:3.8b`) | **NO — not referenced in any script** |
| Mistral-7B-Instruct | `mistral:7b-instruct` | yes |
| Qwen2.5-7B | `qwen2.5:7b` | yes |
| Qwen2.5-14B | (e.g. `qwen2.5:14b`) | **NO — not referenced anywhere** |
| Qwen2.5-32B | (e.g. `qwen2.5:32b`) | **NO — not referenced anywhere** |
| (NOT registered) | `llama3.1:8b` | present — **must be removed** |

Grep across all `Experiments\*.py` for `phi`, `14b`, `32b`, `3.5`: **zero hits** for `phi3.5`, `qwen2.5:14b`, `qwen2.5:32b`. The other model-listing scripts (`pilot_runner.py:46`, and the local set echoed in calibration data) use the same Mistral/Qwen-7B/Llama-8B trio. The frontier IDs (`cross_model_check.py:51-52`) are correct.

**Verdict:** code covers only 2 of 5 registered local models; 3 registered models (Phi-3.5, Qwen-14B, Qwen-32B) are absent and the non-registered Llama-3.1-8B must be dropped. The exact Ollama tags for the three missing models must be chosen and confirmed pulled in the local install (not verifiable in this read-only audit; do not run ollama).

---

## E. OUTPUT SCHEMA

### `calibration_runner.py` cell record (`:130-148`) — confirmed identical to actual JSON keys
```python
cell = {
    "model": model_id, "frame_id": item["frame_id"], "category": item["category"],
    "regime": regime,                       # "high" | "low"
    "condition": condition, "replication": rep, "seed": seed, "prompt": user,
    "raw_response": raw, "extracted_final_answer": extracted,
    "n_steps_heuristic": n_steps, "n_sentences": n_sentences,
    "step_count_method": step_method,
    "tokens_in": prompt_eval_count, "tokens_out": eval_count,
    "latency_s": round(latency, 2), "error": err,
}
```
(Actual `calibration_responses.json` record keys verified: exactly these 17 fields.)

### `frontier_read.py` record (`:193-211`)
Same 17 fields **plus** `regime`, but note: `model` = the full pinned model id (e.g. `claude-sonnet-4-5-20250929`). It does **not** carry a separate `model_tag`. Schema is otherwise field-identical to the local record (both include `category`, `seed`, `prompt`, `n_sentences`, `step_count_method`, `regime`). So the two runner schemas **are aligned** — good for a merged file.

### What the scorers require
- **`calibration_analysis.py` `score_cells` (`:172-192`)** reads: `error`, `raw_response`, `frame_id`, `extracted_final_answer`, **`regime`** (string "high"/"low"), `n_steps_heuristic`, `model`, `condition`. All present in both runner schemas → **compatible**. (Note it keys high vs low off the **`regime` field**, unlike the confirmatory scorer.)
- **`confirmatory_analysis.py` (registered scorer)** — its `load_pilot()` ingester (`:381-407`) reads `frame_id`, `extracted_final_answer`, `n_steps_heuristic`, `model`; it derives `regime` by **`fid.startswith(("K","A"))`** (a *crude pilot proxy*, explicitly labeled so at `:403`) and `y` via `pilot_analysis.GOLD_ANSWERS`. The PyMC `run()`/`prepare()` path requires tidy columns `{y, steps, regime, model, item}` (`:89-92`). There is **no confirmatory ingester** that scores the full `confirmatory_responses.json` into that frame.

**Mismatch flags:**
1. **No confirmatory scoring/ingestion function exists.** `confirmatory_analysis.py` only has `load_pilot()` (sparse pilot proxy) and `__main__` runs that pilot smoke test. A real ingester — score every cell (cbd for high, gold-match for low), assign regime from the registered quartile bins (not a `startswith` proxy), emit `{y, steps, regime, model, item}` — is **missing**. The richer, correct scoring logic lives in `calibration_analysis.py` (`is_cbd`, `gold_match`, `delatex`, sig-fig matching) and would need to be reused.
2. **`confirmatory_analysis.py` expects `confirmatory_responses.json`** but no runner writes that filename (`calibration_responses.json` / `frontier_read_responses.json`). Must reconcile.
3. **Regime assignment differs by path.** Calibration uses an explicit `regime` field; the confirmatory pilot proxy uses `frame_id` prefix. For the real run, regime must come from the registered quartile pre-screen, recorded per cell — neither path does this yet.

No field-name mismatch will crash an ingest of the existing schema, but items 1–3 mean the analysis pipeline cannot currently consume a confirmatory run end-to-end.

---

## F. CONDITIONS C1–C5

All five templates exist in `pilot_runner.py:69-75` (`CONDITION_TEMPLATES`), imported by both runners. Verbatim (the `{q}` is the item question):

- **C1** (`:70`): "Give a single-line answer of the form 'Final answer: <X>' with no other text." — shortest, no CoT.
- **C2** (`:71`): "Think step by step, briefly, in 3 steps or fewer. Then give a single-line final answer…"
- **C3** (`:72`): "Think step by step in about 7 steps. Then give a single-line final answer…"
- **C4** (`:73`): "Reason through this carefully, considering multiple angles, in approximately 15 steps. Then give a single-line final answer…"
- **C5** (`:74`): "Think step by step. When you've reached a conclusion, finish with a single-line answer…" — **unconstrained** (no target step count).

**Known C5 behavioral issue (CONFIRMED in calibration data; `c5_diag_out.txt`).** C5 is "unconstrained," but on the 7–8B calibration models it produces **far shorter** completions and **fewer** steps than C4:

| cond | mean tokens_out | mean steps |
|---|---|---|
| C3 | 354.9 | 10.79 |
| C4 | 572.3 | **17.89** |
| C5 | 254.4 | **6.62** |

So C5 < C4 < C3 in length is **violated** — C5's length sits in the C2–C3 band. The diagnostic verdict: **BEHAVIORAL** (not a step-counter bug — C5 steps/sentence 0.927 ≈ C4 0.780, so no under-segmentation); the unconstrained prompt simply yields shorter completions on small models. Consequences:
- This is **why `frontier_read.py:149` drops C5** from the frontier dose read ("C5 dropped, see Amendment 2 §6").
- The registered primary test is the **assigned-length condition-ITT** (long = C4/C5 vs short = C1/C2), which does not require C5 > C4 monotonicity, so the primary contrast is robust. But any **dose-response / monotonic-length** secondary analysis that assumes C5 is the longest is mis-specified. Flag for the analysis plan; whether C5 is even retained in the confirmatory run should be an explicit, registered decision (the prereg target still lists 5 conditions).

---

## G. COST / TIME INPUTS (`frontier_read.py`)

Raw constants:
- `MAX_TOKENS = 2048` (`frontier_read.py:80`) — max output tokens per API call (Claude & GPT-4o). Comment: "matches calibration's Ollama num_predict." (Local `NUM_PREDICT = 2048`, `pilot_runner.py:49`.)
- Per-1M-token list prices, USD (`frontier_read.py:152-156`):
  - Claude (`claude`): **input $3.00, output $15.00**
  - GPT-4o (`gpt4`): **input $2.50, output $10.00**
- Per-condition output-token estimate used only for the cost projection (`frontier_read.py:159`): `EST_OUT = {"C1":80, "C2":250, "C3":500, "C4":750, "C5":500}` (capped at MAX_TOKENS).
- `REQUEST_TIMEOUT = 90` s for frontier calls (`cross_model_check.py:64`, used by `_post_with_retry`). (`pilot_runner.REQUEST_TIMEOUT = 600` is the local-Ollama timeout, `:50`.)
- **Throttle/spacing/retry (these DO exist for frontier, via `cross_model_check`):**
  - Per-provider min interval (`cross_model_check.py:72-77`): `claude 0.0`, `gpt4 0.0`, `gemini 4.2`, `mistral 2.1` s. So **Claude/GPT-4o have NO inter-call spacing** (`_provider_throttle` is a no-op for them, `cross_model_check.py:82-94`; comment at `frontier_read.py:177`).
  - Retry/backoff (`cross_model_check.py:65-66,164-189`): `RETRY_STATUSES = {429,500,502,503,504}`, `MAX_RETRIES = 6`, exponential backoff `2**attempt + uniform(0,0.5)` (~63s worst case). This **is** wired into the frontier calls via `cmc._post_with_retry`.
  - `frontier_read.py` makes calls **serially** in a single loop (no concurrency); local Ollama calls in `calibration_runner.py` have **no timeout-spacing and no retry** (one failed cell is recorded as `error` and skipped, `:112-116`).
- Local generation knobs: `TEMPERATURE = 0.7` (`pilot_runner.py:47`), `TOP_P = 0.95` (`:48`), `NUM_PREDICT = 2048` (`:49`).
- `frontier_read.py` defaults to a **DRY RUN** (prints plan + cost estimate, no paid calls); `--smoke` = 1 cell/model; `--run` = full (`:30-32, 308-314`). `PRICES` is used by `estimate_cost()` (`:224-244`) to print a projected total but does not gate anything.

---

## H. GAPS LIST (prioritized)

**BLOCKERS (must fix before launch):**
1. **No confirmatory data-collection runner exists.** Only `confirmatory_analysis.py` (analysis). A runner producing `confirmatory_responses.json` must be written/forked from `calibration_runner.py` + `frontier_read.py`. — *no file*
2. **No resume / idempotent restart.** Both runners start from `[]` and overwrite the output on restart; a crash mid-run loses everything in a 45–75h job. Add per-(model,frame,condition,rep) skip against the existing JSON. — *`calibration_runner.py:182,197-200,203`; `frontier_read.py:282,294-300`*
3. **Wrong pool version (v0.3).** Every pool reference is v0.1/v0.2/v0.3; confirmatory must use v0.4. Both runners load v0.3 via `pilot_runner.py:41`. — *`pilot_runner.py:41`, `calibration_analysis.py:44`, `cross_model_check.py:46`, `pilot_analysis.py:32`, `k3_floor_check.py:48`*
4. **No registered 79-frame selection.** The on-disk v0.4 pool has **139** frames; the registered 79 is a quartile pre-screen (pre-reg §3.4) that **no code computes**. The runners hardcode the 42-item calibration subset. The 79-frame target must be defined/enumerated. — *`calibration_runner.py:67-79`; `frontier_read.py:162-169`; pre-reg §3.4*
5. **Local model set wrong/incomplete.** Missing Phi-3.5, Qwen-14B, Qwen-32B; includes non-registered `llama3.1:8b`. — *`calibration_runner.py:64`*
6. **Conditions/reps not at registered 5×5.** `calibration_runner.py` default reps=3 (`:212`); `frontier_read.py` defaults to C1–C4 (C5 dropped) and reps=2 (`:149-150`). The registered run is 5 conditions × 5 reps; these must be set explicitly. — *`calibration_runner.py:212`; `frontier_read.py:149-150`*
7. **No confirmatory scorer/ingester.** `confirmatory_analysis.py` only has `load_pilot()` (sparse pilot proxy with a `frame_id`-prefix regime guess); nothing scores a full `confirmatory_responses.json` into `{y, steps, regime, model, item}` using the real cbd/gold logic + registered regime bins. — *`confirmatory_analysis.py:381-407`*

**IMPORTANT (correctness / integration):**
8. **Output filename mismatch.** `confirmatory_analysis.py` would consume `confirmatory_responses.json`; no runner writes it (`calibration_responses.json` / `frontier_read_responses.json`). — *`confirmatory_analysis.py:381-391`*
9. **Regime must be recorded from the registered quartile bins per cell**, not the `K/A`-prefix proxy used in the pilot path. — *`confirmatory_analysis.py:403`*
10. **No retry/spacing for local Ollama**, and **no inter-call spacing for Claude/GPT-4o** (both throttle intervals are 0.0). Frontier *does* have 6× exponential backoff on 429/5xx via `_post_with_retry`; local has none. For ~3,950 frontier calls (2×79×5×5) and ~9,875 local calls (5×79×5×5), confirm rate limits/robustness. — *`cross_model_check.py:72-77,164-189`; `calibration_runner.py:96,112-116`*
11. **C5 length saturation/inversion** (C5 254 tok / 6.6 steps vs C4 572 tok / 17.9 steps). Primary ITT (long=C4/C5 vs short=C1/C2) is robust, but decide explicitly whether C5 is retained, and flag any dose-response/monotonic secondary analysis as mis-specified. — *`c5_diag_out.txt`; `pilot_runner.py:74`; `frontier_read.py:149`*

**MINOR (note, not blocking):**
12. **Stale docstring:** `calibration_runner.py:4-5` says "3 reps = 1,890 cells" — that matches the actual data (1,890 cells, 3 reps), so the docstring is correct; the MEMORY note of "reps=5/6,300 cells" is what is stale (the on-disk calibration is 3 reps). Reconcile expectations before forking.
13. **`cross_model_check.py` docstring** still references `feh79_item_pool_v0.1.yaml` (`:9`) while its code uses v0.3 (`:46`) — cosmetic drift.
14. **`frontier_read.py` and `calibration_runner.py` truncate their logs** (`open(..., "w")`) each run; combined with no resume, prior-run logs are lost. — *`frontier_read.py:284`; `calibration_runner.py:184`*

**On the "double-.py" question:** `frontier_read.py.py` and `pilot_runner.py.py` **do not exist** — there are no `*.py.py` files anywhere in the repo. Only `frontier_read.py` and `pilot_runner.py` are present. They are neither real runners-in-disguise nor junk duplicates; the double-extension files simply are not there.
