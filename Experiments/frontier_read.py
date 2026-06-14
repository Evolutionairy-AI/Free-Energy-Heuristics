"""
frontier_read.py — cheap FRONTIER feasibility read for FEH H1.

Purpose. The local calibration panel (3 × 7-8 B Ollama models) showed the H1
dissociation descriptively but several high-regime items FLOORED (the small
models cannot recognise "cannot-be-determined" even once). The open question
that gates the paper's central claim — a counter-narrative to o1/o3
reasoning-scaling — is whether H1 is bigger/cleaner on capable frontier models
that are OFF the floor, or whether it is an artefact of small-model
incompetence. This script runs the *identical* calibration manipulation on
1-2 frontier API models to answer that, cheaply, before committing to a full
confirmatory design.

This is FEASIBILITY / exploratory, NOT the confirmatory test (eq. 6.1'). Model
snapshots are pinned for reproducibility. The output schema matches
calibration_responses.json so calibration_analysis.py scoring + the
condition-ITT fit apply unchanged.

Comparability. SYSTEM_PROMPT, the C1-C5 condition templates, temperature (0.7),
top_p (0.95) and the item sets are imported from the SAME modules the local
calibration used (pilot_runner, calibration_runner). The only differences are
the model endpoints and max_tokens (2048 here vs Ollama num_predict 2048 — the
same).

Safety. Defaults to a DRY RUN: it prints the plan + a token/cost estimate and
builds the prompts WITHOUT making any paid API call. Use --smoke for a 1-cell
wiring check (a few cents) and --run for the full read.

Usage:
  python frontier_read.py                 # dry run: plan + cost estimate only
  python frontier_read.py --smoke         # 1 real cell per model (wiring check)
  python frontier_read.py --run           # full read -> frontier_read_responses.json
  --models=claude,gpt4o                    # default: claude,gpt4o
  --conditions=1,2,3,4                      # default: 1,2,3,4 (C5 dropped, see Amendment 2 §6)
  --reps=2                                  # default: 2
  --items=K1-001,...                        # default: 28 cbd Knightian + 14 controls
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

# Proven API plumbing: truststore.inject_into_ssl() runs at import, plus
# exponential-backoff retry, key loading from API_KEYS/, and per-provider
# throttling. We reuse these and pinned model IDs; we do NOT use its C1-only
# prompt (we need the full C1-C5 manipulation below).
import cross_model_check as cmc  # noqa: E402

# Identical manipulation + scoring path as the local calibration panel.
from pilot_runner import (  # noqa: E402
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
    build_user_prompt,
    extract_final_answer,
    cell_seed,
    load_pool,
)
from calibration_runner import HIGH_ITEMS, LOW_ITEMS  # noqa: E402
from step_counter import step_count  # noqa: E402

# Windows console / redirected-file default is cp1252; force UTF-8 so a stray
# glyph in a printed answer never kills the run (mirrors calibration_runner).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

RESULTS_PATH = HERE / "frontier_read_responses.json"
LOG_PATH = HERE / "frontier_read.log"

MAX_TOKENS = 2048  # matches calibration's Ollama num_predict

# ---------------------------------------------------------------------------
# Provider call adapters (return text + token usage; reuse cmc._post_with_retry).
# Bodies mirror cross_model_check but inject the calibration SYSTEM_PROMPT and
# max_tokens, and parse the usage block so we capture tokens_out for the
# behavioural dose check (the C5 token-vs-step analysis).
# ---------------------------------------------------------------------------


def call_claude(key: str, user: str, seed: int | None = None):
    # Anthropic has no seed param; reps rely on temperature stochasticity.
    body = {
        "model": cmc.CLAUDE_MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user}],
    }
    r = cmc._post_with_retry(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json_body=body,
    )
    j = r.json()
    text = "".join(
        b.get("text", "") for b in j.get("content", []) if b.get("type") == "text"
    )
    usage = j.get("usage") or {}
    return text, usage.get("output_tokens"), usage.get("input_tokens")


def call_openai(key: str, user: str, seed: int | None = None):
    body = {
        "model": cmc.OPENAI_MODEL,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
    }
    if seed is not None:
        body["seed"] = seed
    r = cmc._post_with_retry(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json_body=body,
    )
    j = r.json()
    text = j["choices"][0]["message"]["content"]
    usage = j.get("usage") or {}
    return text, usage.get("completion_tokens"), usage.get("prompt_tokens")


# short-key -> (throttle-provider, pinned model id, API_KEYS provider, call fn)
MODELS = {
    "claude": ("claude", cmc.CLAUDE_MODEL, "anthropic", call_claude),
    "gpt4o": ("gpt4", cmc.OPENAI_MODEL, "openai", call_openai),
}
DEFAULT_MODELS = ["claude", "gpt4o"]
DEFAULT_CONDS = ["C1", "C2", "C3", "C4"]  # C5 dropped from the dose read (Amendment 2 §6)
DEFAULT_REPS = 2

# Rough list-price USD per 1M tokens (May 2026; correct these if stale).
PRICES = {  # provider -> (input_per_M, output_per_M)
    "claude": (3.00, 15.00),
    "gpt4": (2.50, 10.00),
}
# Generous per-condition output-token estimate for costing (frontier models
# write more than the 7-8 B panel; capped at MAX_TOKENS).
EST_OUT = {"C1": 80, "C2": 250, "C3": 500, "C4": 750, "C5": 500}


def _frames(items):
    """[(frame_id, regime)] for the requested items (default: calib 28+14)."""
    if items:
        hi = [f for f in items if f in HIGH_ITEMS]
        lo = [f for f in items if f in LOW_ITEMS]
    else:
        hi, lo = HIGH_ITEMS, LOW_ITEMS
    return [(f, "high") for f in hi] + [(f, "low") for f in lo]


def run_cell(mkey, item, regime, condition, rep, log) -> dict:
    provider, model_id, keyname, call = MODELS[mkey]
    keys = run_cell._keys
    seed = cell_seed(model_id, item["frame_id"], condition, rep)
    user = build_user_prompt(item, condition)
    cmc._provider_throttle(provider)  # no-op for claude/gpt4; spaces gemini/mistral
    t0 = time.time()
    try:
        raw, out_tok, in_tok = call(keys[keyname], user, seed)
        err = None
    except Exception as e:  # noqa: BLE001  (record + continue; one bad cell != lost run)
        raw, out_tok, in_tok = "", None, None
        err = f"{type(e).__name__}: {e}"
    latency = time.time() - t0
    extracted = extract_final_answer(raw) if raw else ""
    if raw:
        sc = step_count(raw)
        n_steps, n_sentences, step_method = sc.n_steps, len(sc.sentences), sc.method
    else:
        n_steps, n_sentences, step_method = 0, 0, "n/a"

    cell = {
        "model": model_id,
        "frame_id": item["frame_id"],
        "category": item["category"],
        "regime": regime,
        "condition": condition,
        "replication": rep,
        "seed": seed,
        "prompt": user,
        "raw_response": raw,
        "extracted_final_answer": extracted,
        "n_steps_heuristic": n_steps,
        "n_sentences": n_sentences,
        "step_count_method": step_method,
        "tokens_in": in_tok,
        "tokens_out": out_tok,
        "latency_s": round(latency, 2),
        "error": err,
    }
    ot = str(out_tok) if out_tok is not None else "-"
    msg = (
        f"  {model_id:24s} {item['frame_id']:8s} {condition} r{rep}  "
        f"steps={n_steps:3d} out_tok={ot:>4s} ans={extracted[:32]:32s} "
        f"({latency:5.1f}s){' ERR='+err if err else ''}"
    )
    print(msg)
    log.write(msg + "\n")
    log.flush()
    return cell


def estimate_cost(model_list, frames, cond_list, n_reps, pool) -> None:
    print("\n--- cost estimate (rough; list prices, generous output) ---")
    sys_tok = len(SYSTEM_PROMPT) / 4
    grand = 0.0
    for mkey in model_list:
        provider, model_id, _, _ = MODELS[mkey]
        pin, pout = PRICES.get(provider, (0.0, 0.0))
        in_tok = out_tok = 0
        for frame_id, _regime in frames:
            item = pool[frame_id]
            for cond in cond_list:
                prompt = build_user_prompt(item, cond)
                in_tok += (len(prompt) / 4 + sys_tok) * n_reps
                out_tok += min(EST_OUT.get(cond, 400), MAX_TOKENS) * n_reps
        cost = in_tok / 1e6 * pin + out_tok / 1e6 * pout
        grand += cost
        print(
            f"  {model_id:24s}  ~{in_tok/1000:6.0f}k in + ~{out_tok/1000:6.0f}k out"
            f"  -> ~${cost:5.2f}"
        )
    print(f"  {'TOTAL':24s}  {'':>22s}  -> ~${grand:5.2f}\n")


def main(mode: str, model_list, cond_list, n_reps, items) -> None:
    pool = load_pool()
    frames = _frames(items)
    missing = [f for f, _ in frames if f not in pool]
    if missing:
        raise SystemExit(f"missing from pool: {missing}")

    n_cells = len(model_list) * len(frames) * len(cond_list) * n_reps
    print(
        f"[plan] {len(model_list)} models {model_list} × {len(frames)} items "
        f"({sum(r=='high' for _,r in frames)} high + {sum(r=='low' for _,r in frames)} low) "
        f"× {len(cond_list)} conds {cond_list} × {n_reps} reps = {n_cells} cells"
    )

    # Pre-flight: confirm the needed keys load (never printed).
    keys = cmc.load_keys()
    need = {MODELS[m][2] for m in model_list}
    have = {k: bool(keys.get(k)) for k in need}
    print(f"[keys] present: " + ", ".join(f"{k}={'yes' if v else 'NO'}" for k, v in have.items()))
    if not all(have.values()):
        raise SystemExit(f"missing API keys for: {[k for k,v in have.items() if not v]}")
    run_cell._keys = keys

    estimate_cost(model_list, frames, cond_list, n_reps, pool)

    if mode == "dry":
        print("[dry-run] no API calls made. Re-run with --smoke (wiring) or --run (full).")
        return

    if mode == "smoke":
        frames = frames[:1]          # 1 high item
        cond_list = cond_list[:1]    # C1
        n_reps = 1
        print(f"[smoke] {len(model_list)} real cells (1 item × C1 × 1 rep per model)")

    cells: list = []
    t_start = time.time()
    with open(LOG_PATH, "w", encoding="utf-8") as log:
        log.write(f"[start] mode={mode} models={model_list} conds={cond_list} reps={n_reps}\n")
        for mkey in model_list:
            log.write(f"[model] {MODELS[mkey][1]}\n")
            log.flush()
            for frame_id, regime in frames:
                item = pool[frame_id]
                for cond in cond_list:
                    for rep in range(1, n_reps + 1):
                        cells.append(run_cell(mkey, item, regime, cond, rep, log))
                        if mode == "run" and len(cells) % 10 == 0:
                            RESULTS_PATH.write_text(json.dumps(cells, indent=2), encoding="utf-8")
        log.write(f"[done] {len(cells)} cells in {time.time()-t_start:.0f}s\n")

    n_err = sum(1 for c in cells if c["error"])
    if mode == "run":
        RESULTS_PATH.write_text(json.dumps(cells, indent=2), encoding="utf-8")
        print(f"[done] wrote {RESULTS_PATH.name}; {len(cells)} cells ({n_err} errors) "
              f"in {time.time()-t_start:.0f}s")
    else:  # smoke
        print(f"[smoke done] {len(cells)} cells ({n_err} errors); not written. "
              f"Inspect the log if any ERR above.")


if __name__ == "__main__":
    argv = sys.argv[1:]
    mode = "dry"
    if "--run" in argv:
        mode = "run"
    elif "--smoke" in argv:
        mode = "smoke"
    model_list = DEFAULT_MODELS
    cond_list = DEFAULT_CONDS
    n_reps = DEFAULT_REPS
    items = None
    for a in argv:
        if a.startswith("--models="):
            model_list = [s.strip() for s in a.split("=", 1)[1].split(",") if s.strip()]
            bad = [m for m in model_list if m not in MODELS]
            if bad:
                raise SystemExit(f"unknown models {bad}; known: {list(MODELS)}")
        elif a.startswith("--conditions="):
            cond_list = [s if s.startswith("C") else f"C{s}"
                         for s in (x.strip() for x in a.split("=", 1)[1].split(",")) if s]
        elif a.startswith("--reps="):
            n_reps = int(a.split("=", 1)[1])
        elif a.startswith("--items="):
            items = [s.strip() for s in a.split("=", 1)[1].split(",") if s.strip()]
    main(mode, model_list, cond_list, n_reps, items)
