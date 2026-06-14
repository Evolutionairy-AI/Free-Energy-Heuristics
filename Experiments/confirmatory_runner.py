"""
confirmatory_runner.py — REGISTERED confirmatory data collection for FEH H1.

This is the real run (pre-registration v0.4, eq. 6.1' assigned-length ITT). It is
the auto-scorable subset agreed with the PI (2026-05-31):

  PANEL  : 7 models — 5 local (Ollama) + 2 frontier (API).
  ITEMS  : 31 cbd-scorable Knightian items (high regime) + control items (low
           regime), DERIVED FROM THE v0.4 POOL by grading_method (no hardcoded
           transcription — stays in sync with the pool).
  DESIGN : 5 conditions C1-C5 × 5 replications.
  OUTCOME: high regime → cbd-correctness; low regime → gold-match (scored later
           by confirmatory_analyze.py, reusing the hardened calibration scorer).

The 46 expert-coherence Knightian items (grading_method = coherence_rubric)
require a human 3-rater panel that does not exist; they are OUT of this run by
design and deferred to a future secondary study.

WHY THIS FILE EXISTS (vs calibration_runner.py): the calibration/frontier
scripts are pre-data feasibility tooling — 3 wrong models, 42-item subset, pool
v0.3, NO resume. A 45-75h registered run that is interrupted by sleep/power/crash
MUST resume without re-spending or corrupting prior work. This runner:
  * keys every cell on (model, frame_id, condition, rep) and SKIPS cells already
    completed successfully in the output file (idempotent resume);
  * NEVER discards prior results — it loads, extends, and rewrites the union;
  * reads pool v0.4;
  * targets the 7 registered models;
  * is SAFE BY DEFAULT — bare run = dry plan + frontier cost (no calls);
    --smoke = a few real cells; --run = full collection.

Reuses pure helpers from pilot_runner (prompts/extraction/seed) and the frontier
API plumbing from frontier_read (lazy-imported only when a frontier model runs).

Output: confirmatory_responses.json (+ confirmatory_run.log)
CLI:
  python confirmatory_runner.py                      # DRY: plan + cost, no calls
  python confirmatory_runner.py --smoke              # 1 high+1 low cell/model, C1, rep1
  python confirmatory_runner.py --run                # full registered run (resumes)
  python confirmatory_runner.py --run --models=qwen2.5:32b   # one model (e.g. 32B time-test)
  python confirmatory_runner.py --run --local        # only the 5 local models
  python confirmatory_runner.py --run --frontier     # only the 2 frontier models
  --conditions=C1,C2  --reps=5  --high-only  --low-only   # further subsetting
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

from pilot_runner import (  # noqa: E402  (pure helpers; v0.3 POOL_PATH is NOT used here)
    NUM_PREDICT,
    OLLAMA_URL,
    REQUEST_TIMEOUT,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
    build_user_prompt,
    cell_seed,
    extract_final_answer,
)
from step_counter import step_count  # noqa: E402

# Windows console / redirected file defaults to cp1252 and crashes on non-cp1252
# glyphs in model answers — force UTF-8 so a stray glyph never kills the run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

POOL_PATH = REPO_ROOT / "feh79_item_pool_v0.4.yaml"   # REGISTERED pool (NOT v0.3)
RESULTS_PATH = HERE / "confirmatory_responses.json"
LOG_PATH = HERE / "confirmatory_run.log"

# ---- Registered panel (7 models) ------------------------------------------- #
# Local Ollama tags for the 5 registered local models (3B -> 32B size ladder).
LOCAL_MODELS = [
    "phi3.5",            # Phi-3.5-mini (3.8B)
    "mistral:7b-instruct",
    "qwen2.5:7b",
    "qwen2.5:14b",
    "qwen2.5:32b",
]
# Frontier API models (snapshot-pinned in cross_model_check).
FRONTIER_MODELS = ["claude", "gpt4o"]   # resolved to pinned snapshots at call time

# Pinned frontier snapshots — the canonical model id RECORDED and RESUME-KEYED on
# (must match cross_model_check.CLAUDE_MODEL / OPENAI_MODEL; asserted at call time
# in _frontier_setup). Hardcoded so dry-run/resume naming needs no API import.
FRONTIER_SNAPSHOT = {
    "claude": "claude-sonnet-4-5-20250929",
    "gpt4o": "gpt-4o-2024-11-20",
}


def canonical_model_id(model_id: str) -> str:
    """Identity for local Ollama tags; pinned snapshot id for frontier short keys.
    Used for BOTH the recorded 'model' field and the resume key, so a restart
    recognizes completed frontier cells (otherwise it would re-spend every time)."""
    return FRONTIER_SNAPSHOT.get(model_id, model_id)

# ---- Item selection (derived from the v0.4 pool by grading_method) ---------- #
# cbd-scorable Knightian items = K-family graded by the auto string-matcher (the
# v0.4 encoding for items whose normatively-correct answer is a fixed string,
# incl. 'cannot be determined'). VERIFIED against the pool inventory: of the 77 K
# items, exactly K1×20 + K2×9 + K4×2 = 31 carry grading_method=auto-string-match;
# the other 46 (all K3 + the rest of K2/K4) are expert-coherence (human panel),
# excluded by design. The is_k_item gate keeps the 4 auto-string-match R controls
# (R-003/004/005/007) out of the high-regime set.
CBD_GRADING = {"auto-string-match"}
# Low-regime controls: the calibration's proven easy set (determinate answers
# small models can actually produce — avoids R-001=47x83 flooring the 3.8B model,
# which would inject a competence confound into the low regime rather than a clean
# "reasoning doesn't hurt" baseline). Matches the 12-14 controls the power
# analysis assumed.
LOW_ITEMS_DEFAULT = [
    "R-101", "R-102", "R-103", "R-104", "R-105", "R-106",
    "A-001", "A-002", "A-003", "A-004", "A-005", "A-006", "A-007", "A-008",
]

CONDITIONS_ALL = ["C1", "C2", "C3", "C4", "C5"]
N_REPS_DEFAULT = 5

# Frontier cost constants (mirror frontier_read).
MAX_TOKENS = 2048
PRICES = {"claude": (3.0, 15.0), "gpt4o": (2.5, 10.0)}   # USD / 1M (in, out)
EST_OUT = {"C1": 80, "C2": 250, "C3": 500, "C4": 750, "C5": 500}


def load_pool() -> dict:
    docs = list(yaml.safe_load_all(POOL_PATH.read_text(encoding="utf-8")))
    return {d["frame_id"]: d for d in docs if d and "frame_id" in d}


def is_k_item(fid: str) -> bool:
    return fid.startswith(("K1", "K2", "K3", "K4"))


def select_items(pool: dict, low_items=None) -> list:
    """Return [(frame_id, regime)], regime in {'high','low'}, derived from pool.

    HIGH = cbd-scorable Knightian items (grading_method in CBD_GRADING).
    LOW  = the configured control list (default LOW_ITEMS_DEFAULT).
    Raises if any configured low item is absent from the pool.
    """
    high = sorted(
        fid for fid, d in pool.items()
        if is_k_item(fid) and d.get("grading_method") in CBD_GRADING
    )
    low = list(low_items if low_items is not None else LOW_ITEMS_DEFAULT)
    missing = [f for f in low if f not in pool]
    if missing:
        raise SystemExit(f"low-regime control items missing from pool: {missing}")
    return [(f, "high") for f in high] + [(f, "low") for f in low]


def is_frontier(model_id: str) -> bool:
    return model_id in ("claude", "gpt4o")


# --------------------------------------------------------------------------- #
# Model calls
# --------------------------------------------------------------------------- #
def call_ollama(model_id: str, prompt: str, seed: int) -> tuple:
    """(raw_text, out_tok, in_tok) from one deterministic Ollama generate call."""
    body = {
        "model": model_id,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "num_predict": NUM_PREDICT,
            "seed": seed,
        },
    }
    r = requests.post(OLLAMA_URL, json=body, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    return j.get("response", ""), j.get("eval_count"), j.get("prompt_eval_count")


_FR = {"mod": None, "keys": None}   # lazy frontier_read module + API keys cache


def _frontier_setup():
    """Lazy import frontier_read (truststore inject side-effect) + load API keys
    once. Only happens when a frontier model is actually run."""
    if _FR["mod"] is None:
        import frontier_read as fr  # noqa: E402
        import cross_model_check as cmc  # noqa: E402
        _FR["mod"] = fr
        _FR["cmc"] = cmc
        _FR["keys"] = cmc.load_keys()
    return _FR["mod"], _FR["cmc"], _FR["keys"]


# short key -> (frontier_read MODELS entry). Resolved lazily.
_FR_SPEC = {
    "claude": ("claude", "anthropic", "call_claude"),
    "gpt4o": ("gpt4", "openai", "call_openai"),
}


def call_frontier(model_id: str, user: str, seed: int) -> tuple:
    """(raw_text, out_tok, in_tok) via the frontier API plumbing. Mirrors
    frontier_read.run_cell exactly: per-provider throttle, then call(key, user,
    seed). Keys are loaded once and cached. Asserts the hardcoded snapshot ids
    still match cross_model_check (catches snapshot drift before any spend)."""
    fr, cmc, keys = _frontier_setup()
    expect = cmc.CLAUDE_MODEL if model_id == "claude" else cmc.OPENAI_MODEL
    if FRONTIER_SNAPSHOT[model_id] != expect:
        raise RuntimeError(
            f"snapshot drift for {model_id}: hardcoded {FRONTIER_SNAPSHOT[model_id]!r} "
            f"!= cross_model_check {expect!r}; update FRONTIER_SNAPSHOT")
    provider, keyname, fn_name = _FR_SPEC[model_id]
    cmc._provider_throttle(provider)   # no-op for claude/gpt4; spaces gemini/mistral
    fn = getattr(fr, fn_name)
    return fn(keys[keyname], user, seed)


# --------------------------------------------------------------------------- #
# Resume / checkpoint
# --------------------------------------------------------------------------- #
def cell_key(model_id: str, frame_id: str, condition: str, rep: int) -> str:
    return f"{model_id}|{frame_id}|{condition}|{rep}"


def load_existing() -> tuple:
    """Return (registry, done) where registry = {cell_key: cell} (deduped, last
    wins) and done = keys that completed successfully (no error & raw text).

    Keying on cell_key makes the output idempotent: a retried cell (smoke re-run
    or error retry) REPLACES its prior record instead of appending a duplicate,
    so the scorer never sees two records for one (model, item, condition, rep).
    Also collapses any pre-existing duplicates on load.
    """
    if not RESULTS_PATH.exists():
        return {}, set()
    try:
        cells = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001  (corrupt/partial file -> back up + start clean)
        bak = RESULTS_PATH.with_suffix(".json.corrupt")
        RESULTS_PATH.replace(bak)
        print(f"[resume] WARNING existing output unreadable ({e}); moved to "
              f"{bak.name}, starting fresh")
        return {}, set()
    registry = {}
    for c in cells:
        registry[cell_key(c["model"], c["frame_id"], c["condition"],
                          c["replication"])] = c
    done = {k for k, c in registry.items()
            if not c.get("error") and c.get("raw_response")}
    return registry, done


# --------------------------------------------------------------------------- #
# One cell
# --------------------------------------------------------------------------- #
def run_cell(model_id: str, item: dict, regime: str, condition: str, rep: int,
             log) -> dict:
    frontier = is_frontier(model_id)
    # Record + seed on the canonical id (snapshot for frontier) so the seed matches
    # frontier_read's and resume keying is consistent with load_existing().
    recorded_model = canonical_model_id(model_id)
    seed = cell_seed(recorded_model, item["frame_id"], condition, rep)
    user = build_user_prompt(item, condition)
    t0 = time.time()
    try:
        if frontier:
            raw, out_tok, in_tok = call_frontier(model_id, user, seed)
        else:
            raw, out_tok, in_tok = call_ollama(model_id, user, seed)
        err = None
    except Exception as e:  # noqa: BLE001  (record + continue; one bad cell != lost run)
        raw, out_tok, in_tok, err = "", None, None, f"{type(e).__name__}: {e}"
    latency = time.time() - t0
    extracted = extract_final_answer(raw) if raw else ""

    if raw:
        sc = step_count(raw)
        n_steps, n_sentences, step_method = sc.n_steps, len(sc.sentences), sc.method
    else:
        n_steps, n_sentences, step_method = 0, 0, "n/a"

    cell = {
        "model": recorded_model,
        "frame_id": item["frame_id"],
        "category": item.get("category", ""),
        "regime": regime,
        "condition": condition,
        "replication": rep,
        "seed": seed,
        "source": "frontier" if is_frontier(model_id) else "local",
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
    tag = "ERR" if err else f"steps={n_steps:3d}"
    msg = (f"  {model_id:18s} {item['frame_id']:8s} {condition} r{rep}  {tag:9s} "
           f"ans={extracted[:30]:30s} ({latency:5.1f}s)"
           + (f"  !{err[:60]}" if err else ""))
    print(msg)
    log.write(msg + "\n")
    log.flush()
    return cell


def atomic_write(registry) -> None:
    """Write the full union atomically (tmp + replace) so a crash mid-write never
    truncates the checkpoint. Accepts the {cell_key: cell} registry or a list.

    On Windows os.replace raises PermissionError [WinError 5] if ANY other process
    has the target open for even an instant (AV scanner, file indexer, or our own
    health-check json.load/ConvertFrom-Json). That is transient, so retry with
    backoff instead of crashing the whole run on a momentary lock."""
    cells = list(registry.values()) if isinstance(registry, dict) else registry
    tmp = RESULTS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cells, indent=2), encoding="utf-8")
    last_err = None
    for attempt in range(10):
        try:
            tmp.replace(RESULTS_PATH)
            return
        except PermissionError as e:  # WinError 5: target momentarily open elsewhere
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    raise last_err


# --------------------------------------------------------------------------- #
# Cost (dry run)
# --------------------------------------------------------------------------- #
def estimate_frontier_cost(items, conditions, reps) -> dict:
    sys_tok = len(SYSTEM_PROMPT) / 4
    pool = load_pool()
    per = {}
    for key, price in (("claude", PRICES["claude"]), ("gpt4o", PRICES["gpt4o"])):
        tin = tout = 0.0
        for fid, _ in items:
            item = pool[fid]
            for cond in conditions:
                prompt = build_user_prompt(item, cond)
                in_tok = len(prompt) / 4 + sys_tok
                out_tok = min(EST_OUT.get(cond, 500), MAX_TOKENS)
                tin += in_tok * reps
                tout += out_tok * reps
        cost = tin / 1e6 * price[0] + tout / 1e6 * price[1]
        per[key] = dict(in_tok=int(tin), out_tok=int(tout), usd=round(cost, 2))
    per["total_usd"] = round(sum(v["usd"] for v in per.values()
                                 if isinstance(v, dict)), 2)
    return per


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv):
    cfg = dict(mode="dry", models=None, conditions=CONDITIONS_ALL,
               reps=N_REPS_DEFAULT, regime=None, low_items=None)
    for a in argv:
        if a == "--run":
            cfg["mode"] = "run"
        elif a == "--smoke":
            cfg["mode"] = "smoke"
        elif a == "--local":
            cfg["models"] = list(LOCAL_MODELS)
        elif a == "--frontier":
            cfg["models"] = list(FRONTIER_MODELS)
        elif a == "--high-only":
            cfg["regime"] = "high"
        elif a == "--low-only":
            cfg["regime"] = "low"
        elif a.startswith("--models="):
            cfg["models"] = [s.strip() for s in a.split("=", 1)[1].split(",") if s.strip()]
        elif a.startswith("--conditions="):
            cfg["conditions"] = [s.strip() for s in a.split("=", 1)[1].split(",") if s.strip()]
        elif a.startswith("--reps="):
            cfg["reps"] = int(a.split("=", 1)[1])
        elif a.startswith("--low-items="):
            cfg["low_items"] = [s.strip() for s in a.split("=", 1)[1].split(",") if s.strip()]
    return cfg


def main(argv) -> None:
    cfg = parse_args(argv)
    pool = load_pool()
    items = select_items(pool, low_items=cfg["low_items"])
    if cfg["regime"]:
        items = [(f, r) for f, r in items if r == cfg["regime"]]
    n_high = sum(1 for _, r in items if r == "high")
    n_low = sum(1 for _, r in items if r == "low")

    models = cfg["models"] or (LOCAL_MODELS + FRONTIER_MODELS)
    conditions = cfg["conditions"]
    reps = cfg["reps"]

    planned = len(models) * len(items) * len(conditions) * reps
    print(f"[plan] pool=v0.4  models={models}")
    print(f"[plan] items={len(items)} ({n_high} high cbd + {n_low} low control)  "
          f"conditions={conditions}  reps={reps}")
    print(f"[plan] => {planned} cells total")

    # Resume accounting.
    existing, done = load_existing()
    todo = [
        (m, fid, regime, c, rep)
        for m in models
        for (fid, regime) in items
        for c in conditions
        for rep in range(1, reps + 1)
        if cell_key(canonical_model_id(m), fid, c, rep) not in done
    ]
    print(f"[resume] {len(existing)} cells in output, {len(done)} completed OK; "
          f"{len(todo)} of {planned} cells remain to collect")

    if cfg["mode"] == "dry":
        fr_models = [m for m in models if is_frontier(m)]
        if fr_models:
            cost = estimate_frontier_cost(items, conditions, reps)
            print(f"[cost] frontier estimate (full, not just remaining): "
                  f"${cost['total_usd']}  "
                  f"(claude ${cost['claude']['usd']}, gpt4o ${cost['gpt4o']['usd']})")
        print("[dry] no calls made. Re-run with --smoke (a few real cells) then "
              "--run (full; resumes automatically).")
        return

    if cfg["mode"] == "smoke":
        # 1 high + 1 low cell per model, first condition, rep 1 — wiring test.
        hi = next(((f, r) for f, r in items if r == "high"), None)
        lo = next(((f, r) for f, r in items if r == "low"), None)
        sset = [x for x in (hi, lo) if x]
        todo = [(m, f, r, conditions[0], 1) for m in models for (f, r) in sset]
        print(f"[smoke] {len(todo)} real cells ({[m for m in models]} × "
              f"{[f for f, _ in sset]} × {conditions[0]} × rep1)")

    registry = dict(existing)
    n_new = n_err = 0
    consec_err = 0          # tripwire: consecutive failures (credit-exhaustion / outage signature)
    TRIP_CONSEC = 8         # abort after this many back-to-back errors (saves hours of doomed retries)
    t0 = time.time()
    with open(LOG_PATH, "a", encoding="utf-8") as log:
        log.write(f"\n[start {time.strftime('%Y-%m-%d %H:%M:%S')}] mode={cfg['mode']} "
                  f"models={models} todo={len(todo)}\n")
        for (m, fid, regime, c, rep) in todo:
            cell = run_cell(m, pool[fid], regime, c, rep, log)
            registry[cell_key(canonical_model_id(m), fid, c, rep)] = cell  # canonical key = no dup, resumes
            n_new += 1
            if cell["error"]:
                n_err += 1
                consec_err += 1
            else:
                consec_err = 0
            if consec_err >= TRIP_CONSEC:
                atomic_write(registry)
                msg = (f"[ABORT] tripwire: {consec_err} consecutive errors "
                       f"(last: {str(cell['error'])[:120]}). Likely API credit/outage. "
                       f"Stopping after {n_new} attempts ({n_err} err). Fix, then re-run to resume.")
                print(msg)
                log.write(msg + "\n")
                raise SystemExit(2)
            if n_new % 10 == 0:
                atomic_write(registry)
                eta = (time.time() - t0) / n_new * (len(todo) - n_new)
                print(f"    [ckpt] {n_new}/{len(todo)} new ({n_err} err)  "
                      f"~{eta/3600:.1f}h remaining")
        atomic_write(registry)
        log.write(f"[done] +{n_new} new cells ({n_err} err) in "
                  f"{time.time()-t0:.0f}s; total {len(registry)}\n")

    atomic_write(registry)
    print(f"[done] +{n_new} new cells ({n_err} errors); {len(registry)} total in "
          f"{RESULTS_PATH.name} ({time.time()-t0:.0f}s)")
    if cfg["mode"] == "run" and not todo:
        print("[done] nothing to do — run already complete.")


if __name__ == "__main__":
    main(sys.argv[1:])
