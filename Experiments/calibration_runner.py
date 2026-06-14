"""
calibration_runner.py — pre-data effect-size + σ_g calibration for FEH H1.

Runs 3 local Ollama models × (28 cbd-scorable Knightian items + 14 easy
controls) × 5 length-graded CoT conditions × 3 reps = 1,890 cells.

High-regime outcome (scored later in calibration_analysis.py): cbd-correctness
— the normatively correct answer to a genuine Knightian item is
'cannot-be-determined'. H1 predicts cbd-correctness DROPS as reasoning steps
increase (Theorem 2.6.1 cue-truncation). Low-regime outcome: gold-match on easy
reference/aleatory items.

NOT the confirmatory test (that is pre-registered for the full 5-model panel,
eq. 6.1). This is a feasibility / effect-size / σ_g estimate to decide whether
to launch the full run. See Calibration_protocol.md.

Pure helpers are imported from pilot_runner.py (that working file is NOT
edited). Only the Ollama call is re-implemented to take a model_id argument.

Outputs:
  calibration_responses.json — per-cell records (raw + extracted + steps + meta)
  calibration_run.log
"""

import json
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))        # for pilot_runner
sys.path.insert(0, str(REPO_ROOT))   # for step_counter

from pilot_runner import (  # noqa: E402  (pure helpers; no side effects on import)
    CONDITION_TEMPLATES,
    NUM_PREDICT,
    OLLAMA_URL,
    REQUEST_TIMEOUT,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
    build_user_prompt,
    cell_seed,
    extract_final_answer,
    load_pool,
)
from step_counter import step_count  # noqa: E402

# Windows console / redirected-file default is cp1252, which crashes on model
# answers containing non-cp1252 glyphs (≈, ×, ², non-Latin scripts). Force UTF-8
# with errors="replace" so a stray glyph in a printed answer never kills the run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001  (older/odd streams without reconfigure)
        pass

RESULTS_PATH = HERE / "calibration_responses.json"
LOG_PATH = HERE / "calibration_run.log"

MODELS = ["mistral:7b-instruct", "qwen2.5:7b", "llama3.1:8b"]

# 28 cbd-scorable Knightian items (auto-string-match grading + cbd choice).
HIGH_ITEMS = [
    "K1-001", "K1-002", "K1-003", "K1-004", "K1-006", "K1-007", "K1-008",
    "K1-009", "K1-010", "K1-011", "K1-012", "K1-013", "K1-014", "K1-015",
    "K1-016", "K1-017", "K1-018", "K1-019", "K1-020",
    "K2-003", "K2-005", "K2-006", "K2-008", "K2-012", "K2-015", "K2-016",
    "K4-002", "K4-003",
]
# 14 low-regime controls small models can actually do (avoids R-001 = 47×83,
# which floors 7B models). R-101..R-106 = easy arithmetic tier; A = aleatory.
LOW_ITEMS = [
    "R-101", "R-102", "R-103", "R-104", "R-105", "R-106",
    "A-001", "A-002", "A-003", "A-004", "A-005", "A-006", "A-007", "A-008",
]


def call_ollama(model_id: str, prompt: str, seed: int) -> dict:
    """Single Ollama generate call with a deterministic seed."""
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
    return r.json()


def run_cell(model_id: str, item: dict, regime: str, condition: str, rep: int,
             log) -> dict:
    seed = cell_seed(model_id, item["frame_id"], condition, rep)
    user = build_user_prompt(item, condition)
    t0 = time.time()
    try:
        resp = call_ollama(model_id, user, seed)
        raw = resp.get("response", "")
        err = None
        eval_count = resp.get("eval_count")
        prompt_eval_count = resp.get("prompt_eval_count")
    except Exception as e:  # noqa: BLE001  (record and continue; one bad cell != lost run)
        raw = ""
        err = f"{type(e).__name__}: {e}"
        eval_count = None
        prompt_eval_count = None
    latency = time.time() - t0
    extracted = extract_final_answer(raw) if raw else ""

    if raw:
        sc = step_count(raw)
        n_steps = sc.n_steps
        n_sentences = len(sc.sentences)
        step_method = sc.method
    else:
        n_steps = 0
        n_sentences = 0
        step_method = "n/a"

    cell = {
        "model": model_id,
        "frame_id": item["frame_id"],
        "category": item["category"],
        "regime": regime,                       # "high" | "low"
        "condition": condition,
        "replication": rep,
        "seed": seed,
        "prompt": user,
        "raw_response": raw,
        "extracted_final_answer": extracted,
        "n_steps_heuristic": n_steps,
        "n_sentences": n_sentences,
        "step_count_method": step_method,
        "tokens_in": prompt_eval_count,
        "tokens_out": eval_count,
        "latency_s": round(latency, 2),
        "error": err,
    }
    out_tok = str(eval_count) if eval_count is not None else "-"
    msg = (
        f"  {model_id:20s} {item['frame_id']:8s} {condition} r{rep}  "
        f"steps={n_steps:3d} out_tok={out_tok:>4s} "
        f"ans={extracted[:32]:32s} ({latency:5.1f}s)"
    )
    print(msg)
    log.write(msg + "\n")
    log.flush()
    return cell


def main(smoke: bool = False, models=None, n_reps: int = 3) -> None:
    pool = load_pool()
    frames = [(f, "high") for f in HIGH_ITEMS] + [(f, "low") for f in LOW_ITEMS]
    missing = [f for f, _ in frames if f not in pool]
    if missing:
        raise SystemExit(f"missing from pool: {missing}")
    model_list = models or MODELS
    cond_list = ["C1", "C2", "C3", "C4", "C5"]

    if smoke:
        frames = frames[:1] + frames[-1:]   # 1 high + 1 low
        model_list = model_list[:1]
        cond_list = cond_list[:1]
        n_reps = 1

    n_cells = len(model_list) * len(frames) * len(cond_list) * n_reps
    print(
        f"[start] {len(model_list)} models × {len(frames)} items × "
        f"{len(cond_list)} conditions × {n_reps} reps = {n_cells} cells"
    )

    cells: list = []
    t_start = time.time()
    with open(LOG_PATH, "w", encoding="utf-8") as log:
        log.write(f"[start] {n_cells} cells, models={model_list}\n")
        # Model-outer loop: each model loads into VRAM once.
        for model_id in model_list:
            log.write(f"[model] {model_id}\n")
            log.flush()
            for frame_id, regime in frames:
                item = pool[frame_id]
                for condition in cond_list:
                    for rep in range(1, n_reps + 1):
                        cells.append(
                            run_cell(model_id, item, regime, condition, rep, log)
                        )
                        if len(cells) % 10 == 0:   # crash-safe snapshot
                            RESULTS_PATH.write_text(
                                json.dumps(cells, indent=2), encoding="utf-8"
                            )
        log.write(f"[done] {len(cells)} cells in {time.time()-t_start:.0f}s\n")

    RESULTS_PATH.write_text(json.dumps(cells, indent=2), encoding="utf-8")
    elapsed = time.time() - t_start
    n_err = sum(1 for c in cells if c["error"])
    print(f"[done] wrote {RESULTS_PATH.name}; {len(cells)} cells "
          f"({n_err} errors) in {elapsed:.0f}s")


if __name__ == "__main__":
    arg_smoke = "--smoke" in sys.argv[1:]
    arg_reps = 3
    arg_models = None
    for arg in sys.argv[1:]:
        if arg.startswith("--reps="):
            arg_reps = int(arg.split("=", 1)[1])
        elif arg.startswith("--models="):
            arg_models = [s.strip() for s in arg.split("=", 1)[1].split(",") if s.strip()]
    main(smoke=arg_smoke, models=arg_models, n_reps=arg_reps)
