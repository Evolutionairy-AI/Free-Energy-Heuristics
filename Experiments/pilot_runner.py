"""
pilot_runner.py — §4.9 pilot data collection for FEH-79 v0.3.

Per pre-registration §11 + Chapter 4 §4.9:
  Mistral-7B-Instruct (via Ollama) × 10 designer-curated pilot frames
  × 5 length-graded CoT conditions × 3 replications = 150 cells.

For each cell:
  1. Build the C{1..5} user prompt per `prompt_templates.md`.
  2. Generate via Ollama POST /api/generate (deterministic seed per cell).
  3. Extract the "Final answer:" label (FINAL_ANSWER_RE from §4.6).
  4. Compute the heuristic step count via `step_counter.py` (Pass 1 only;
     LLM-judge step counting is run separately during analysis).
  5. Record raw + extracted + n_steps + latency + seed + token counts.

Outputs:
  pilot_responses.json — full structured record (raw responses + metadata).
  pilot_run.log         — per-cell progress log.

Args:
  --smoke           run 1 cell (1 item × 1 condition × 1 rep) and exit
  --items=A,B,C     restrict to specific frame_ids (otherwise full pilot 10)
  --conditions=1,3  restrict to specific conditions (default: 1..5)
  --reps=N          number of replications per cell (default 3)
"""

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from step_counter import step_count  # noqa: E402

POOL_PATH = REPO_ROOT / "feh79_item_pool_v0.3.yaml"
RESULTS_PATH = REPO_ROOT / "Experiments" / "pilot_responses.json"
LOG_PATH = REPO_ROOT / "Experiments" / "pilot_run.log"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_ID = "mistral:7b-instruct"
TEMPERATURE = 0.7
TOP_P = 0.95
NUM_PREDICT = 2048
REQUEST_TIMEOUT = 600

# §4.9.1 — designer-curated pilot subset.
PILOT_FRAMES = [
    "K1-001", "K1-005",
    "K2-005", "K2-006",
    "K3-001", "K3-005",
    "K4-003", "K4-004",
    "R-001",
    "A-003",
]

SYSTEM_PROMPT = (
    "You are a thoughtful assistant. Answer the user's question to the "
    "best of your ability. If asked to think step by step, structure "
    "your reasoning clearly. Finish with a single-line final answer of "
    "the form 'Final answer: <X>'."
)

CONDITION_TEMPLATES = {
    "C1": "{q}\n\nGive a single-line answer of the form 'Final answer: <X>' with no other text.",
    "C2": "{q}\n\nThink step by step, briefly, in 3 steps or fewer. Then give a single-line final answer of the form 'Final answer: <X>'.",
    "C3": "{q}\n\nThink step by step in about 7 steps. Then give a single-line final answer of the form 'Final answer: <X>'.",
    "C4": "{q}\n\nReason through this carefully, considering multiple angles, in approximately 15 steps. Then give a single-line final answer of the form 'Final answer: <X>'.",
    "C5": "{q}\n\nThink step by step. When you've reached a conclusion, finish with a single-line answer of the form 'Final answer: <X>'.",
}

FINAL_ANSWER_RE = re.compile(
    r"final\s*answer\s*[:\-=]\s*(.+?)(?:\n|$)",
    re.IGNORECASE | re.DOTALL,
)


def cell_seed(model_id: str, frame_id: str, condition: str, rep: int) -> int:
    """Deterministic seed per (model, item, condition, replication) cell."""
    h = hashlib.sha256(f"{model_id}|{frame_id}|{condition}|{rep}".encode()).hexdigest()
    return int(h[:8], 16)  # 32-bit unsigned


def build_user_prompt(item: dict, condition: str) -> str:
    """C1..C5 prompt with answer-choice injection per §4.6."""
    q = item["question"].strip()
    choices = item.get("answer_choices")
    if choices:
        q = q + f"\n\nChoose from: {', '.join(str(c) for c in choices)}"
    return CONDITION_TEMPLATES[condition].format(q=q)


def extract_final_answer(text: str) -> str:
    m = FINAL_ANSWER_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(".").lower()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return (lines[-1] if lines else "").rstrip(".").lower()


def call_mistral_ollama(prompt: str, seed: int) -> dict:
    """Single Ollama call with deterministic seed."""
    body = {
        "model": MODEL_ID,
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


def load_pool() -> dict:
    with open(POOL_PATH, encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))
    return {d["frame_id"]: d for d in docs if d and "frame_id" in d}


def run_cell(item: dict, condition: str, rep: int, log) -> dict:
    seed = cell_seed(MODEL_ID, item["frame_id"], condition, rep)
    user = build_user_prompt(item, condition)
    t0 = time.time()
    try:
        resp = call_mistral_ollama(user, seed)
        raw = resp.get("response", "")
        err = None
        eval_count = resp.get("eval_count")
        prompt_eval_count = resp.get("prompt_eval_count")
    except Exception as e:
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
        "model": MODEL_ID,
        "frame_id": item["frame_id"],
        "category": item["category"],
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
    out_tok_str = str(eval_count) if eval_count is not None else "-"
    msg = (
        f"  {item['frame_id']:8s} {condition} r{rep}  steps={n_steps:3d}  "
        f"out_tok={out_tok_str:>4s}  ans={extracted[:40]:40s}  ({latency:5.1f}s)"
    )
    print(msg)
    log.write(msg + "\n")
    log.flush()
    return cell


def main(
    smoke: bool = False,
    items: list[str] | None = None,
    conditions: list[str] | None = None,
    n_reps: int = 3,
) -> None:
    pool = load_pool()
    frames = items or PILOT_FRAMES
    missing = [f for f in frames if f not in pool]
    if missing:
        raise SystemExit(f"missing from pool: {missing}")
    pilot_items = [pool[f] for f in frames]
    cond_list = conditions or ["C1", "C2", "C3", "C4", "C5"]

    if smoke:
        pilot_items = pilot_items[:1]
        cond_list = cond_list[:1]
        n_reps = 1

    n_cells = len(pilot_items) * len(cond_list) * n_reps
    print(
        f"[start] {len(pilot_items)} items × {len(cond_list)} conditions × {n_reps} reps "
        f"= {n_cells} cells (model={MODEL_ID})"
    )

    cells: list = []
    t_start = time.time()
    with open(LOG_PATH, "w", encoding="utf-8") as log:
        log.write(f"[start] {n_cells} cells, model={MODEL_ID}\n")
        for item in pilot_items:
            for condition in cond_list:
                for rep in range(1, n_reps + 1):
                    cells.append(run_cell(item, condition, rep, log))
                    # Snapshot every 10 cells in case of crash.
                    if len(cells) % 10 == 0:
                        RESULTS_PATH.write_text(
                            json.dumps(cells, indent=2), encoding="utf-8"
                        )
        log.write(f"[done] {len(cells)} cells in {time.time()-t_start:.0f}s\n")

    RESULTS_PATH.write_text(json.dumps(cells, indent=2), encoding="utf-8")
    elapsed = time.time() - t_start
    print(f"[done] wrote {RESULTS_PATH.name}; {len(cells)} cells in {elapsed:.0f}s")


if __name__ == "__main__":
    arg_smoke = False
    arg_items = None
    arg_cond = None
    arg_reps = 3
    for arg in sys.argv[1:]:
        if arg == "--smoke":
            arg_smoke = True
        elif arg.startswith("--items="):
            arg_items = [s.strip() for s in arg.split("=", 1)[1].split(",") if s.strip()]
        elif arg.startswith("--conditions="):
            arg_cond = [f"C{s.strip()}" if not s.strip().startswith("C") else s.strip()
                        for s in arg.split("=", 1)[1].split(",") if s.strip()]
        elif arg.startswith("--reps="):
            arg_reps = int(arg.split("=", 1)[1])
    main(smoke=arg_smoke, items=arg_items, conditions=arg_cond, n_reps=arg_reps)
