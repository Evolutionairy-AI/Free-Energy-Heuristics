"""
step_counter_kappa.py — §4.8.3 IRR validation for the step-counter pipeline.

Per Chapter 4 §4.8: the heuristic step classifier is the fast Pass-1 baseline,
the LLM-judge is the primary Pass-2 classifier. We need to demonstrate that
either (a) the heuristic agrees with the LLM-judge well enough to be useful as
a sanity-check, or (b) the LLM-judge agreement against a hand-coded subset
clears Cohen's kappa >= 0.7.

This script implements (a): on a stratified subsample (1 rep per
frame x condition cell = 50 cells = ~540 sentences) drawn from the §4.9 pilot
data, we compute Cohen's kappa between the heuristic classifier and Claude
(claude-sonnet-4-5) acting as the LLM-judge. Outputs:

  Experiments/kappa_validation.json — per-sentence labels + summary
  Experiments/kappa_validation.md   — markdown report (kappa, confusion, n)

If the user later supplies a hand-coded subset, the same kappa machinery in
step_counter.cohen_kappa will be reusable for the human-vs-LLM comparison
(§4.8.3 primary validation).

Args:
  --smoke              run on 5 sentences and exit (cost check)
  --no-throttle        skip the per-call sleep (use only with paid tier)
  --max-cells=N        cap subsample at N cells (default: all 50 strat cells)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Norton SSL MITM workaround (cf. cross_model_check.py).
import truststore
truststore.inject_into_ssl()

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from step_counter import (  # noqa: E402
    LLM_JUDGE_PROMPT,
    cohen_kappa,
    is_step_heuristic,
    segment_sentences,
)

PILOT_PATH = REPO_ROOT / "Experiments" / "pilot_responses.json"
KEYS_DIR = REPO_ROOT / "Experiments" / "API_KEYS"
OUT_JSON = REPO_ROOT / "Experiments" / "kappa_validation.json"
OUT_MD = REPO_ROOT / "Experiments" / "kappa_validation.md"

CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
JUDGE_TEMPERATURE = 0.0  # deterministic judging
JUDGE_MAX_TOKENS = 8     # only need "STEP" or "NOT-STEP"
THROTTLE_S = 0.5         # per-call delay; Anthropic tier-2 allows ~50 RPM safely
REQUEST_TIMEOUT = 120


def load_anthropic_key() -> str:
    text = (KEYS_DIR / "Claude_Key.txt").read_text(encoding="utf-8").strip()
    if "=" in text and "\n" not in text:
        text = text.split("=", 1)[1].strip()
    return text


def claude_judge(key: str, sentence: str) -> str:
    body = {
        "model": CLAUDE_MODEL,
        "max_tokens": JUDGE_MAX_TOKENS,
        "temperature": JUDGE_TEMPERATURE,
        "messages": [
            {"role": "user", "content": LLM_JUDGE_PROMPT.format(sentence=sentence)}
        ],
    }
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=body,
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    text = r.json()["content"][0]["text"].strip().upper()
    if text.startswith("STEP"):
        return "STEP"
    return "NOT-STEP"


def confusion(heur: list[str], judge: list[str]) -> dict:
    """2x2 confusion: judge as gold, heuristic as predicted."""
    out = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for h, j in zip(heur, judge):
        if j == "STEP" and h == "STEP":
            out["TP"] += 1
        elif j == "NOT-STEP" and h == "STEP":
            out["FP"] += 1
        elif j == "STEP" and h == "NOT-STEP":
            out["FN"] += 1
        else:
            out["TN"] += 1
    return out


def main(smoke: bool, throttle: bool, max_cells: int | None) -> None:
    pilot = json.loads(PILOT_PATH.read_text(encoding="utf-8"))
    # Stratified: one replication per (frame, condition) cell.
    strat = [c for c in pilot if c.get("replication") == 1 and c.get("raw_response")]
    if max_cells is not None:
        strat = strat[:max_cells]

    # Build (cell_meta, sent_idx, sentence, heuristic_label) records.
    records: list[dict] = []
    for c in strat:
        sents = segment_sentences(c["raw_response"])
        for i, s in enumerate(sents):
            records.append({
                "frame_id": c["frame_id"],
                "category": c["category"],
                "condition": c["condition"],
                "sent_idx": i,
                "sentence": s,
                "heuristic": "STEP" if is_step_heuristic(s) else "NOT-STEP",
            })

    if smoke:
        records = records[:5]
        print(f"[smoke] {len(records)} sentences")

    print(f"[start] judging {len(records)} sentences with {CLAUDE_MODEL}")
    key = load_anthropic_key()
    t0 = time.time()
    n_err = 0
    for idx, rec in enumerate(records):
        try:
            rec["judge"] = claude_judge(key, rec["sentence"])
        except Exception as e:
            rec["judge"] = "ERROR"
            rec["error"] = f"{type(e).__name__}: {e}"
            n_err += 1
        if (idx + 1) % 25 == 0 or idx + 1 == len(records):
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (len(records) - idx - 1) / rate if rate > 0 else 0
            print(
                f"  {idx + 1:4d}/{len(records)}  errs={n_err}  "
                f"rate={rate:.1f}/s  eta={eta:.0f}s"
            )
        if throttle and idx + 1 < len(records):
            time.sleep(THROTTLE_S)

    judged = [r for r in records if r["judge"] in ("STEP", "NOT-STEP")]
    heur = [r["heuristic"] for r in judged]
    judge = [r["judge"] for r in judged]
    kappa = cohen_kappa(heur, judge) if judged else 0.0
    cm = confusion(heur, judge)
    n = len(judged)
    accuracy = (cm["TP"] + cm["TN"]) / n if n else 0.0
    p_step_heur = sum(1 for h in heur if h == "STEP") / n if n else 0.0
    p_step_judge = sum(1 for j in judge if j == "STEP") / n if n else 0.0

    summary = {
        "model": CLAUDE_MODEL,
        "n_sentences": n,
        "n_errors": n_err,
        "kappa": round(kappa, 3),
        "accuracy": round(accuracy, 3),
        "p_step_heuristic": round(p_step_heur, 3),
        "p_step_judge": round(p_step_judge, 3),
        "confusion": cm,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print()
    print(f"[done] kappa = {kappa:+.3f}  (target >= 0.7)")
    print(f"       accuracy = {accuracy:.3f}  n = {n}  errors = {n_err}")

    OUT_JSON.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2),
        encoding="utf-8",
    )

    md = _build_md(summary, records)
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"[write] {OUT_JSON.name}  {OUT_MD.name}")


def _build_md(summary: dict, records: list[dict]) -> str:
    """Produce the kappa_validation.md report."""
    cm = summary["confusion"]
    verdict = "PASS" if summary["kappa"] >= 0.7 else "FAIL"
    lines = [
        "# §4.8.3 Step-counter IRR Validation",
        "",
        f"**Judge model**: `{summary['model']}`  ",
        f"**Sample**: {summary['n_sentences']} sentences from §4.9 pilot "
        "(rep=1, all 10 frames x 5 conditions)  ",
        f"**Errors during judging**: {summary['n_errors']}",
        "",
        "## Headline",
        "",
        f"| Metric | Value | Target |",
        f"|--------|-------|--------|",
        f"| Cohen's kappa (heuristic vs LLM-judge) | **{summary['kappa']:+.3f}** | >= 0.70 ({verdict}) |",
        f"| Raw agreement (accuracy) | {summary['accuracy']:.3f} | - |",
        f"| P(STEP) heuristic | {summary['p_step_heuristic']:.3f} | - |",
        f"| P(STEP) LLM-judge | {summary['p_step_judge']:.3f} | - |",
        "",
        "## Confusion (LLM-judge as gold, heuristic as predicted)",
        "",
        "| | judge=STEP | judge=NOT-STEP |",
        "|---|---|---|",
        f"| heuristic=STEP | {cm['TP']} | {cm['FP']} |",
        f"| heuristic=NOT-STEP | {cm['FN']} | {cm['TN']} |",
        "",
    ]

    # Per-condition breakdown.
    by_cond: dict[str, list[dict]] = {}
    for r in records:
        if r["judge"] in ("STEP", "NOT-STEP"):
            by_cond.setdefault(r["condition"], []).append(r)
    lines += [
        "## Per-condition breakdown",
        "",
        "| condition | n_sent | kappa | accuracy | P(STEP) heur | P(STEP) judge |",
        "|-----------|--------|-------|----------|--------------|---------------|",
    ]
    for cond in sorted(by_cond):
        rs = by_cond[cond]
        h = [r["heuristic"] for r in rs]
        j = [r["judge"] for r in rs]
        k = cohen_kappa(h, j)
        acc = sum(1 for a, b in zip(h, j) if a == b) / len(rs)
        ph = sum(1 for x in h if x == "STEP") / len(rs)
        pj = sum(1 for x in j if x == "STEP") / len(rs)
        lines.append(
            f"| {cond} | {len(rs)} | {k:+.3f} | {acc:.3f} | {ph:.3f} | {pj:.3f} |"
        )

    # Disagreement examples — first 10 cases where heur != judge.
    disagree = [r for r in records if r["judge"] in ("STEP", "NOT-STEP")
                and r["heuristic"] != r["judge"]]
    lines += [
        "",
        f"## Disagreement examples ({len(disagree)} total; first 10 shown)",
        "",
        "| frame | cond | heur | judge | sentence |",
        "|-------|------|------|-------|----------|",
    ]
    for r in disagree[:10]:
        sent = r["sentence"].replace("|", "\\|").replace("\n", " ")
        if len(sent) > 120:
            sent = sent[:117] + "..."
        lines.append(
            f"| {r['frame_id']} | {r['condition']} | {r['heuristic']} | "
            f"{r['judge']} | {sent} |"
        )

    lines += [
        "",
        f"## Verdict",
        "",
        f"**{verdict}** — kappa = {summary['kappa']:+.3f} vs threshold 0.70.",
        "",
        "*Interpretation*: this validates that the fast heuristic step-counter "
        "agrees with the LLM-judge well enough to be used as the Pass-1 "
        "step-count metric in the full study, with the LLM-judge reserved for "
        "the §4.8.2 primary measurement on a per-cell sub-sample. If kappa "
        "falls below 0.7, the LLM-judge becomes the only valid step-counter "
        "and per-cell judging cost is the binding constraint on study size.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    arg_smoke = "--smoke" in sys.argv
    arg_no_throttle = "--no-throttle" in sys.argv
    arg_max = None
    for a in sys.argv[1:]:
        if a.startswith("--max-cells="):
            arg_max = int(a.split("=", 1)[1])
    main(smoke=arg_smoke, throttle=not arg_no_throttle, max_cells=arg_max)
