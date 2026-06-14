"""
pilot_analysis.py — Evaluate §4.9.2 pilot success criteria on
pilot_responses.json.

Scope (descriptive only; not the confirmatory test, per §4.9.3):
  - **(P1)** Regime score on 8 K items > on R + A items (rank-order check).
        Pilot regime indicator = cross-seed disagreement σ_b only.
        Cross-prompt (σ_a) and calibration (σ_c) require paraphrases /
        confidence elicitations not in the pilot spec; deferred to full
        run.
  - **(P3)** Auto-grading + extraction completeness ≥ 90%.
  - **(P4)** Descriptive: in how many of the 8 K items does the
        accuracy/coherence trajectory across C1..C5 show a non-monotone
        (peak-then-decline) pattern?

P2 (step-count κ vs human) requires human coding of a subsample and is
out of scope for this script.

Outputs:
  pilot_analysis.json   — structured per-item summary.
  pilot_analysis.md     — human-readable report with verdicts.
"""

import json
import re
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
POOL_PATH = REPO_ROOT / "feh79_item_pool_v0.3.yaml"
RESPONSES_PATH = REPO_ROOT / "Experiments" / "pilot_responses.json"
ANALYSIS_JSON = REPO_ROOT / "Experiments" / "pilot_analysis.json"
ANALYSIS_MD = REPO_ROOT / "Experiments" / "pilot_analysis.md"

# Gold answers for R/A control items (the only pilot items with gold).
GOLD_ANSWERS = {
    "R-001": ["3901", "3,901"],          # 47 × 83
    "A-003": ["1/6", "0.1667", "16.67%", "0.166666...", "1/6 ≈ 0.1667"],
    # (A-003 = P(sum on two dice = 7) = 6/36 = 1/6.)
}

# CBD-equivalent normalized answers (per §4.3 v0.2).
CBD_TOKENS = {
    "cannot-be-determined", "cannot be determined", "cannot-be",
    "unknown", "uncertain", "unclear", "depends", "conditional",
    "cant-tell", "indeterminate", "cannot-determine", "cannot determine",
    "cannot predict", "cannot-predict", "no prediction", "no-prediction",
    "no answer", "no-answer", "indeterminable",
}


def normalize_with_synonyms(answer: str, synonyms: dict | None) -> str:
    """Map extracted answer to canonical answer_choice via synonyms dict.
    Mirrors cross_model_check.py — normalization for K2-validated categorical
    items. Falls back to CBD detection for cbd-equivalent phrasings."""
    if not answer:
        return ""
    norm = answer.lower().strip().rstrip(".,!? ").strip("'\"")

    # CBD detection (works regardless of synonyms dict)
    for cbd in CBD_TOKENS:
        if cbd in norm:
            return "cannot-be-determined"

    if not synonyms:
        return norm

    for canonical, syns in synonyms.items():
        if norm == str(canonical).lower():
            return str(canonical).lower()
        for s in syns or []:
            if norm == str(s).lower():
                return str(canonical).lower()
    # Substring containment.
    for canonical, syns in synonyms.items():
        for token in [canonical] + list(syns or []):
            if str(token).lower() in norm:
                return str(canonical).lower()
    # Prefix overlap.
    if len(norm) >= 4:
        for canonical, syns in synonyms.items():
            for token in [canonical] + list(syns or []):
                t = str(token).lower()
                if t.startswith(norm) or norm.startswith(t):
                    return str(canonical).lower()
    return norm


# Numeric normalization for R/A items (also handles A-003 fraction formats).
NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?(?:/\d+)?")


def numeric_normalize(answer: str) -> str:
    """Extract first number-like token, normalize fractions and percents to a
    decimal string with 4 sig figs. Returns '' if no number found."""
    if not answer:
        return ""
    m = NUMBER_RE.search(answer)
    if not m:
        return ""
    s = m.group(0).replace(",", "")
    try:
        if "/" in s:
            num, den = s.split("/")
            v = float(num) / float(den)
        else:
            v = float(s)
        # Detect percent — if "%" appears in the answer text near the number, divide by 100.
        if "%" in answer.lower():
            v = v / 100.0
        return f"{v:.4g}"
    except (ValueError, ZeroDivisionError):
        return s


def categorical_disagreement(answers: list[str]) -> float:
    """1 - max-frequency. Range [0, (n-1)/n]. Higher = more disagreement."""
    answers = [a for a in answers if a]
    if not answers:
        return 0.0
    counts = Counter(answers)
    max_c = max(counts.values())
    return 1.0 - max_c / len(answers)


def load_pool() -> dict:
    with open(POOL_PATH, encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))
    return {d["frame_id"]: d for d in docs if d and "frame_id" in d}


def is_correct(extracted: str, gold_list: list[str]) -> bool:
    """Loose match against any gold answer string."""
    if not extracted:
        return False
    e = extracted.strip().lower().rstrip(".")
    for g in gold_list:
        gn = g.strip().lower().rstrip(".")
        if e == gn or gn in e or e in gn:
            return True
    return False


def main() -> None:
    cells = json.loads(RESPONSES_PATH.read_text(encoding="utf-8"))
    pool = load_pool()

    # Group by (frame_id, condition)
    by_fc: dict = {}
    for c in cells:
        by_fc.setdefault((c["frame_id"], c["condition"]), []).append(c)

    # Per-frame summary
    frames = sorted({c["frame_id"] for c in cells})
    per_frame = {}
    for fid in frames:
        category = next(c["category"] for c in cells if c["frame_id"] == fid)
        item = pool.get(fid, {})
        synonyms = item.get("synonyms")
        is_open_ended = (
            item.get("answer_format") in ("short-open", "open")
            or not item.get("answer_choices")
        )
        cond_summary = {}
        cell_count = 0
        extract_ok = 0
        for cond in ("C1", "C2", "C3", "C4", "C5"):
            cells_fc = by_fc.get((fid, cond), [])
            if not cells_fc:
                continue
            raw_answers = [c["extracted_final_answer"] for c in cells_fc]
            # Normalize: numeric for R/A items, synonyms-based for K items,
            # raw for K3 open-ended (where σ_b is poorly defined anyway).
            if fid in GOLD_ANSWERS:
                norm_answers = [numeric_normalize(a) or normalize_with_synonyms(a, synonyms) for a in raw_answers]
            elif synonyms:
                norm_answers = [normalize_with_synonyms(a, synonyms) for a in raw_answers]
            else:
                # Open-ended K3: try CBD detection only; otherwise leave raw.
                norm_answers = [normalize_with_synonyms(a, None) or a for a in raw_answers]
            sig_b = categorical_disagreement(norm_answers)
            cbd_rate = sum(1 for a in norm_answers if a == "cannot-be-determined") / len(norm_answers)
            answers = norm_answers  # downstream uses normalized values
            n_steps = [c["n_steps_heuristic"] for c in cells_fc]
            mean_steps = sum(n_steps) / len(n_steps) if n_steps else 0
            cell_count += len(cells_fc)
            extract_ok += sum(1 for a in answers if a)
            # Accuracy if gold available
            if fid in GOLD_ANSWERS:
                correct = [is_correct(a, GOLD_ANSWERS[fid]) for a in answers]
                acc = sum(correct) / len(correct) if correct else None
            else:
                acc = None
            cond_summary[cond] = {
                "n_cells": len(cells_fc),
                "answers": answers,
                "raw_answers": raw_answers,
                "sig_b": round(sig_b, 3),
                "cbd_rate": round(cbd_rate, 3),
                "mean_steps": round(mean_steps, 1),
                "accuracy": (round(acc, 3) if acc is not None else None),
            }
        # Aggregate sig_b and cbd_rate across conditions (mean)
        sig_bs = [v["sig_b"] for v in cond_summary.values() if v["sig_b"] is not None]
        cbd_rates = [v["cbd_rate"] for v in cond_summary.values() if v["cbd_rate"] is not None]
        mean_sig_b = round(sum(sig_bs) / len(sig_bs), 3) if sig_bs else 0.0
        mean_cbd_rate = round(sum(cbd_rates) / len(cbd_rates), 3) if cbd_rates else 0.0
        # Composite "Knightian-signal" indicator: high if either sig_b is high
        # OR cbd_rate is high. Captures both K2-pass-disagreement (high sig_b)
        # and K2-pass-cbd (high cbd_rate) flavors of meta-uncertainty.
        knightian_signal = round(max(mean_sig_b, mean_cbd_rate), 3)
        per_frame[fid] = {
            "category": category,
            "is_open_ended": is_open_ended,
            "n_cells": cell_count,
            "extract_ok": extract_ok,
            "extract_ok_frac": round(extract_ok / cell_count, 3) if cell_count else 0,
            "mean_sig_b": mean_sig_b,
            "mean_cbd_rate": mean_cbd_rate,
            "knightian_signal": knightian_signal,
            "by_condition": cond_summary,
        }

    # P1 — rank-order: K items vs R+A on the composite Knightian-signal
    # indicator (the §3.2 regime score's pilot-feasible proxy).
    # Excludes K3 open-ended items (sig_b not well-defined for short-open
    # responses without semantic clustering — flagged for full-run protocol).
    k_frames = [f for f in frames if f.startswith(("K1-", "K2-", "K3-", "K4-"))]
    k_frames_categorical = [f for f in k_frames if not per_frame[f]["is_open_ended"]]
    ra_frames = [f for f in frames if f.startswith(("R-", "A-"))]
    k_signals = [per_frame[f]["knightian_signal"] for f in k_frames_categorical]
    ra_signals = [per_frame[f]["knightian_signal"] for f in ra_frames]
    k_mean = sum(k_signals) / len(k_signals) if k_signals else 0
    ra_mean = sum(ra_signals) / len(ra_signals) if ra_signals else 0
    p1_pass = bool(k_signals) and bool(ra_signals) and k_mean > ra_mean

    # P3 — extraction success ≥ 90% across all cells
    total = len(cells)
    extract_ok_total = sum(1 for c in cells if c["extracted_final_answer"])
    p3_frac = extract_ok_total / total if total else 0.0
    p3_pass = p3_frac >= 0.9

    # P4 — non-monotone pattern in accuracy or σ_b across C1..C5 for K items
    # Pilot uses σ_b trajectory across conditions as proxy for regime structure.
    p4_count = 0
    for fid in k_frames:
        sb_traj = [per_frame[fid]["by_condition"].get(c, {}).get("sig_b")
                   for c in ("C1", "C2", "C3", "C4", "C5")]
        sb_traj = [s for s in sb_traj if s is not None]
        if len(sb_traj) >= 3:
            # Non-monotone if max is strictly interior (not at C1 or last)
            mx = max(sb_traj)
            mx_idx = sb_traj.index(mx)
            if 0 < mx_idx < len(sb_traj) - 1:
                p4_count += 1

    summary = {
        "n_cells_total": total,
        "extract_ok_total": extract_ok_total,
        "extract_ok_frac": round(p3_frac, 3),
        "k_frames": k_frames,
        "k_frames_categorical": k_frames_categorical,
        "ra_frames": ra_frames,
        "k_mean_signal": round(k_mean, 3),
        "ra_mean_signal": round(ra_mean, 3),
        "P1_pass": p1_pass,
        "P3_pass": p3_pass,
        "P4_n_K_non_monotone": p4_count,
        "per_frame": per_frame,
    }
    ANALYSIS_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Markdown report
    lines = ["# §4.9 Pilot Analysis — FEH-79 v0.3, Mistral-7B-Instruct\n"]
    lines.append(
        f"**Cells**: {total}. **Extraction success**: {extract_ok_total}/{total} = {p3_frac:.1%}\n"
    )
    lines.append("## Pilot success criteria (per §4.9.2)\n")
    lines.append(
        f"- **P1** (K mean Knightian-signal > R+A mean, rank-order on "
        f"composite of σ_b + cbd_rate; K3 open-ended excluded): "
        f"{'✅ pass' if p1_pass else '⚠️ fail'} "
        f"(K={summary['k_mean_signal']}, R+A={summary['ra_mean_signal']})"
    )
    lines.append("- **P2** (step-count κ ≥ 0.7): out-of-scope for this script — requires human coding")
    lines.append(
        f"- **P3** (extract ≥ 90%): "
        f"{'✅ pass' if p3_pass else '⚠️ fail'} ({p3_frac:.1%})"
    )
    lines.append(
        f"- **P4** (≥1 K item non-monotone σ_b trajectory): "
        f"{'✅ pass' if p4_count >= 1 else '⚠️ fail'} ({p4_count} of {len(k_frames)})"
    )
    lines.append("")
    lines.append("## Per-frame summary\n")
    lines.append(
        "| frame | cat | K-signal | mean σ_b | mean cbd-rate | mean steps C1→C5 | acc |\n"
        "|-------|-----|----------|----------|---------------|------------------|-----|"
    )
    for fid in frames:
        pf = per_frame[fid]
        bc = pf["by_condition"]
        steps_traj = [bc.get(c, {}).get("mean_steps") for c in ("C1", "C2", "C3", "C4", "C5")]
        steps_str = "→".join((f"{s:.0f}" if s is not None else "-") for s in steps_traj)
        # accuracy across all conds for items with gold
        accs = [bc.get(c, {}).get("accuracy") for c in ("C1", "C2", "C3", "C4", "C5")]
        accs = [a for a in accs if a is not None]
        acc_str = f"{sum(accs)/len(accs):.2f}" if accs else "-"
        oe_mark = " *(open-ended; sig_b not applicable)*" if pf["is_open_ended"] else ""
        lines.append(
            f"| {fid} | {pf['category'][:14]} | **{pf['knightian_signal']:.2f}** | "
            f"{pf['mean_sig_b']:.2f} | {pf['mean_cbd_rate']:.2f} | "
            f"{steps_str} | {acc_str}{oe_mark} |"
        )

    # Per-condition cbd_rate trajectory for K items (the regime-shift signature)
    lines.append("")
    lines.append("## CBD-rate trajectory across conditions (K items only)\n")
    lines.append(
        "*Theorem 2.6.1's prediction: under meta-uncertainty, more reasoning "
        "may push the model out of cbd-recognition into substantive "
        "confabulation. Drop in cbd-rate from C1 to C5 = directional "
        "evidence for the regime shift.*\n"
    )
    lines.append(
        "| frame | C1 | C2 | C3 | C4 | C5 | trajectory |\n"
        "|-------|----|----|----|----|----|------------|"
    )
    for fid in k_frames:
        pf = per_frame[fid]
        cbds = [pf["by_condition"].get(c, {}).get("cbd_rate") for c in ("C1", "C2", "C3", "C4", "C5")]
        cbds_str = [(f"{r:.2f}" if r is not None else "-") for r in cbds]
        # Trajectory direction: C1 vs C5
        c1, c5 = cbds[0], cbds[4]
        if c1 is None or c5 is None:
            traj = "?"
        elif c5 < c1 - 0.1:
            traj = "↓ (drop)"
        elif c5 > c1 + 0.1:
            traj = "↑ (rise)"
        else:
            traj = "≈"
        lines.append(
            f"| {fid} | {cbds_str[0]} | {cbds_str[1]} | {cbds_str[2]} | "
            f"{cbds_str[3]} | {cbds_str[4]} | {traj} |"
        )

    lines.append("")
    lines.append(
        "*σ_b = categorical disagreement = 1 - max-frequency over the 3 "
        "replications per (item, condition) cell. Higher = more cross-seed "
        "variance, the §3.2 signature (b) of meta-uncertainty.*"
    )
    lines.append(
        "\n*Pilot is descriptive only per §4.9.3. The confirmatory H1 test "
        "is conducted on the full data set, not the pilot.*"
    )
    ANALYSIS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] {ANALYSIS_JSON.name} + {ANALYSIS_MD.name}")
    print(f"  P1: {'pass' if p1_pass else 'fail'} (K={summary['k_mean_signal']}, R+A={summary['ra_mean_signal']})")
    print(f"  P3: {'pass' if p3_pass else 'fail'} ({p3_frac:.1%})")
    print(f"  P4: {p4_count} of {len(k_frames)} K items show non-monotone sig_b trajectory")


if __name__ == "__main__":
    main()
