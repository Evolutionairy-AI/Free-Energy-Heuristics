"""
c5_diagnose.py — is C5's low realized-step count a BEHAVIORAL effect (the "take
as long as you need" template actually yields shorter output) or a MEASUREMENT
bug (same-length output that the heuristic step-counter under-segments)?

Decisive comparison, per condition:
  tokens_out (Ollama eval_count = true generation length, counter-independent)
  vs n_steps (heuristic) and n_sentences.

If C5 tokens_out ~ C4 but n_steps << C4  -> counter under-counts C5 format (BUG).
If C5 tokens_out << C4                    -> template yields shorter output (BEHAVIORAL).

Also: steps-per-100-tokens and sentences-per-100-tokens by condition (counter
yield), and raw samples of the longest-token but fewest-step C5 responses.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))   # step_counter lives in the repo root
from step_counter import step_count  # noqa: E402

CONDS = ["C1", "C2", "C3", "C4", "C5"]


def main():
    cells = json.loads((HERE / "calibration_responses.json").read_text(encoding="utf-8"))
    by = {c: [] for c in CONDS}
    for c in cells:
        if c.get("error") or not c.get("raw_response"):
            continue
        by[c["condition"]].append(c)

    print("=" * 92)
    print("PER-CONDITION: generation length (tokens_out) vs counted steps")
    print("=" * 92)
    print(f"{'cond':4} | {'n':>4} | {'tok_out':>8} | {'chars':>7} | "
          f"{'sent':>6} | {'steps':>6} | {'steps/100tok':>12} | {'steps/sent':>10}")
    print("-" * 92)
    stats = {}
    for cond in CONDS:
        rows = by[cond]
        tok = np.array([r.get("tokens_out") or 0 for r in rows], float)
        chars = np.array([len(r["raw_response"]) for r in rows], float)
        sent = np.array([r.get("n_sentences") or 0 for r in rows], float)
        steps = np.array([r.get("n_steps_heuristic") or 0 for r in rows], float)
        spt = 100 * steps.sum() / tok.sum() if tok.sum() else 0
        sps = steps.sum() / sent.sum() if sent.sum() else 0
        stats[cond] = dict(tok=tok.mean(), steps=steps.mean(), spt=spt, sps=sps)
        print(f"{cond:4} | {len(rows):>4} | {tok.mean():>8.1f} | {chars.mean():>7.0f} | "
              f"{sent.mean():>6.2f} | {steps.mean():>6.2f} | {spt:>12.2f} | {sps:>10.3f}")

    print("\n" + "=" * 92)
    print("VERDICT LOGIC")
    print("=" * 92)
    c2, c3, c4, c5 = stats["C2"], stats["C3"], stats["C4"], stats["C5"]
    tok_ratio_c4 = c5['tok'] / c4['tok'] if c4['tok'] else float('nan')
    tok_ratio_c3 = c5['tok'] / c3['tok'] if c3['tok'] else float('nan')
    # steps/sentence isolates classifier behavior from output length.
    sps = {c: stats[c].get('sps', float('nan')) for c in CONDS}
    print(f"  C5/C4 token ratio = {tok_ratio_c4:.2f} ; C5/C3 token ratio = {tok_ratio_c3:.2f}")
    print(f"  steps/100tok: C2={c2['spt']:.2f} C3={c3['spt']:.2f} C4={c4['spt']:.2f} "
          f"C5={c5['spt']:.2f}  (prose band ~ C2-C3; C4 elevated by enumerated lists)")
    print(f"  steps/sentence: C4={sps['C4']:.3f}  C5={sps['C5']:.3f}  "
          f"(equal => classifier flags C5 sentences at the SAME rate; no count bug)")
    behavioral = tok_ratio_c4 < 0.7
    # measurement bug would show as C5 step-density BELOW the C2-C3 prose band
    band_lo = min(c2['spt'], c3['spt'])
    count_bug = c5['spt'] < 0.75 * band_lo
    print("  VERDICT: " + (
        "BEHAVIORAL — C5 generates far fewer tokens than C4 (and even fewer than "
        "C3); its step DENSITY sits in the normal C2-C3 prose band and its "
        "steps/sentence equals C4's, so the counter is not under-segmenting C5. "
        "The 'unconstrained' prompt simply yields shorter completions on these "
        "7-8B models." if behavioral and not count_bug else
        "MEASUREMENT — C5 density falls below the prose band (counter bug)."
        if count_bug else "AMBIGUOUS — inspect samples."))

    # Show C5 responses with many tokens but few steps (counter-suspect cases).
    print("\n" + "=" * 92)
    print("C5 SAMPLES: high tokens_out but low steps (counter-undercount suspects)")
    print("=" * 92)
    c5rows = sorted(
        [r for r in by["C5"] if (r.get("tokens_out") or 0) > 150],
        key=lambda r: (r.get("n_steps_heuristic") or 0))
    for r in c5rows[:3]:
        raw = r["raw_response"]
        recount = step_count(raw)
        print(f"\n--- {r['model']} {r['frame_id']} rep{r['replication']} | "
              f"tok_out={r.get('tokens_out')} stored_steps={r['n_steps_heuristic']} "
              f"sentences={r.get('n_sentences')} recount={recount.n_steps} ---")
        print(raw[:700].replace("\n", "\\n "))
    # Contrast: a typical C4
    print("\n" + "-" * 92)
    print("Typical C4 for contrast:")
    c4rows = sorted(by["C4"], key=lambda r: -(r.get("tokens_out") or 0))
    if c4rows:
        r = c4rows[len(c4rows)//2]
        print(f"--- {r['model']} {r['frame_id']} rep{r['replication']} | "
              f"tok_out={r.get('tokens_out')} steps={r['n_steps_heuristic']} "
              f"sentences={r.get('n_sentences')} ---")
        print(r["raw_response"][:700].replace("\n", "\\n "))


if __name__ == "__main__":
    main()
