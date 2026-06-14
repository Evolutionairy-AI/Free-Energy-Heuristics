"""Backwards-compat + correctness test for the LaTeX-robust numeric scorer.

Run: python test_latex_scoring.py   (exit 0 = all pass)

Guarantees:
  1. LaTeX-formatted correct answers now MATCH (the frontier confound fix).
  2. Plain-decimal scoring is UNCHANGED (strict superset — no regression).
  3. delatex() is a no-op on strings without LaTeX.
  4. Genuinely wrong answers still score wrong (no false positives introduced).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_analysis import gold_match, delatex, numeric_candidates

FAILS = []


def check(name, cond):
    if not cond:
        FAILS.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


# 1. LaTeX correct answers now match the numeric gold.
print("LaTeX answers now match (the fix):")
check("\\( \\frac{1}{4} \\) == 0.25", gold_match(r"\( \frac{1}{4} \)", 0.25))
check("\\dfrac{1}{6} == 0.16667", gold_match(r"\dfrac{1}{6}", 0.16667))
check("$\\frac{671}{1296}$ == 0.5177", gold_match(r"$\frac{671}{1296}$", 0.5177))
check("prose + \\(\\frac{1}{15}\\) == 0.06667",
      gold_match(r"the probability is \(\frac{1}{15}\)", 0.06667))

# 2. Plain-decimal scoring is unchanged (backwards compat).
print("Plain decimals unchanged (no regression):")
check("0.5 == 0.5", gold_match("0.5", 0.5))
check("'1/4' == 0.25", gold_match("1/4", 0.25))
check("'50%' == 0.5", gold_match("50%", 0.5))

# 2b. Precision-aware matching (#54): non-integer probability golds are stored as
# roundings of an exact real; a model may express it as a fraction, an
# over-precise decimal, or a coarser rounding. These must MATCH (which form the
# model uses correlates with CoT length -> a fixed tolerance mis-scores by
# condition).
print("Precision-aware: legit roundings/fractions of non-integer golds MATCH:")
check("'0.001982...' == 0.00198 (rounding of 5148/2598960)",
      gold_match("0.001982 (or about 1 in 505)", 0.00198))
check("'0.0019807923' == 0.00198 (over-precise decimal)",
      gold_match("0.0019807923", 0.00198))
check("'33/16660' == 0.00198 (exact fraction)", gold_match("33/16660", 0.00198))
check("'0.706' == 0.7063 (valid 3-sf rounding)", gold_match("0.706", 0.7063))
check("'671/1296' == 0.5177 (exact fraction)", gold_match("671/1296", 0.5177))

# 2c. Genuine value errors on non-integer golds must still be REJECTED — the 3rd
# significant figure differs, so these are computational mistakes, not roundings.
print("Precision-aware: genuine value errors on non-integer golds REJECTED:")
check("'0.703' != 0.7063 (3rd sig fig wrong)", not gold_match("0.703", 0.7063))
check("'0.7069' != 0.7063 (4th sig fig wrong)", not gold_match("0.7069", 0.7063))
check("vague '0.5' != 0.5177 (under-precise, 3-sf floor)",
      not gold_match("0.5", 0.5177))

# 2d. INTEGER golds (R-set arithmetic competence) require EXACT match — an
# estimate is a genuine failure, not a rounding.
print("Integer golds require exact match:")
check("'3901' == 3901", gold_match("3901", 3901))
check("'394' != 391 (arithmetic error)", not gold_match("394", 391))
check("'approximately 1860' != 1857 (estimate of exact)",
      not gold_match("approximately 1860", 1857))

# 3. delatex is a no-op without LaTeX.
print("delatex no-op without LaTeX:")
for s in ["0.5", "1/4", "0.001982 (or about 1 in 505)", "cannot-be-determined", ""]:
    check(f"delatex({s!r}) unchanged", delatex(s) == s)

# 4. Genuinely wrong answers still wrong (no false positives).
print("Wrong answers still wrong:")
check("\\frac{1}{3} != 0.25", not gold_match(r"\frac{1}{3}", 0.25))
check("0.9 != 0.5", not gold_match("0.9", 0.5))

print()
if FAILS:
    print(f"FAILED: {FAILS}")
    sys.exit(1)
print("ALL TESTS PASS")
