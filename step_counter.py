"""
step_counter.py — Two-pass step-counting pipeline for FEH-79 benchmark.

Implements Chapter 4 §4.8 step-counting:
  Pass 1: regex-based sentence segmentation
  Pass 2: LLM-judge step/not-step classification

A reasoning step is a sentence that contains at least one of:
  (S1) inferential connective (therefore, so, because, ...)
  (S2) intermediate computation (numeric operator + operand)
  (S3) intermediate task-relevant claim

Sentences that are pure meta-commentary do NOT count as steps.

The pipeline returns the realized step count for an LLM response, plus a
breakdown of which sentences were classified as STEP vs NOT-STEP.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Callable


# ===========================================================================
# Pass 1: Regex-based sentence segmentation
# ===========================================================================

# Common abbreviations that should NOT terminate a sentence.
ABBREVIATIONS = {
    "Dr", "Mr", "Mrs", "Ms", "Prof", "Sr", "Jr",
    "e.g", "i.e", "etc", "viz", "cf", "vs",
    "Inc", "Ltd", "Corp", "Co",
    "St", "Mt", "Ave", "Blvd",
}

# Sentence-terminating punctuation followed by whitespace and capital letter.
# We use a placeholder approach: replace abbreviation periods with a sentinel,
# segment, then restore.
ABBREV_SENTINEL = "\x00DOT\x00"


def _protect_abbreviations(text: str) -> str:
    """Replace periods inside known abbreviations with a sentinel."""
    for abbr in ABBREVIATIONS:
        pattern = re.compile(rf"\b{re.escape(abbr)}\.", flags=re.IGNORECASE)
        text = pattern.sub(abbr + ABBREV_SENTINEL, text)
    return text


def _restore_abbreviations(text: str) -> str:
    return text.replace(ABBREV_SENTINEL, ".")


# Math/code blocks should be treated as single sentences for segmentation.
MATH_BLOCK = re.compile(r"(\$\$.*?\$\$|\$.*?\$|`[^`]*`|```.*?```)", re.DOTALL)
MATH_SENTINEL_FMT = "\x00MATH{}\x00"


def _protect_math(text: str) -> tuple[str, list[str]]:
    """Replace math blocks with sentinels and return the originals."""
    blocks: list[str] = []
    def sub(m: re.Match) -> str:
        idx = len(blocks)
        blocks.append(m.group(0))
        return MATH_SENTINEL_FMT.format(idx)
    protected = MATH_BLOCK.sub(sub, text)
    return protected, blocks


def _restore_math(text: str, blocks: list[str]) -> str:
    for i, b in enumerate(blocks):
        text = text.replace(MATH_SENTINEL_FMT.format(i), b)
    return text


SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\d])"
)


def segment_sentences(response_text: str) -> list[str]:
    """Segment response into sentences.

    Handles common abbreviations, LaTeX math blocks, and code blocks.
    """
    body = _strip_final_answer(response_text)
    body = _protect_abbreviations(body)
    protected, math_blocks = _protect_math(body)
    raw_sentences = SENTENCE_BOUNDARY.split(protected)
    sentences = [
        _restore_abbreviations(_restore_math(s.strip(), math_blocks))
        for s in raw_sentences
        if s.strip()
    ]
    return sentences


def _strip_final_answer(response_text: str) -> str:
    """Remove the 'Final answer: X' tail from the response.

    Returns the body up to (but not including) the final-answer line.
    """
    final_answer_re = re.compile(
        r"\n\s*final\s*answer\s*[:\-=]\s*.+?\s*\Z",
        re.IGNORECASE | re.DOTALL,
    )
    match = final_answer_re.search(response_text)
    if match:
        return response_text[: match.start()].rstrip()
    return response_text.rstrip()


# ===========================================================================
# Pass 2: Heuristic step classification (fast baseline; LLM-judge is primary)
# ===========================================================================

INFERENTIAL_CONNECTIVES = [
    r"\btherefore\b", r"\bthus\b", r"\bso\b", r"\bhence\b",
    r"\bbecause\b", r"\bsince\b", r"\bgiven that\b",
    r"\bwhich means\b", r"\bimplies?\b", r"\bfollows that\b",
    r"\bconsequently\b", r"\bas a result\b",
]
INFER_RE = re.compile("|".join(INFERENTIAL_CONNECTIVES), re.IGNORECASE)

# Intermediate computation: any explicit numeric operation.
COMPUTATION_RE = re.compile(
    r"\d+\s*[+\-*/×÷=≈<>≤≥]\s*\d+"
    r"|=\s*\d"
    r"|\d+\s*(?:percent|%)"
)

# Pure meta-commentary patterns (NOT steps).
META_PATTERNS = [
    r"\blet me think\b",
    r"\blet's think\b",
    r"\blet me consider\b",
    r"\bI'?ll consider\b",
    r"\bnow I'?ll\b",
    r"\bnow let'?s\b",
    r"\bfirst,?\s*$",
    r"\bsecond,?\s*$",
    r"\bnext,?\s*$",
    r"\bokay\b\s*[\.,]?\s*$",
    r"\balright\b\s*[\.,]?\s*$",
]
META_RE = re.compile("|".join(META_PATTERNS), re.IGNORECASE)


def is_step_heuristic(sentence: str) -> bool:
    """Heuristic baseline classifier. Returns True if the sentence is a step.

    Used as a sanity check against the LLM-judge classifier (Pass 2 primary).
    """
    s = sentence.strip()
    if len(s) < 5:
        return False
    # Pure meta-commentary → not a step.
    stripped_lower = s.lower().strip(" .,!?")
    if META_RE.fullmatch(stripped_lower) or (
        META_RE.search(s) and len(s.split()) <= 6
    ):
        return False
    # Inferential connective → step.
    if INFER_RE.search(s):
        return True
    # Intermediate computation → step.
    if COMPUTATION_RE.search(s):
        return True
    # Substantive declarative sentence with task-relevant content.
    # Heuristic: at least 6 words and is not a question.
    if len(s.split()) >= 6 and not s.rstrip().endswith("?"):
        return True
    return False


# ===========================================================================
# Pass 2 primary: LLM-judge step classifier
# ===========================================================================

LLM_JUDGE_PROMPT = """\
Below is a sentence from a chain-of-thought reasoning trace. Classify it.

Definition: A reasoning STEP is a sentence that contains at least one of:
  (a) an inferential connective ("therefore", "so", "because", "since", "thus",
      "which means", "implies", "follows that", etc.)
  (b) an intermediate computation (a numeric operation with at least one
      operator and one operand explicitly stated)
  (c) an intermediate substantive claim that bears on the question being
      reasoned about

A NOT-STEP is a sentence that is only meta-commentary ("Let me think about
this", "Now I'll consider another angle", "First", "Next", "Okay") without
substantive content.

Sentence: {sentence}

Output exactly one token: STEP or NOT-STEP.
"""


@dataclass
class StepCountResult:
    """Output of step_count()."""
    n_steps: int
    sentences: list[str]
    classifications: list[str]   # one of STEP / NOT-STEP per sentence
    method: str                  # "heuristic" or "llm-judge"


LLMJudge = Callable[[str], str]
# Callable contract: takes the LLM_JUDGE_PROMPT-formatted text, returns
# "STEP" or "NOT-STEP".


def step_count(
    response_text: str,
    llm_judge: Optional[LLMJudge] = None,
) -> StepCountResult:
    """Count reasoning steps in an LLM response.

    Two passes:
      Pass 1: regex sentence segmentation.
      Pass 2: classify each sentence as STEP or NOT-STEP. If llm_judge is
              provided, use it (the §4.8.2 primary method). Otherwise fall
              back to the heuristic classifier (the §4.8.2 sanity-check
              baseline).
    """
    sentences = segment_sentences(response_text)
    classifications: list[str] = []
    if llm_judge is None:
        for sent in sentences:
            classifications.append("STEP" if is_step_heuristic(sent) else "NOT-STEP")
        method = "heuristic"
    else:
        for sent in sentences:
            prompt = LLM_JUDGE_PROMPT.format(sentence=sent)
            verdict = llm_judge(prompt).strip().upper()
            if verdict.startswith("STEP"):
                classifications.append("STEP")
            else:
                classifications.append("NOT-STEP")
        method = "llm-judge"
    n_steps = sum(1 for c in classifications if c == "STEP")
    return StepCountResult(
        n_steps=n_steps,
        sentences=sentences,
        classifications=classifications,
        method=method,
    )


# ===========================================================================
# Inter-rater reliability against human coding
# ===========================================================================

def cohen_kappa(rater1: list[str], rater2: list[str]) -> float:
    """Cohen's kappa for two raters on a binary classification (STEP/NOT-STEP).

    Returns kappa in [-1, 1] with 1 = perfect agreement, 0 = chance, <0 = worse
    than chance. Threshold for §4.8.3 validation: kappa >= 0.7.
    """
    if len(rater1) != len(rater2):
        raise ValueError("Rater label sequences must have equal length")
    n = len(rater1)
    if n == 0:
        return 0.0
    # Observed agreement
    p_o = sum(1 for a, b in zip(rater1, rater2) if a == b) / n
    # Expected agreement by chance
    p1_step = sum(1 for a in rater1 if a == "STEP") / n
    p2_step = sum(1 for b in rater2 if b == "STEP") / n
    p_e = p1_step * p2_step + (1 - p1_step) * (1 - p2_step)
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


# ===========================================================================
# Demo / smoke test
# ===========================================================================

def _demo() -> None:
    """Smoke test against a small hand-coded example."""
    sample_response = (
        "Let me think about this carefully.\n\n"
        "First, the patient is a 35-year-old single parent. "
        "Patient B is a 65-year-old physician. "
        "Therefore the utilitarian framework would favor patient A based on "
        "expected remaining life-years.\n\n"
        "However, patient B can potentially save additional lives in their "
        "professional role, so a different framework might favor B.\n\n"
        "Now I'll consider deontological perspectives. "
        "These would reject the use of social-role information entirely. "
        "Random allocation follows from treating both patients as equal moral "
        "agents.\n\n"
        "Final answer: random-allocation"
    )
    result = step_count(sample_response, llm_judge=None)
    print("=" * 70)
    print("step_counter.py demo (heuristic classifier)")
    print("=" * 70)
    print(f"sentences segmented: {len(result.sentences)}")
    print(f"step count: {result.n_steps}")
    print()
    for s, c in zip(result.sentences, result.classifications):
        marker = "[STEP]    " if c == "STEP" else "[not-step]"
        print(f"  {marker}  {s}")
    print()

    # Synthetic IRR check
    human1 = ["NOT-STEP", "STEP", "STEP", "STEP", "STEP",
              "NOT-STEP", "STEP", "STEP"]
    auto = result.classifications[: len(human1)]
    kappa = cohen_kappa(human1, auto)
    print(f"Synthetic kappa vs hand-coded reference: {kappa:+.3f}")
    print("(target for §4.8.3 validation: kappa >= 0.7)")


if __name__ == "__main__":
    _demo()
