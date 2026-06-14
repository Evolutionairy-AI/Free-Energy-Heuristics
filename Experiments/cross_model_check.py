"""
cross_model_check.py — K2 cross-model disagreement validation for FEH-79.

Implements Chapter 4 §4.3 (K2) operational criterion: items eliciting
unanimous high-confidence agreement across frontier LLMs share a prior and
are candidates for replacement (they are not Knightian for the LLM
population).

For each categorical Knightian frame in feh79_item_pool_v0.1.yaml:
  1. Pose the question to Claude, GPT-4, Gemini, and Mistral under the C1
     (direct-answer, no CoT) prompt of Chapter 4 §4.6.
  2. Extract the 'Final answer: X' label.
  3. Tag the item with the modal answer and the unique-answer count.
  4. Items with unique-answer count == 1 across 4 models are flagged.

Outputs:
  cross_model_results.json — full per-call responses (verbatim).
  cross_model_report.md     — summary table per category.

API keys are loaded from Experiments/API_KEYS/ at runtime and never
printed.
"""

import json
import random
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import truststore  # use the OS trust store (handles Norton/Zscaler MITM CAs)
truststore.inject_into_ssl()

import requests
import yaml

# ============================================================================
# Configuration
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
KEYS_DIR = REPO_ROOT / "Experiments" / "API_KEYS"
POOL_PATH = REPO_ROOT / "feh79_item_pool_v0.3.yaml"
RESULTS_PATH = REPO_ROOT / "Experiments" / "cross_model_results.json"
REPORT_PATH = REPO_ROOT / "Experiments" / "cross_model_report.md"

# Model IDs — current frontier defaults per provider, May 2026.
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
OPENAI_MODEL = "gpt-4o-2024-11-20"
GEMINI_MODEL = "gemini-2.5-flash"
MISTRAL_MODEL = "mistral-large-latest"

# Generation knobs (match §4.6 baseline)
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_TOKENS = 2000  # generous; some models pad with reasoning despite the C1 instruction

# Request concurrency
MAX_WORKERS = 2              # outer (items in parallel) — kept low to respect Mistral RPS
INNER_MAX_WORKERS = 16       # inner (provider × seed cells per item)
REQUEST_TIMEOUT = 90         # seconds
RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_RETRIES = 6              # exponential backoff up to ~63s total

# Per-provider minimum interval between request starts, seconds. Set high
# enough to stay under each provider's free-tier RPM ceiling. Gemini 2.5 Flash
# free tier ≈ 15 RPM (4s spacing). Mistral free tier ≈ 30 RPM (2s spacing).
# Claude / GPT-4 paid tiers tolerate higher rates.
PROVIDER_MIN_INTERVAL_S = {
    "claude": 0.0,
    "gpt4": 0.0,
    "gemini": 4.2,
    "mistral": 2.1,
}
_PROVIDER_LOCKS = {p: threading.Lock() for p in PROVIDER_MIN_INTERVAL_S}
_PROVIDER_LAST_T = {p: 0.0 for p in PROVIDER_MIN_INTERVAL_S}


def _provider_throttle(provider: str) -> None:
    """Block until at least PROVIDER_MIN_INTERVAL_S has passed since the last
    request to this provider. Thread-safe via per-provider lock."""
    interval = PROVIDER_MIN_INTERVAL_S.get(provider, 0.0)
    if interval <= 0:
        return
    lock = _PROVIDER_LOCKS[provider]
    with lock:
        now = time.time()
        wait = (_PROVIDER_LAST_T[provider] + interval) - now
        if wait > 0:
            time.sleep(wait)
        _PROVIDER_LAST_T[provider] = time.time()

# CBD-equivalent normalized answers (per §4.3 v0.2 — these indicate
# shared meta-uncertainty recognition, not shared substantive prior).
CBD_TOKENS = {
    "cannot-be-determined", "cannot be determined", "cannot-be",
    "unknown", "uncertain", "unclear", "depends", "conditional",
    "cant-tell", "indeterminate", "cannot-determine", "cannot determine",
}

# §4.3 v0.2 K2 verdict threshold: modal answer is "unanimous-substantive"
# if it appears in ≥ this fraction of (model × seed) cells.
K2_UNANIMITY_THRESHOLD = 0.8


# ============================================================================
# Key loading
# ============================================================================


def _read_key_file(name: str) -> str:
    path = KEYS_DIR / name
    text = path.read_text(encoding="utf-8").strip()
    # Tolerate `KEY=...` or bare-key files.
    if "=" in text and "\n" not in text:
        text = text.split("=", 1)[1].strip()
    return text


def load_keys() -> dict:
    return {
        "anthropic": _read_key_file("Claude_Key.txt"),
        "openai": _read_key_file("OpenAI_RI.key.txt"),
        "google": _read_key_file("Free_energy_API_Key.txt"),
        "mistral": _read_key_file("Mistral_RI_Key.txt"),
    }


# ============================================================================
# Prompt construction
# ============================================================================


def build_user_prompt(item: dict) -> str:
    """C1 (direct-answer) prompt per §4.6 with answer-choice injection."""
    question = item["question"].strip()
    choices = item.get("answer_choices")
    parts = [question]
    if choices:
        parts.append(f"Choose from: {', '.join(str(c) for c in choices)}")
    parts.append(
        "Give a single-line answer of the form 'Final answer: <X>' with no "
        "other text."
    )
    return "\n\n".join(parts)


SYSTEM_PROMPT = (
    "You are a thoughtful assistant. Answer the user's question to the "
    "best of your ability. If asked to think step by step, structure "
    "your reasoning clearly. Finish with a single-line final answer of "
    "the form 'Final answer: <X>'."
)


# ============================================================================
# Provider adapters
# ============================================================================


def _post_with_retry(url, *, headers, json_body, timeout=REQUEST_TIMEOUT):
    """POST with exponential-backoff retry on 429 and 5xx."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(url, headers=headers, json=json_body, timeout=timeout)
            if r.status_code in RETRY_STATUSES:
                wait = (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(wait)
                last_exc = requests.HTTPError(
                    f"{r.status_code} retryable; attempt {attempt + 1}/{MAX_RETRIES}",
                    response=r,
                )
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last_exc = e
            if attempt < MAX_RETRIES - 1:
                wait = (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(wait)
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("retry loop exited with no result")


def call_claude(key: str, user: str, seed: int | None = None) -> str:
    # Anthropic does not accept a `seed` param; rely on temperature stochasticity.
    body = {
        "model": CLAUDE_MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user}],
    }
    r = _post_with_retry(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json_body=body,
    )
    return r.json()["content"][0]["text"]


def call_openai(key: str, user: str, seed: int | None = None) -> str:
    body = {
        "model": OPENAI_MODEL,
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
    r = _post_with_retry(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json_body=body,
    )
    return r.json()["choices"][0]["message"]["content"]


def call_gemini(key: str, user: str, seed: int | None = None) -> str:
    gen_config = {
        "temperature": TEMPERATURE,
        "topP": TOP_P,
        "maxOutputTokens": MAX_TOKENS,
    }
    if seed is not None:
        gen_config["seed"] = seed
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": gen_config,
    }
    r = _post_with_retry(
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={key}",
        headers={"Content-Type": "application/json"},
        json_body=body,
    )
    data = r.json()
    candidates = data.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def call_mistral(key: str, user: str, seed: int | None = None) -> str:
    body = {
        "model": MISTRAL_MODEL,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
    }
    if seed is not None:
        body["random_seed"] = seed
    r = _post_with_retry(
        "https://api.mistral.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json_body=body,
    )
    return r.json()["choices"][0]["message"]["content"]


PROVIDERS = {
    "claude": call_claude,
    "gpt4": call_openai,
    "gemini": call_gemini,
    "mistral": call_mistral,
}


KEY_FOR_PROVIDER = {
    "claude": "anthropic",
    "gpt4": "openai",
    "gemini": "google",
    "mistral": "mistral",
}


# ============================================================================
# Answer extraction & normalization
# ============================================================================

FINAL_ANSWER_RE = re.compile(
    r"final\s*answer\s*[:\-=]\s*(.+?)(?:\n|$)",
    re.IGNORECASE | re.DOTALL,
)


def extract_final_answer(text: str) -> str:
    m = FINAL_ANSWER_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(".").lower()
    # Fallback: last non-empty line.
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return (lines[-1] if lines else "").rstrip(".").lower()


def normalize_with_synonyms(answer: str, synonyms: dict | None) -> str:
    """Map extracted answer to canonical answer_choice via synonyms dict."""
    if not synonyms:
        return answer
    norm = answer.lower().strip().rstrip(".,!? ").strip("'\"")
    for canonical, syns in synonyms.items():
        if norm == str(canonical).lower():
            return str(canonical).lower()
        for s in syns or []:
            if norm == str(s).lower():
                return str(canonical).lower()
    # No exact synonym match — try substring containment (token-in-answer).
    for canonical, syns in synonyms.items():
        for token in [canonical] + list(syns or []):
            if str(token).lower() in norm:
                return str(canonical).lower()
    # Reverse: answer is a prefix of a canonical/synonym (handles models that
    # output truncated forms like "cannot-be" for "cannot-be-determined").
    if len(norm) >= 4:
        for canonical, syns in synonyms.items():
            for token in [canonical] + list(syns or []):
                t = str(token).lower()
                if t.startswith(norm) or norm.startswith(t):
                    return str(canonical).lower()
    return norm


# ============================================================================
# Per-item validation
# ============================================================================


_KEY_LEAK_RE = re.compile(r"key=[A-Za-z0-9_\-]+")


def _scrub(msg: str) -> str:
    """Redact any leaked URL-embedded API keys from error messages."""
    return _KEY_LEAK_RE.sub("key=<REDACTED>", msg)


def query_one(provider: str, key: str, item: dict, seed: int | None = None) -> dict:
    """Single (provider, item, seed) call. Returns dict with raw and normalized."""
    user = build_user_prompt(item)
    fn = PROVIDERS[provider]
    _provider_throttle(provider)
    t0 = time.time()
    try:
        raw = fn(key, user, seed=seed)
        err = None
    except Exception as e:
        raw = ""
        err = _scrub(f"{type(e).__name__}: {e}")
    dt = time.time() - t0
    extracted = extract_final_answer(raw) if raw else ""
    normalized = normalize_with_synonyms(extracted, item.get("synonyms"))
    return {
        "provider": provider,
        "seed": seed,
        "raw": raw,
        "extracted": extracted,
        "normalized": normalized,
        "latency_s": round(dt, 2),
        "error": err,
    }


def k2_verdict(cells: list, threshold: float = K2_UNANIMITY_THRESHOLD) -> dict:
    """Apply §4.3 v0.2 K2 criterion across cells."""
    valid = [c for c in cells if not c.get("error") and c.get("normalized")]
    n_total = len(valid)
    if n_total == 0:
        return {
            "verdict": "K2-error",
            "n_total": 0, "n_cbd": 0, "n_substantive": 0,
            "modal_substantive": None, "modal_substantive_count": 0,
            "modal_substantive_frac": 0.0,
            "unique_substantive": [],
        }

    cbd_count = sum(1 for c in valid if c["normalized"] in CBD_TOKENS)
    substantive_norms = [c["normalized"] for c in valid if c["normalized"] not in CBD_TOKENS]
    sub_counter = Counter(substantive_norms)
    n_substantive = len(substantive_norms)
    cbd_frac = cbd_count / n_total

    if sub_counter:
        modal_sub, modal_sub_count = sub_counter.most_common(1)[0]
        modal_sub_frac = modal_sub_count / n_total
    else:
        modal_sub, modal_sub_count, modal_sub_frac = None, 0, 0.0

    if cbd_frac >= threshold:
        verdict = "K2-pass-cbd"
    elif modal_sub_frac >= threshold:
        verdict = "K2-fail-substantive"
    else:
        verdict = "K2-pass-disagreement"

    return {
        "verdict": verdict,
        "n_total": n_total,
        "n_cbd": cbd_count,
        "n_substantive": n_substantive,
        "cbd_frac": round(cbd_frac, 3),
        "modal_substantive": modal_sub,
        "modal_substantive_count": modal_sub_count,
        "modal_substantive_frac": round(modal_sub_frac, 3),
        "unique_substantive": sorted(sub_counter.keys()),
        "substantive_distribution": dict(sub_counter),
    }


def validate_item(item: dict, keys: dict, n_seeds: int = 1, providers: list | None = None) -> dict:
    """Run selected providers × n_seeds in parallel for one item."""
    if providers is None:
        providers = list(PROVIDERS.keys())
    cells: list = []
    with ThreadPoolExecutor(max_workers=INNER_MAX_WORKERS) as pool:
        futs = {}
        for prov in providers:
            for s in range(1, n_seeds + 1):
                fut = pool.submit(query_one, prov, keys[KEY_FOR_PROVIDER[prov]], item, s)
                futs[fut] = (prov, s)
        for fut in as_completed(futs):
            prov, seed = futs[fut]
            try:
                cells.append(fut.result())
            except Exception as e:
                cells.append({
                    "provider": prov, "seed": seed,
                    "raw": "", "extracted": "", "normalized": "",
                    "error": _scrub(f"{type(e).__name__}: {e}"),
                })

    cells.sort(key=lambda c: (c.get("provider", ""), c.get("seed", 0)))
    verdict_info = k2_verdict(cells)
    # Per-provider modal (for the report's table view).
    by_provider: dict = {}
    for c in cells:
        by_provider.setdefault(c["provider"], []).append(c.get("normalized", ""))
    per_provider_modal = {
        p: (max(set(ns), key=ns.count) if ns else "")
        for p, ns in by_provider.items()
    }

    return {
        "frame_id": item["frame_id"],
        "category": item["category"],
        "question": item["question"].strip(),
        "answer_choices": item.get("answer_choices"),
        "n_seeds": n_seeds,
        "cells": cells,
        "per_provider_modal": per_provider_modal,
        **verdict_info,
    }


# ============================================================================
# Driver
# ============================================================================


def is_categorical_knightian(item: dict) -> bool:
    """Restrict the pre-screen to categorical Knightian items (auto-string-match)."""
    knightian_cats = {
        "non-recurrent-forecasting",
        "novel-synthetic",
        "open-ended-dilemma",
        "strategic-uncertainty",
    }
    if item["category"] not in knightian_cats:
        return False
    if item.get("answer_format") not in ("categorical", "numeric"):
        return False
    if item.get("grading_method") != "auto-string-match":
        return False
    return True


def load_pool() -> list:
    with open(POOL_PATH, encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))
    return [d for d in docs if d is not None]


VERDICT_MARK = {
    "K2-pass-cbd":          "✅ PASS-cbd",
    "K2-pass-disagreement": "✅ PASS-disag",
    "K2-fail-substantive":  "⚠️ FAIL-sub",
    "K2-error":             "❌ ERROR",
}


def write_report(results: list, path: Path, n_seeds: int) -> None:
    lines = [f"# Multi-Seed K2 Cross-Model Pre-Screen — FEH-79 ({POOL_PATH.name})\n"]
    lines.append(
        f"Models: Claude ({CLAUDE_MODEL}), GPT-4 ({OPENAI_MODEL}), "
        f"Gemini ({GEMINI_MODEL}), Mistral ({MISTRAL_MODEL}).\n"
    )
    lines.append(
        f"Seeds per (model, item) cell: **{n_seeds}**. "
        f"Cells per item: {4 * n_seeds}. "
        f"Verdict threshold (§4.3 v0.2): modal answer ≥ {int(K2_UNANIMITY_THRESHOLD*100)}% of cells.\n"
    )
    n = len(results)
    if n == 0:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    verdict_counts = Counter(r["verdict"] for r in results)
    lines.append("**Verdict summary**:")
    for v in ("K2-pass-disagreement", "K2-pass-cbd", "K2-fail-substantive", "K2-error"):
        lines.append(f"- {VERDICT_MARK.get(v, v)}: **{verdict_counts.get(v, 0)}** / {n}")
    lines.append("")

    by_cat: dict = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)

    for cat in sorted(by_cat):
        items = by_cat[cat]
        cat_failed = sum(1 for r in items if r["verdict"] == "K2-fail-substantive")
        lines.append(f"\n## {cat} — {cat_failed} failing of {len(items)}\n")
        lines.append(
            "| frame | verdict | modal-sub (frac) | cbd-frac | claude | gpt4 | gemini | mistral |\n"
            "|-------|---------|------------------|----------|--------|------|--------|---------|"
        )
        for r in items:
            pm = r.get("per_provider_modal", {})
            ms = r.get("modal_substantive") or "-"
            lines.append(
                "| {fid} | {v} | `{ms}` ({mf:.0%}) | {cf:.0%} | `{c}` | `{g}` | `{ge}` | `{mi}` |".format(
                    fid=r["frame_id"],
                    v=VERDICT_MARK.get(r["verdict"], r["verdict"]),
                    ms=str(ms)[:18],
                    mf=r.get("modal_substantive_frac", 0.0),
                    cf=r.get("cbd_frac", 0.0),
                    c=str(pm.get("claude", ""))[:18],
                    g=str(pm.get("gpt4", ""))[:18],
                    ge=str(pm.get("gemini", ""))[:18],
                    mi=str(pm.get("mistral", ""))[:18],
                )
            )

    # Detailed distribution for failing items
    failing = [r for r in results if r["verdict"] == "K2-fail-substantive"]
    if failing:
        lines.append("\n## Detail — failing items (substantive answer distribution across all cells)\n")
        for r in failing:
            dist = r.get("substantive_distribution", {})
            lines.append(
                f"- **{r['frame_id']}**: cbd={r['n_cbd']}/{r['n_total']}, "
                f"substantive distribution = {dist}"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(
    limit: int | None = None,
    frame_filter: set | None = None,
    n_seeds: int = 1,
    providers: list | None = None,
) -> None:
    if providers is None:
        providers = list(PROVIDERS.keys())
    keys = load_keys()
    pool = load_pool()
    pre_screen = [it for it in pool if is_categorical_knightian(it)]
    if frame_filter:
        pre_screen = [it for it in pre_screen if it["frame_id"] in frame_filter]
    if limit is not None:
        pre_screen = pre_screen[:limit]
    print(
        f"[start] {len(pre_screen)} items × {len(providers)} providers ({','.join(providers)}) × {n_seeds} seeds "
        f"= {len(pre_screen) * len(providers) * n_seeds} cells"
    )

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as outer:
        futs = {
            outer.submit(validate_item, it, keys, n_seeds, providers): it["frame_id"]
            for it in pre_screen
        }
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            ms = r.get("modal_substantive") or "-"
            print(
                f"[{i:3d}/{len(pre_screen)}] {r['frame_id']:8s} "
                f"verdict={r['verdict']:24s} modal-sub={str(ms)[:18]:18s} "
                f"({r.get('modal_substantive_frac', 0):.0%} of {r['n_total']} cells)"
            )

    results.sort(key=lambda r: r["frame_id"])
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results, REPORT_PATH, n_seeds)
    print(f"[done] wrote {RESULTS_PATH.name} and {REPORT_PATH.name}")


if __name__ == "__main__":
    arg_limit = None
    arg_filter: set | None = None
    arg_seeds = 1
    arg_providers: list | None = None
    for arg in sys.argv[1:]:
        if arg == "--smoke":
            arg_limit = 1
        elif arg.startswith("--limit="):
            arg_limit = int(arg.split("=", 1)[1])
        elif arg.startswith("--frames="):
            arg_filter = {f.strip() for f in arg.split("=", 1)[1].split(",")}
        elif arg.startswith("--seeds="):
            arg_seeds = int(arg.split("=", 1)[1])
        elif arg.startswith("--providers="):
            arg_providers = [p.strip() for p in arg.split("=", 1)[1].split(",") if p.strip()]
            unknown = [p for p in arg_providers if p not in PROVIDERS]
            if unknown:
                raise SystemExit(f"Unknown provider(s): {unknown}. Valid: {list(PROVIDERS)}")
    main(limit=arg_limit, frame_filter=arg_filter, n_seeds=arg_seeds, providers=arg_providers)
