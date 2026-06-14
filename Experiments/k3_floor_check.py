"""
k3_floor_check.py — K3 training-data-sparsity check for FEH-79 K2 items.

Implements the §4.3 K3 operational criterion (Chapter 4): each novel-synthetic
item should reference entities, terms, or scenarios that are author-coined and
have no presence in any indexed corpus. Operational test: search the
distinctive coined phrasing on a public web index and verify the result floor
is below the §4.3 threshold (~100).

This script:
1. Loads K2 items from feh79_item_pool_v0.2.yaml.
2. Maps each K2 item to its distinctive coined phrase(s).
3. Queries DuckDuckGo's HTML interface for each phrase.
4. Parses returned HTML to count result blocks and detect "no results" pages.
5. Reports each item with its first-page result count and a sample of
   returned titles (so the user can review whether matches are spurious).

Notes:
- DuckDuckGo's HTML interface returns the first ~20 results on a single page;
  we cap our K3 verdict at "<= 20 first-page results" as a proxy for sparse
  indexing. This is a strict-er proxy than the §4.3 100-result Google
  criterion: any K2 item that fails the DDG check almost certainly fails the
  Google criterion, but the converse is weaker.
- For items where the DDG count is >= 5, the script writes the returned
  titles to the report so the user can audit whether matches are genuine
  contamination or coincidental real-word collisions (e.g., Karsk is a real
  place in Norway; the K2 question references a fictional planet of the
  same name).
"""

import json
import re
import sys
import time
from pathlib import Path

import truststore  # use OS trust store (Norton/Zscaler MITM root CA)
truststore.inject_into_ssl()

import requests
import yaml

# ============================================================================
# Configuration
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
POOL_PATH = REPO_ROOT / "feh79_item_pool_v0.2.yaml"
RESULTS_PATH = REPO_ROOT / "Experiments" / "k3_floor_results.json"
REPORT_PATH = REPO_ROOT / "Experiments" / "k3_floor_report.md"

# DDG threshold per the §4.3 K3 proxy used here.
# DDG first-page results below this are considered K3-pass.
K3_FIRST_PAGE_PASS_THRESHOLD = 5

# Search endpoint — Mojeek returns clean HTML SERP and tolerates moderate
# scripted access. Originally this script targeted DuckDuckGo HTML, but DDG
# now serves a 202 anti-bot page even with a browser-like User-Agent.
SEARCH_URL = "https://www.mojeek.com/search"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)

# Polite delay between requests (per Mojeek robots/ToS courtesy)
INTER_QUERY_DELAY = 8.0  # seconds; Mojeek 403s aggressively at faster rates

# ============================================================================
# Coined-entity registry per K2 item
# ============================================================================

# For each K2 item: (queries, must_contain_tokens). The queries are searched
# verbatim. The must_contain_tokens are checked against each returned title:
# only titles that contain at least one of the tokens (case-insensitive)
# count toward the K3 result tally. This guards against engines that ignore
# exact-quote semantics and return matches on individual words inside the
# query (e.g., "Marlovian" matching Christopher-Marlowe content unrelated to
# the fictional school of mathematics).
COINED_ENTITIES: dict[str, dict] = {
    "K2-001": {"queries": ['"Quogard" sintering', '"Quogard people"'],
               "must_contain": ["Quogard"]},
    "K2-002": {"queries": ['"Vasrenian" threadcounting', '"Vasrenian people"'],
               "must_contain": ["Vasrenian"]},
    "K2-003": {"queries": ['"anticipative formalism"'],
               "must_contain": ["anticipative formalism"]},
    "K2-004": {"queries": ['"zorinite"'],
               "must_contain": ["zorinite"]},
    "K2-005": {"queries": ['"Drelvian-Lindner"', '"Drelvian-Lindner interferometer"',
                            '"Institute for Photonic Standards" Bern'],
               "must_contain": ["Drelvian-Lindner", "Drelvian Lindner"]},
    "K2-006": {"queries": ['"drandology"', '"fnobel"'],
               "must_contain": ["drandology", "fnobel"]},
    # K2-007: hypothetical universe with two time dimensions — no coined entity
    "K2-008": {"queries": ['"Vermex" game', '"game of Vermex"'],
               "must_contain": ["Vermex"]},
    "K2-009": {"queries": ['"planet Karsk"', '"Karskian"'],
               "must_contain": ["planet Karsk", "Karskian"]},
    "K2-010": {"queries": ['"Lassic Mathematicians"', '"Federation of Lassic"'],
               "must_contain": ["Lassic"]},
    "K2-011": {"queries": ['"Yethra people"', '"Yethra"'],
               "must_contain": ["Yethra"]},
    "K2-012": {"queries": ['"tirstent"'],
               "must_contain": ["tirstent"]},
    "K2-013": {"queries": ['"Prendic" anticipatory autonomy', '"anticipatory autonomy" Prendic'],
               "must_contain": ["Prendic"]},
    "K2-014": {"queries": ['"Physarum consilium"'],
               "must_contain": ["Physarum consilium"]},
    "K2-015": {"queries": ['"Frobenian field theory"'],
               "must_contain": ["Frobenian"]},
    "K2-016": {"queries": ['"Volnari" trust quotient', '"Volnari civic"'],
               "must_contain": ["Volnari"]},
    "K2-017": {"queries": ['"Larcot" phantom debt', '"Larcot legal tradition"'],
               "must_contain": ["Larcot"]},
    "K2-018": {"queries": ['"tenebrith"'],
               "must_contain": ["tenebrith"]},
    "K2-019": {"queries": ['"Korlin people" memory', '"Korlin people"'],
               "must_contain": ["Korlin people"]},
    "K2-020": {"queries": ['"Iridian Council"'],
               "must_contain": ["Iridian Council"]},
}


# ============================================================================
# DDG search
# ============================================================================

# Mojeek SERP structure: each result is an <h2><a>title</a></h2> wrapped in
# class="ob" or similar. The robust pattern is to grab all h2 > a sequences.
RESULT_TITLE_RE = re.compile(
    r"<h2[^>]*>\s*<a[^>]*>(.*?)</a>\s*</h2>",
    re.IGNORECASE | re.DOTALL,
)


def _strip_tags(html_fragment: str) -> str:
    return re.sub(r"<[^>]+>", "", html_fragment).strip()


def search_engine(query: str, must_contain: list[str] | None = None) -> dict:
    """GET-search Mojeek. Returns total raw count + filtered (containing-coined) count + titles."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        r = requests.get(SEARCH_URL, params={"q": query}, headers=headers, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        return {
            "query": query, "error": f"{type(e).__name__}: {e}",
            "n_raw": 0, "n_filtered": 0, "titles": [],
        }

    html = r.text
    titles_html = RESULT_TITLE_RE.findall(html)
    titles = [_strip_tags(t) for t in titles_html]
    titles = [t for t in titles if t]
    n_raw = len(titles)

    if must_contain:
        tokens = [t.lower() for t in must_contain]
        kept = [t for t in titles if any(tok in t.lower() for tok in tokens)]
    else:
        kept = titles
    return {
        "query": query,
        "error": None,
        "n_raw": n_raw,
        "n_filtered": len(kept),
        "titles": kept[:10],          # only show titles that actually contain the coined entity
        "all_titles": titles[:10],    # plus the raw set, for audit
    }


# ============================================================================
# Pool loader (re-used from cross_model_check)
# ============================================================================


def load_pool() -> list:
    with open(POOL_PATH, encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))
    return [d for d in docs if d is not None]


def k3_verdict(per_query_results: list) -> str:
    """Combine multiple queries for one item into a single K3 verdict.

    Counts only `n_filtered` (titles containing the coined entity), not
    `n_raw` (any title returned by the engine). This avoids false positives
    from substring matches on real-word components of the query.
    """
    counts = [r.get("n_filtered", 0) for r in per_query_results if r.get("error") is None]
    if not counts:
        return "K3-error"
    max_n = max(counts)
    if max_n == 0:
        return "K3-pass-clean"
    if max_n < K3_FIRST_PAGE_PASS_THRESHOLD:
        return "K3-pass-marginal"
    return "K3-fail-contaminated"


# ============================================================================
# Driver
# ============================================================================


def main(frame_filter: set | None = None) -> None:
    pool = load_pool()
    pool_by_id = {d["frame_id"]: d for d in pool}

    items = sorted(COINED_ENTITIES.items())
    if frame_filter:
        items = [(fid, spec) for fid, spec in items if fid in frame_filter]

    results = []
    for i, (fid, spec) in enumerate(items, 1):
        item = pool_by_id.get(fid)
        if item is None:
            print(f"[{i:2d}/{len(items)}] {fid}: NOT IN POOL")
            continue
        queries = spec["queries"]
        must_contain = spec.get("must_contain", [])
        per_query = []
        for q in queries:
            print(f"  [{fid}] querying: {q}")
            res = search_engine(q, must_contain=must_contain)
            per_query.append(res)
            time.sleep(INTER_QUERY_DELAY)
        verdict = k3_verdict(per_query)
        max_filt = max(
            (r.get("n_filtered", 0) for r in per_query if r.get("error") is None),
            default=0,
        )
        max_raw = max(
            (r.get("n_raw", 0) for r in per_query if r.get("error") is None),
            default=0,
        )
        print(
            f"[{i:2d}/{len(items)}] {fid:8s} verdict={verdict:24s} "
            f"max_filtered={max_filt} (max_raw={max_raw})"
        )
        results.append({
            "frame_id": fid,
            "category": item["category"],
            "question_excerpt": item["question"].strip()[:120],
            "must_contain": must_contain,
            "queries": per_query,
            "max_filtered": max_filt,
            "max_raw": max_raw,
            "verdict": verdict,
        })

    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results)
    print(f"[done] wrote {RESULTS_PATH.name} and {REPORT_PATH.name}")


def write_report(results: list) -> None:
    from collections import Counter
    lines = ["# K3 Training-Data-Sparsity Pre-Screen — FEH-79 K2 items (pool v0.2)\n"]
    lines.append(
        "Search engine: Mojeek (https://www.mojeek.com/search), exact-quoted "
        "phrases. (DuckDuckGo HTML was attempted first but now serves an "
        "anti-bot page in response to scripted GET/POST.)\n"
    )
    lines.append(
        f"K3 verdict thresholds (first-page result count): "
        f"0 → K3-pass-clean, 1-{K3_FIRST_PAGE_PASS_THRESHOLD-1} → K3-pass-marginal, "
        f"≥{K3_FIRST_PAGE_PASS_THRESHOLD} → K3-fail-contaminated.\n"
    )
    lines.append(
        "Note: K3-fail-contaminated does not necessarily mean training-data "
        "contamination — many coined names collide with real proper nouns "
        "(e.g., 'Karsk' is a Norwegian place name, 'Korlin' is a surname). "
        "Each contaminated item should be audited by inspecting the returned "
        "titles below.\n"
    )

    n = len(results)
    counts = Counter(r["verdict"] for r in results)
    lines.append("**Verdict summary**:")
    for v in ("K3-pass-clean", "K3-pass-marginal", "K3-fail-contaminated", "K3-error"):
        lines.append(f"- **{v}**: {counts.get(v, 0)} / {n}")
    lines.append("")

    lines.append("\n## Per-item results\n")
    lines.append("| frame | verdict | max-filtered | max-raw | queries |")
    lines.append("|-------|---------|--------------|---------|---------|")
    for r in results:
        qs = "; ".join(q["query"] for q in r["queries"])
        lines.append(
            f"| {r['frame_id']} | {r['verdict']} | {r['max_filtered']} | "
            f"{r['max_raw']} | `{qs[:80]}` |"
        )

    flagged = [r for r in results if r["verdict"] == "K3-fail-contaminated"]
    if flagged:
        lines.append("\n## Detail — items flagged K3-fail-contaminated (manual audit needed)\n")
        for r in flagged:
            lines.append(f"\n### {r['frame_id']}")
            lines.append(f"\nQuestion: _{r['question_excerpt']}..._\n")
            for q in r["queries"]:
                lines.append(
                    f"\n**Query: `{q['query']}`** — {q.get('n_filtered', 0)} "
                    f"hits containing coined entity (of {q.get('n_raw', 0)} raw results)"
                )
                if q.get("titles"):
                    lines.append("\n_Titles containing coined entity:_")
                    for t in q["titles"]:
                        lines.append(f"  - {t[:120]}")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    arg_filter: set | None = None
    for arg in sys.argv[1:]:
        if arg.startswith("--frames="):
            arg_filter = {f.strip() for f in arg.split("=", 1)[1].split(",")}
    main(frame_filter=arg_filter)
