# Multi-Seed K2 Cross-Model Pre-Screen — FEH-79 (feh79_item_pool_v0.3.yaml)

**Validation date**: 2026-05-14. **Items screened**: K1-005, K2-005, K4-003 (the 3 v0.3 replacement items). All 3 pass. The other 33 categorical Knightian items in the pool retain their pool-v0.2 multi-seed verdicts (26 PASS-disag + 7 PASS-cbd, see archived `cross_model_results.json` from 2026-05-13).

Models: Claude (claude-sonnet-4-5-20250929), GPT-4 (gpt-4o-2024-11-20), Gemini (gemini-2.5-flash), Mistral (mistral-large-latest).

Seeds per (model, item) cell: **5**. Cells per item: 20. Verdict threshold (§4.3 v0.2): modal answer ≥ 80% of cells.

**Verdict summary**:
- ✅ PASS-disag: **3** / 3
- ✅ PASS-cbd: **0** / 3
- ⚠️ FAIL-sub: **0** / 3
- ❌ ERROR: **0** / 3


## non-recurrent-forecasting — 0 failing of 1

| frame | verdict | modal-sub (frac) | cbd-frac | claude | gpt4 | gemini | mistral |
|-------|---------|------------------|----------|--------|------|--------|---------|
| K1-005 | ✅ PASS-disag | `us-eu` (64%) | 0% | `us-eu` | `no-treaty-by-2032` | `` | `` |

## novel-synthetic — 0 failing of 1

| frame | verdict | modal-sub (frac) | cbd-frac | claude | gpt4 | gemini | mistral |
|-------|---------|------------------|----------|--------|------|--------|---------|
| K2-005 | ✅ PASS-disag | `optical-rotation` (21%) | 79% | `cannot-be-determin` | `cannot-be-determin` | `` | `` |

## strategic-uncertainty — 0 failing of 1

| frame | verdict | modal-sub (frac) | cbd-frac | claude | gpt4 | gemini | mistral |
|-------|---------|------------------|----------|--------|------|--------|---------|
| K4-003 | ✅ PASS-disag | `commit` (33%) | 67% | `cannot-be-determin` | `commit` | `` | `cannot-be-determin` |
