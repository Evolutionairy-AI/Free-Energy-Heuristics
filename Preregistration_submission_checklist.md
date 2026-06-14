# Pre-Registration Submission Checklist

Practical step-by-step guide for uploading `Preregistration_full_empirical_v0.1.docx` to the Open Science Framework or AsPredicted.

## Before submission

- [ ] Read through `Preregistration_full_empirical_v0.1.docx` end-to-end one more time. Pay attention to: (i) hypotheses (Section 2) — are they exactly what you want locked in?, (ii) inference criteria (Section 7) — could the falsification thresholds plausibly be tightened?, (iii) Appendix C unfinished items.
- [ ] Finalize Appendix C placeholders:
  - Theoretical pre-print arXiv DOI (will be filled once Chapters 2+3 are uploaded to arXiv as a working paper — this can be done in parallel with the pre-registration).
  - GitHub repository (create empty repo now, before pre-registration; commit MIT LICENSE and a README; record the URL).
  - OSF project URL (created automatically during submission below).
  - Item pool finalization status (Section 9 of pre-reg permits pilot-driven amendment, so this need not be 100% finalized at submission time — but the §4 benchmark design chapter should be drafted enough to give a reviewer confidence in the procedure).
  - Step-counting κ-validation: can be committed-to-procedure (specify the human-coded subsample protocol) without yet having run it; the κ ≥ 0.7 threshold is the pre-registered commitment.

## Submission flow — OSF Pre-registration (recommended)

1. Create an OSF account if you don't have one: https://osf.io/register.
2. Create a new OSF project: "Free Energy Heuristics — Full Empirical."
3. Upload the theoretical pre-print (FEH_TheoreticalFoundation_Draft_v0.7.docx) and Chapter 3 (FEH_Chapter3_Operationalization_v0.1.docx) to the project's "Files" tab.
4. Navigate to "Registrations" → "New Registration" → choose the **OSF Pre-registration** schema (NOT OSF Prereg-Direct, which is for already-completed studies).
5. Fill in the schema fields using the corresponding sections from `Preregistration_full_empirical_v0.1.docx`:
   - Study Information → §1
   - Hypotheses → §2
   - Design → §3
   - Sampling → §4
   - Variables → §5
   - Analysis Plan → §6
   - Other → §7-§11
6. Upload `Preregistration_full_empirical_v0.1.docx` as an attached document at the schema's optional file-upload field (it provides the long-form record that backs the schema fields).
7. Submit. OSF returns a registration DOI and a timestamped, immutable registration page.
8. Update Appendix C of the local pre-registration file with the OSF DOI; commit to your repo.

Typical time to complete: ~45-60 minutes if all Appendix C items are ready; ~90 minutes including any reading-through pauses.

## Submission flow — AsPredicted (lightweight alternative)

If you prefer a faster, less-comprehensive registry:

1. Visit https://aspredicted.org.
2. Create the registration using Appendix B (the AsPredicted-format summary) as the basis.
3. Identify co-authors / collaborators if any (otherwise just yourself).
4. Click "Create the registration." AsPredicted returns a private link.
5. Click "Approve & Finalize" to make the registration immutable.
6. The final registration has a permanent timestamp.

Typical time: ~15-20 minutes. AsPredicted is less suitable for a multi-model, multi-condition study but is acceptable if the OSF schema feels too heavyweight.

## After submission

- [ ] Add the OSF (or AsPredicted) URL/DOI to your manuscript draft as "Pre-registration: https://osf.io/{XXXX}".
- [ ] Add the URL to the GitHub repo README under a "Pre-registration" section.
- [ ] Save the timestamp screenshot to your project files as evidence (some venues require it).
- [ ] Begin §4 benchmark design.
- [ ] If §4 pilot reveals an issue requiring amendment: navigate to OSF → your registration → "File an amendment" (do NOT edit the original; amendments are timestamped and visible).

## Strategic note on positioning

For maximum credibility in the LLM-evaluation community, the OSF pre-registration is preferable because:
- It's more comprehensive (the hierarchical Bayes specification, the falsification criteria, and the deviations protocol are unusual for the LLM-evaluation literature and signal seriousness).
- It accepts long-form attachments (the full DOCX), which preserves the conceptual rationale alongside the technical specifications.
- It's the standard in psychology and cognitive science, which is where the theoretical positioning lives.

AsPredicted is fine as a backup or supplement but should not be the primary registration for a paper aiming for influential placement in the LLM-evaluation conversation.

## What this pre-registration accomplishes for the paper

Beyond procedural correctness, the pre-registration document itself serves three rhetorical functions for the eventual paper:

1. **It converts §3.5 from "we recommend pre-registering" into "we did."** This is the difference between a paper that aspires to high methodological standards and one that demonstrates them.
2. **It signals to reviewers that the headline claim is genuinely falsifiable.** The pre-specified falsification criteria (Section 7.2) preempt the most common reviewer objection to "X hurts performance" papers in the LLM literature, which is that the result depends on undisclosed analytic choices.
3. **It binds the framework to a single, sharp empirical commitment.** If the prediction fails on the pre-registered analysis, the framework needs revision — and the paper says so explicitly. This is the opposite of the typical "framework that explains everything" failure mode that haunts active-inference papers in particular.

The pre-registration is therefore a substantive contribution to the paper's positioning, not just procedural hygiene.
