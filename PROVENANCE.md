# Provenance Record

## Pre-registration
| field | value |
|---|---|
| Freeze commit | 033ee1ac2ca27e26bc865194b51ba146d2a3d206 |
| Freeze date | 2026-10-06 23:09:04 +0530 |
| PREREGISTRATION.md blob | ed695cb6f45b8e31de1a22ec9a54c34271ce64a3 |
| Verification | git rev-parse 033ee1a:PREREGISTRATION.md = git rev-parse HEAD:PREREGISTRATION.md |

## Registered run
| field | value |
|---|---|
| Run tag | phase2-20261010-174237 |
| Launched | 2026-10-10 17:42:37 IST |
| Compute | Modal, NVIDIA A10G, fp32/TF32 |
| Modal profile | riyann00b |
| Scale | 1200 clips (4 families x 250 + 200 controls), as frozen |
| Cost | 7,554 GPU-seconds (2.10 GPU-hours), 963 s wall |
| Protocol | frozen v1, unmodified |

## Modal account — status: UNRESOLVED for v2

| Field | Value |
|---|---|
| Profile name | riyann00b |
| Account owner | A friend, unnamed by mutual agreement |
| Relationship | Personal contact, not a lab account |
| Consent given | Verbal, one-time use for this run |
| Access terms | Single run only; not authorised for v2 |
| Payment method | Friend's payment method on file; author's accounts were payment-gated |
| Authorised for v2 | NO — flagged as unresolved |
| Reviewer note | Permission is not the same as approved provenance. An acceptable accountable compute route must be agreed before v2 runs. |

## Verdict status
Verdict was read by the automated pipeline during launch. Whether it
was subsequently opened by the author: [TODO: Faisal to fill —
opened or not opened, honestly]

## Gate-crossing deviation
The launch entrypoint (modal_app.py) invokes analyse_remote as its
final step. Consequently, merge, frozen analysis, and the Phase 3
gate executed as part of the launch. The intended
publish-before-analysis ordering did not hold. Recorded in
DEVIATIONS.md.

## Ordering claim
The pre-registration genuinely precedes the data: freeze commit
033ee1a is dated 2026-10-06, four days before the run on 2026-10-10.
The blob hash of PREREGISTRATION.md is identical at the freeze and
at HEAD. However, there is no independent notarised timestamp; the
ordering rests on the author's statement and the git commit dates.

## Git history integrity
- Git history was rewritten once (filter-branch to redact workspace
  names from STATUS.md): 13 commits re-hashed, author dates
  preserved.
- Freeze commit 033ee1a was untouched by the rewrite and is
  independently verifiable.
- No secrets were ever committed. The redaction concerned workspace
  names only.
- No history rewriting is planned.

## Artifacts on disk
- results.json — frozen analysis output
- REPORT.md — automated gate report
- survival curves — time-to-alarm plots
- Contents deliberately not reproduced in this document.
