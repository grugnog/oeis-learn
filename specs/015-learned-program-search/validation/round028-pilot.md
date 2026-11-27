# Eleven-hour matched generator pilot

**The one-seed pilot completed its 4M-token endpoint and retained D.** First-divergence training, which targets the first token that separates a correct program from a misleading one, produced fewer exact continuations than ordinary training at matched exposure. The registered decision was `RETAIN_REFERENCE`. The final assessment stayed sealed.

## Verified one-seed result

Both training arms reached the 0, 1M and 4M checkpoints. Each arm completed 2,126 optimizer updates on 4,000,210 useful tokens, with 210 tokens of final-update overshoot and zero discarded tokens. All three methods were evaluated on the same 128 development cases and 32 retention cases at the endpoint. A solve required an exact continuation of all 80 withheld terms.

| Method | Confirmed solves | Confirmed failures | Unresolved | Total |
| --- | ---: | ---: | ---: | ---: |
| Ordinary training | 26 | 106 | 28 | 160 |
| First-divergence training | 17 | 128 | 15 | 160 |
| Retained D | 27 | 101 | 32 | 160 |

The registered primary comparison weights families equally within composition and unseen structure, then weights the two strata equally. First divergence trailed ordinary training by **14.17 percentage points**. Assigning the unresolved outcomes against or in favor of first divergence gives bounds from **−22.83 to −6.50 points**. Against D the first-divergence result was −11.33 points (bounds −28.67 to −3.67). Ordinary training gained 2.83 points against D (bounds −14.50 to +11.50). Both learners solved 4 of the 32 retention cases, and D solved 6. The negative upper bound on the primary comparison settled the decision at one seed.

### Forecast and charged time

| Research work | Pre-run forecast | Charged time |
| --- | ---: | ---: |
| Search | 8 h 30 min 04 s | 1 h 04 min 49 s |
| Training | 1 h 51 min 58 s | 52 min 01 s |
| Independent scoring | 2 min 09 s | 6 s |
| **Total** | **10 h 24 min 12 s** | **1 h 56 min 56 s** |

The search forecast applied the slower of two previously timed cases in each stratum to the reserved cases. It also priced separate searches for the two zero-token checkpoints. The verified run shared their search and score because the inference inputs matched. That credit lowered the forecast to 8 h 40 min 34 s. The measured search and training roles then finished well within their remaining estimates. Times in the table are rounded independently. The archive reports cumulative charges across 18 roles and a GPU lease of 1 h 56 min 50 s. Re-exporting a completed run reuses its published work. [Original timing sample](round028-timing-results.md)

Controller `04e0fd9c30ce64d835c91fb06e5014799bceccee` verified the copied pilot against the pre-score qualification archived before research. Both the saved registration and pre-score commitment matched. The returned pilot ZIP has SHA-256 `4ad17a6f9e252de72025603a2e5ae7322f77c51b98260196eaec557ab22314fd`, and the verified report identity is `sha256:71c8c2f807cbc9101983a1c74194e04629f535666d878b06664274464b0198ab`. [Pre-score receipt](#verified-workstation-qualification)

## What the allocation covers

The verified historical audit accounted for all 128 round027 cases. On the 64 cases used for routing it found 51 witnessed proposal failures. The original workstation measured both learner profiles and D on eight previously exposed problems, reserved 576 cases and closed the eight-hour study with zero fresh pilot scores. The child binds that original workstation receipt and reuses its reservations. [Audit](round028-audit.md) · [Panel supply](round028-panel-results.md) · [Original timing and receipt](round028-timing-results.md)

The following estimates belong to the original timing collection. The copied pre-score qualification recomputed the forecast for the registered launch at 10 h 24 min 12 s.

| Work | Historical forecast |
| --- | ---: |
| One pilot seed, with one resident search process per arm and look | 10 h 21 min 37 s |
| Headroom in the eleven-hour research allocation | 38 min 23 s |
| Same pilot with a restart every eight cases | 11 h 09 min 28 s |
| Additional independent repeat if earned | 11 h 11 min 48 s |

The command checks the candidate CPU and HIP images, actual updates and checkpoint restore. It tests storage exhaustion on a disposable filesystem, runs recovery cases and exercises the resident search and independent scorer on previously exposed data. It prepares the seed-specific native training stream, checks it against protected cases and freezes its identity, weights, input order and evaluator settings before opening fresh scores. Preparation and qualification happen before the research allocation begins.

The two descriptive looks use the first reserved case in each development family and one attempt per case. The 4M endpoint compares both learners and D on the same 128 development cases and 32 retention cases, with eight attempts per portfolio. Search receives the first 20 terms. A separate CPU process checks the sealed choice on the complete 100-term record. If a case was interrupted during fitting, the result carries that uncertainty. The independent repeat, final assessment and 25M confirmation require separate authority.

## What comes back

The launcher prints one path. A completed `round028-pilot.zip` contains `pilot.json` and the verified parent package. The standalone reader checks the registered denominators, treatment and incumbent contrasts, missing-outcome bounds, retention rule, role costs and spending decision. A positive pilot can earn an independent repeat. The current allocation covers one seed. A diagnostic ZIP identifies the last phase, gives available spending and reports a numerical shortage when measured. Full logs, per-case choices, training data and model checkpoints stay on the workstation.

To verify a copied pilot and its original pre-score evidence, put `round028-pilot.zip` and the returned `round028-pilot-qualification.zip` in the same directory and run this block there. The reader writes `round028-pilot-verification.json`. [Verify the qualification on its own](pilot-resume.md#verify-a-copied-qualification)

```bash
(
  set -euo pipefail
  rev=04e0fd9c30ce64d835c91fb06e5014799bceccee
  report="$(pwd -P)/round028-pilot.zip"
  qualification="$(pwd -P)/round028-pilot-qualification.zip"
  reader=$(mktemp -d)
  trap 'rm -rf -- "$reader"' EXIT
  git -C "$reader" init --quiet
  git -C "$reader" fetch --quiet --depth=1 https://github.com/grugnog/oeis-learn.git "$rev"
  git -C "$reader" checkout --quiet --detach "$rev"
  python3 -S -B "$reader/scripts/foundation_run.py" --engine podman discovery round028 \
    --action verify --archive "$report" --qualification-archive "$qualification" > round028-pilot-verification.json
  python3 -c 'import json; r=json.load(open("round028-pilot-verification.json")); print(r["status"], r["next_action"])'
)
```

## Verified workstation qualification

The pinned reader independently verified the copied pre-score ZIP. The controller and registered worker were both `04e0fd9c30ce64d835c91fb06e5014799bceccee`, and the worker source digest matched a clean checkout of that commit. The embedded parent matched the previously verified round028 workstation ZIP byte for byte. [Original workstation receipt](round028-timing-results.md#workstation-receipt-and-public-reports)

| Measurement | Original workstation |
| --- | ---: |
| Cold search roster | 160 slots on eight exposed problems, first cold attempt |
| Search elapsed / supervisor wall | 6,140.289864 s / 6,142.073628 s |
| Cold checkpoint load and identification / candidate publication | 1.095830 s / 0.292683 s |
| Minimum case allowance / slowest elapsed | 60.000000 s / 60.133643 s |
| Slowest case margin | −0.133643 s |
| Independently scored selections / slots with no selection | 2 / 158 |
| Scoring elapsed / score publication | 0.203206 s / 0.014177 s |
| Scoring supervisor wall | 0.805648 s |
| Time remaining in search / scoring supervisor envelopes | 14,977.926372 s / 21,119.194352 s |
| Free storage at collection | 237,173,972,992 bytes and 119,533,038 inodes |

The 60-second deadline controls when another attempt may start. Under the registered drain rule, the clock also records the time to finish outstanding work. The slowest observed case closed 0.133643 s after its deadline. The search supervisor measured **1 h 42 min 22 s** for the full roster. The ZIP gives per-role timings and omits the collector’s total wall time. [Registered search rule](../contracts/experiment.md)

The verified forecast was **37,452.119219 s** out of **39,600 s**, leaving **35 min 48 s**. At collection the research ledger had zero charges and zero open reservations, and the scientific status was `PILOT_UNMEASURED`. Keep this ZIP for comparison with the later pilot archive.

| Evidence | SHA-256 |
| --- | --- |
| Returned qualification ZIP | `ef61b2e243c2bb95035baaa230b71093ef7ecb36433f577063c39a3d65350812` |
| Saved registration | `4325677a7b083bcc38cc541a3b0999fc328f39d898320457e7f6a72af7a54755` |
| Pre-score commitment | `0867bb3c56089449f7bfae95c0154a2ec041ccda6fad41ef6cddefc3a5797fc6` |

## Qualification fixtures

The installed CPU tests exercised a cold 160-slot roster, a slow admitted choice through both scoring engines, and recovery through the shell launcher with a saved worker distinct from the controller. A separate fixture checked the scorer on 160 exposed replays when the learner selected zero programs. These tests used a simulated native engine.

In a detached donor fixture without its private campaign, the qualification and pilot commands each returned `ORIGINAL_FROZEN_CAMPAIGN_MISSING` and zero GPU dispatches.

## Earlier release verification

The local release suite passed 234 tests. Two tests require the native qualification image. The pilot fixture ran the 0, 1M and 4M planner, the three-arm endpoint and a second invocation with no new research roles. A separate CPU test restored both models and performed real optimizer updates with the original discarded-token charge. Recovery cases covered a partially completed search and an interrupted independent scorer.

The pinned shell launcher was fetched and extracted in a detached checkout at the original donor commit. Metadata preflight, status and pilot status ran without acquiring a device. Two invocations with the private donor evidence absent each produced a diagnostic ZIP identifying the missing original campaign. An isolated Git checkout with standard-library Python verified the real historical workstation ZIP and a synthetic pilot archive built over that real parent. The latter exercised the reader with simulated outcomes.
