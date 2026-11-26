# Eleven-hour matched generator pilot

**The scientific endpoint awaits the original workstation.** The pilot tests whether training on the first token that distinguishes a correct program from a misleading one improves exact continuation. Its main comparison is against ordinary training from the same random start, on the same rows and positive tokens. The retained D model is the incumbent reference.

## What the allocation covers

The verified historical audit accounted for all 128 round027 cases. On the 64 cases used for routing it found 51 witnessed proposal failures. The original workstation measured both learner profiles and D on eight previously exposed problems, reserved 576 cases and closed the eight-hour study with zero fresh pilot scores. The child binds that original workstation receipt and reuses its reservations. [Audit](round028-audit.md) · [Panel supply](round028-panel-results.md) · [Original timing and receipt](round028-timing-results.md)

| Work | Forecast |
| --- | ---: |
| One pilot seed, with one resident search process per arm and look | 10 h 21 min 37 s |
| Headroom in the eleven-hour research allocation | 38 min 23 s |
| Same pilot with a restart every eight cases | 11 h 09 min 28 s |
| Additional independent repeat if earned | 11 h 11 min 48 s |

The command checks the candidate CPU and HIP images, actual updates and checkpoint restore. It tests storage exhaustion on a disposable filesystem, runs recovery cases and exercises the resident search and independent scorer on previously exposed data. It prepares the seed-specific native training stream, checks it against protected cases and freezes its identity, weights, input order and evaluator settings before opening fresh scores. Preparation and qualification happen before the research allocation begins.

The two descriptive looks use the first reserved case in each development family and one attempt per case. The 4M endpoint compares both learners and D on the same 128 development cases and 32 retention cases, with eight attempts per portfolio. Search receives the first 20 terms. A separate CPU process checks the sealed choice on the complete 100-term record. If a case was interrupted during fitting, the result carries that uncertainty. The independent repeat, final assessment and 25M confirmation require separate authority.

## What comes back

The launcher prints one path. A completed `round028-pilot.zip` contains `pilot.json` and the verified parent package. The standalone reader checks the registered denominators, treatment and incumbent contrasts, missing-outcome bounds, retention rule, role costs and spending decision. A positive pilot can earn an independent repeat. The current allocation covers one seed. A diagnostic ZIP identifies the last phase, gives available spending and reports a numerical shortage when measured. Full logs, per-case choices, training data and model checkpoints stay on the workstation.

The implementation pin is `cafceae40f385aa0137355a79f339f3b62e2069d`. To verify a copied pilot ZIP on a computer with Git and Python, run this from the directory containing the ZIP. The reader writes the full result to `round028-pilot-verification.json`.

```bash
(
  set -euo pipefail
  rev=cafceae40f385aa0137355a79f339f3b62e2069d
  report="$(pwd -P)/round028-pilot.zip"
  reader=$(mktemp -d)
  trap 'rm -rf -- "$reader"' EXIT
  git -C "$reader" init --quiet
  git -C "$reader" fetch --quiet --depth=1 https://github.com/grugnog/oeis-learn.git "$rev"
  git -C "$reader" checkout --quiet --detach "$rev"
  python3 -S -B "$reader/scripts/foundation_run.py" --engine podman discovery round028 \
    --action verify --archive "$report" > round028-pilot-verification.json
  python3 -c 'import json; r=json.load(open("round028-pilot-verification.json")); print(r["status"], r["next_action"])'
)
```

## Release verification

The local release suite passed 234 tests. Two tests require the native qualification image. The pilot fixture ran the 0, 1M and 4M planner, the three-arm endpoint and a second invocation with no new research roles. A separate CPU test restored both models and performed real optimizer updates with the original discarded-token charge. Recovery cases covered a partially completed search and an interrupted independent scorer.

The pinned shell launcher was fetched and extracted in a detached checkout at the original donor commit. Metadata preflight, status and pilot status ran without acquiring a device. Two invocations with the private donor evidence absent each produced a diagnostic ZIP identifying the missing original campaign. An isolated Git checkout with standard-library Python verified the real historical workstation ZIP and a synthetic pilot archive built over that real parent. The latter exercised the reader with simulated outcomes. Physical qualification, pilot training and the fresh 4M comparison await the workstation command.
