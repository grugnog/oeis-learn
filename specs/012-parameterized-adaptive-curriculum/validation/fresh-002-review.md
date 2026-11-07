# Review of the amended fresh structural comparison

Reviewed 2026-11-07 from `pac-selector-fresh-002/execution/report.zip`.
The archive is mechanically PASS and scientifically INCOMPLETE, with
`freeze_eligible: false`. D has a substantial descriptive advantage over B on
this certified cohort, under either selector. D completes only 64/68 primary
cases against the unchanged 65/68 floor. This result does not pass the technical
screen or establish a selector win, an overall win over generic templates, or
an across-training-seed effect.

## Evidence and verification

- Archive SHA-256: `6a1840d0c0aa9387359d153c91ee974097a168bd2f0bfc8cf073d99d0a2b8e5a`.
- Plan: `sha256:50fcbbcbc3eb77d8be6a0e582cc6b77841ccfc7893297d3ccbd884c4e634b39a`.
- Result: `sha256:2b16457abdb241852a8a6e963b904008d5ef563ee0175a47c9ac29be0d13324e`.
- Source: `sha256:9232aa2b69d02e461596388cda922393b86676a26eb50d941b8e3e8f8650132b`, matching the reviewed implementation's source snapshot.
- Protocol: `sha256:6603333f19a0dbf13926cb5e3c5d481cd1af01b7d814099cd36ba342e14937e3`, exactly the registered reservoir amendment.

Plan, result, admission, nested availability and both generation-result seals
verify. Admission/result bind the same plan. All 84 per-arm/category/control
and cross-arm/category/selector/control paired summaries recompute exactly
from the exported task outcomes. The eligibility calculation independently
reproduces INCOMPLETE: both generation records PASS and both selector trace
counts are zero, but D's primary completion is below 65 for both selectors.

The cohort is the fixed 128-task, 23-family allocation: 68 certified tasks in
15 families, 26 ambiguous in three, and 34 unresolved in five. The comparison
uses the original final B/D checkpoints and one training seed, with zero new
training tokens and assessment NOT_EXECUTED. B and D both use parameterized
sources and fitting; their distinction is fixed-stream versus bandit allocation,
not the presence of numeric placeholders. This run contains neither C nor a
new B+D training arm.

The guarded host path consumes new CPU/native and actual-HIP qualification.
This ZIP does not include qualification transcripts, so those were not
independently inspected here. The scoring producers are hash-referenced but
not included; the public projection omits their reasons and upstream timings.

## Admission and measured resource use

The fixed enlarged reservoir admitted all quotas without reference rejection
or another amendment. The four previously deficient families have 6, 8, 9 and
10 available candidates respectively, against quotas 2, 1, 2 and 2. Availability
has no deficits. The publication path checks native/reference agreement for
every selected task before model evaluation.

| Preparation measurement | Observed |
|---|---:|
| Fixed draws / indexed reference rows | 7,664 / 7,664 |
| Reference and exclusion time | 32.561 seconds |
| Peak process RSS before native admission | 1,393,975,296 bytes (1.298 GiB) |
| Preparation RSS ceiling | 15,032,385,536 bytes (14 GiB) |
| Reference rejections | 0 |

The exclusion counters are 6,251 finite-window, five group, five numeric-variant,
five source and 11 training-probe-window matches. These reasons overlap and
must not be summed as unique removals. Admission identity is
`sha256:1e2f8cbbfca51f1d0c4ea6dba3e34e9add293ab768754d4f8b862b03240dd7d6`;
availability identity is
`sha256:66685c134979d3eed0cb6242e623bf59f621755d944d44cd5b882849b0774e50`.

Additional selector observation totals are 1.264 seconds for B and 1.322 for D
across all seven controls. These measurements do not include full generation,
fitting or campaign wall time. The archive has no GPU utilization, CPU occupancy,
memory-bandwidth or thermal trace. It supports a small ranking-cost observation
and successful preparation memory gate, not a throughput or accelerator-
saturation claim for the Ryzen AI Max+ 395.

## Primary certified comparison

All success counts below use the fixed 68-task denominator; completion is
reported separately rather than dropping censored tasks.

| Arm / selector | Successes | Complete | Family-macro censoring bounds |
|---|---:|---:|---:|
| B / original | 37 | 66 | 59.39–62.73% |
| B / coverage | 37 | 66 | 59.39–62.73% |
| D / original | 50 | 64 | 74.40–79.45% |
| D / coverage | 53 | 64 | 77.58–82.63% |

The D-minus-B family-macro bounds are +11.67 to +20.06 percentage points with
the original selector and +14.85 to +23.23 with coverage. They remain positive
under worst-case assignments of the censored outcomes on this fixed cohort.
These are not confidence intervals: they contain no sampling or training-seed
uncertainty and cannot override the completion floor.

On the 64 jointly completed B/D tasks, original D has 17 unique successes
against B's three; coverage D has 20 against B's three. Shared successes are
33. These are cross-arm discordances, separate from within-D selector changes.

Coverage recovers three D tasks and regresses none among 64 jointly completed
primary cases. Its family-macro effect bounds are nevertheless -1.88 to +8.23
percentage points because four cases remain censored. B has no paired accuracy
change. Primary choices change on 30 B and 31 D tasks across the full 128-task
roster; changed choices are not equivalent to recovered successes.

### Where the gain occurs

The net D advantage concentrates in three certified families rather than
appearing uniformly across the family roster. Their parameterized sources
simplify as follows (family prefixes are only display abbreviations):

| Family prefix / semantic form | Tasks | B original successes | D original | D coverage | Template |
|---|---:|---:|---:|---:|---:|
| `589833f0edcd` / n(n+p0) | 6 | 0 | 5 | 6 | 0 |
| `988a472f4893` / (2p0+p1+1)n | 7 | 0 | 6 | 7 | 7 |
| `f6bb7c891347` / (p0+p1)p2 n | 6 | 0 | 5 | 6 | 6 |
| `d5849ec7e32b` / p0 p1 n² | 11 | 10 | 7 | 7 | 11 |

Each of the three improving families supplies exactly one selector recovery.
The recovered task IDs are:

- `sha256:ecf85314d426e0857af3f0c028f7880f7bd768e3e63a2e6288fda60d31feb899` (`589833f0edcd`).
- `sha256:46d76930b725b3781d0703998651a86f9f589737d228f00994e3b0efedc1d5d1` (`988a472f4893`).
- `sha256:3eb5744f18978f5e625d5387abd1e6145751114f0dadbe41aa6b2f024d245cd0` (`f6bb7c891347`).

The other eleven certified families have identical B/D success counts. Within
`d5849ec7e32b`, the ten jointly completed cases have six shared successes,
one D-only success and three B-only successes. B succeeds on the remaining
case, where D is censored. Thus the three-success count difference combines
observed discordances with one unknown D outcome.

## Controls and other categories

The template control is an actual unconditioned program-source baseline, not an
oracle. `fresh/template_baseline.py` declares eight fixed generic sources and
accepts no target values or metadata. They undergo the visible fitting and
selection path without neural decoding. It scores 55/68 with complete scoring,
and 80% certified family-macro success for both arms/selectors.

| Certified control | B original = coverage success / complete | D original | D coverage |
|---|---:|---:|---:|
| raw | 0 / 0 | 0 / 0 | 0 / 0 |
| rescue | 33 / 65 | 40 / 68 | 40 / 68 |
| equal_budget | 37 / 66 | 50 / 64 | 53 / 64 |
| shuffled | 20 / 68 | 44 / 65 | 47 / 65 |
| absent | 38 / 68 | 48 / 65 | 48 / 65 |
| template | 55 / 68 | 55 / 68 | 55 / 68 |
| bank | 40 / 68 | 46 / 66 | 46 / 66 |

D coverage has six completed successes the templates miss, all in `589833f0edcd`.
Templates have five completed successes D coverage misses, plus three where D
is censored. This is complementary coverage, not an overall learned-baseline
victory. Template-minus-D-coverage family-macro bounds are -2.63 to +2.42
percentage points. Do not use the 55 versus 53 counts as a family-macro ranking.

D's absent and shuffled controls are strong, so much of the gain can arise
from its proposal distribution and fitting rather than demonstrated dependence
on the correct sequence. Equal-budget original has only two more successes
than absent, with differing completion. Shuffled also has three coverage
recoveries and no regressions, on a different set of tasks. Raw is entirely
censored and is not a measured 0%
completed-case accuracy. All controls remain in the analysis.

| Category / control | B original = coverage | D original = coverage |
|---|---:|---:|
| Ambiguous (26) / raw | 0 / 0 | 0 / 0 |
| Ambiguous / rescue | 0 / 25 | 0 / 26 |
| Ambiguous / equal_budget | 0 / 25 | 0 / 24 |
| Ambiguous / shuffled | 0 / 25 | 0 / 23 |
| Ambiguous / absent | 0 / 26 | 0 / 23 |
| Ambiguous / template | 0 / 26 | 0 / 26 |
| Ambiguous / bank | 0 / 26 | 0 / 24 |
| Unresolved (34) / raw | 0 / 0 | 0 / 0 |
| Unresolved / rescue | 5 / 34 | 0 / 34 |
| Unresolved / equal_budget | 5 / 34 | 1 / 30 |
| Unresolved / shuffled | 0 / 34 | 0 / 28 |
| Unresolved / absent | 0 / 34 | 1 / 29 |
| Unresolved / template | 0 / 34 | 0 / 34 |
| Unresolved / bank | 0 / 34 | 0 / 34 |

Ambiguous cases supply no successes and stay outside the primary endpoint.
Unresolved primary outcomes favor B: one shared success, four B-only successes
on 30 jointly completed cases; D-minus-B censoring bounds are -16.00 to -2.36
percentage points. The certified advantage does not generalize to all categories.

## Completion deficit and next evidence

Both selectors have exactly the same incomplete primary cases. B's two cases
are a subset of D's four; none has a selected candidate. With zero incomplete
selector traces, these are upstream candidate-production/fitting incompleteness,
not four failed executions of selected 100-term programs or four coverage-ranking
timeouts. The exported projection cannot distinguish the upstream mechanisms.

| Task prefix | Family prefix / semantic form | Incomplete arms |
|---|---|---|
| `47ad52a901aa` | `04986e9882b8` / n²+p0-p1 | B, D |
| `adf9e3874f5c` | `04986e9882b8` / n²+p0-p1 | B, D |
| `84ddb6d9b178` | `174a9a2071d5` / 3n through n=3, then p0 n²-p1 | D |
| `e556dabdb491` | `d5849ec7e32b` / p0 p1 n² | D |

Inspect the original B/D scoring producers before proposing another experiment.
Their outcome records retain reasons, upstream generation/fitting evidence and
scoring time, which the public result projects away. The following command only
packages existing artifacts; it does not start any roles, rerun the plan or
change source/qualification. It verifies both producer hashes and sizes against
the report and creates a uniquely named ZIP of about 5 MiB before compression.
It includes no weights or private target files.

```bash
cd /home/owen/workspace/oies-learn && python3 -S - <<'PY'
import hashlib, json, time, zipfile
from pathlib import Path

workspace = Path.cwd().resolve()
root = workspace / 'run-state/generalization/pac-selector-fresh-002/execution'
result = json.loads((root / 'result.json').read_bytes())
paths = [root / 'report.zip']
for arm in ('B', 'D'):
    ref = result['scoring'][arm]
    path = (workspace / ref['path']).resolve(strict=True)
    path.relative_to(workspace)
    data = path.read_bytes()
    if len(data) != ref['bytes'] or 'sha256:' + hashlib.sha256(data).hexdigest() != ref['sha256']:
        raise SystemExit('Scoring producer hash/size mismatch: ' + str(path))
    paths.append(path)
output = root / ('review-details-' + str(time.time_ns()) + '.zip')
with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as archive:
    for path in paths:
        archive.write(path, str(path.relative_to(root)))
print(output)
PY
```

Upload the printed ZIP. These files should identify whether the remaining
deficit is dominated by decode, fitting or another upstream deadline and permit
a bounded CPU/service/queue timing analysis where measured. They cannot recover
hardware utilization that was never recorded.

Retain INCOMPLETE and the immutable plan/execution. Any engine repair must first
be validated on fixed candidates, with source qualification/probe compatibility
handled explicitly before a separately frozen future comparison; it cannot
retroactively complete this report. Do not lower the floor or select another
seed after observing these outcomes. D remains the more promising trained arm
for certified structure, but single-seed, concentrated gains and the template
baseline argue for reliability diagnosis and a prospective baseline-aware
confirmation before HPO. A learned-plus-template portfolio is a reasonable
future hypothesis because their failures differ, provided its rule and total
budget are frozen before new outcomes. This cohort is now observed evidence,
not a tuning or fresh confirmation set.
