# Independent OEIS pilot diagnostics

This observer measures the existing search → verify → train approach. It adds no
reward, branch penalty, admission rule, replay weighting, training input or gate.
It is never imported by production roles. Its own code, frozen sample, existing
CPU image, original inputs and observed runtime identities are recorded separately.

This separation is intentional: files in this directory are independently mounted
analysis tools, outside the existing production qualification closure. No closure
exclusion was added or changed. The original image and production source remain
untouched. The observer cannot grant or extend HIP qualification, and it does not
introduce another HIP rehearsal requirement. Historical qualification remains
valid only for its original scope. Do not move learner/scorer/runtime changes here
to avoid qualification.

## Measurements

For **every admitted program** in the original yield audit:

- Concrete program tokens and parameterised training-target tokens separately.
- Literal count, decimal digits, aggregate magnitude bits and largest literal.
- AST nodes, branches and equality comparisons.
- Direct `$index == integer` tests (either operand order) and distinct indices.
  This is a syntactic indicator: legitimate base cases are counted; disguised
  lookup schemes can be missed. It is never a memorisation verdict or a penalty.

For a **prospectively frozen sample** of at most 32 distinct training groups:

- One deterministically selected member per group; no model scores used.
- Original 100-term checks and the next 100 exact indexed terms checked separately
  by the reference and actual native executors.
- Dynamic branch counts, mismatch count/first index, incomplete execution and
  executor disagreements. Exact truth values are not exported in the report.
- Missing extra terms and absence of an admitted program stay in the denominator;
  no substitutions after search. Terms 101–200 mean ordinal positions, preserving
  each sequence's actual source offset.

The extra range is training-pool diagnostic feedback, not an untouched final
assessment. Do not use it to select candidates during the baseline pilot. Any
future study using these outcomes adaptively must declare that exposure. No
counterexample automatically mutates an archive, checkpoint or admission.

## Existing-image CPU checks

After the corpus image is available, this optional check uses it directly. It
builds no image and runs no GPU work:

```bash
python3 -S tools/oeis_diagnostics/run.py --engine podman \
  --image oeis-learn:corpus \
  self-test --output run-state/discovery/013-oeis-observer-check-001
```

The host requires only the standard library. The container runs the observer's
unit/fixture tests plus actual existing native/reference admission regressions:
a 20-term literal lookup is rejected, a legitimate conditional is admitted, and
the archive prefers the shortest verified representation. Unit tests use an
explicit reference-backed transport double; they are not native evidence. The
real native check is mandatory inside this command and existing Docker CI.

## Freeze before the OEIS pilot

The corpus alone is insufficient: first construct and validate the actual OEIS
cycle plan and its **training-only** pool. This is a later research stage; no
placeholder pool or nonexistent blueprint is generated here. Pool IDs must retain
the canonical `OEIS:Axxxxxx` source identity. The cycle output must not yet exist.

The following variables stand for those future **actual workspace-relative
paths**, not files that already exist in the current preparation session:

```bash
python3 -S tools/oeis_diagnostics/run.py --engine podman \
  --image oeis-learn:corpus freeze \
  --cycle-plan "$CYCLE_PLAN" --cycle-output "$CYCLE_OUTPUT" \
  --corpus data/oeis-corpus-001 \
  --output run-state/discovery/013-oeis-observer-plan-001
```

Save the immutable `plan.json` and `private/extra.json`. The plan records the
original pool/cycle, corpus snapshot, raw b-file hashes, sample, image and tool
identity. Preparation hashes at most 1 GiB of source-read work (three passes
reserved), at most 256 MiB per b-file, plus a streaming check of the corpus
inventory (up to 64 GiB), within a 600-second planning allowance.
The existing corpus is accessed read-only; no exposure index is rebuilt. Short
sources are `UNAVAILABLE`; source corruption/resource deficits block publication
rather than trigger replacement sampling.

## Observe the completed yield audit

After the original cycle publishes `host/yield-audit.json`, run:

```bash
python3 -S tools/oeis_diagnostics/run.py --engine podman audit \
  --plan run-state/discovery/013-oeis-observer-plan-001/plan.json \
  --output run-state/discovery/013-oeis-observer-audit-001
```

The audit uses the immutable image recorded during freeze. It reconciles every
accepted identity with original `phase-*-admit-training*/role/admission.json`
files, validates membership/value/program identities, and mounts only its tool
and copied diagnostic inputs/output. It never mounts the experiment directory,
weights, final-assessment data, engine socket or devices. Network is disabled;
CPU/memory are capped at two CPUs and 2 GiB. Audit work is bounded at 600 seconds
with four seconds per baseline or extra-range execution pair, plus bounded host
startup/cleanup. Original search/training costs are untouched; observer costs are
reported separately. A small native regression check runs before the audit.

This is deliberately a separately scheduled operation: add the freeze to the
upcoming OEIS preparation handoff and the audit after the yield publication. Do
not insert it into candidate selection or make diagnostic success a training
promotion condition. An observer failure leaves the original run unchanged.

Share `<output>/diagnostics.zip`. It contains the public plan/report, case results,
input references, source/image/runtime identities, exact command, bounded logs,
tracebacks and container exit/OOM/cleanup records. Private truth and native caches
stay local. Interrupted audits preserve completed case files. Use a fresh output
for another charged diagnostic attempt; keep prior costs/results. To reconstruct
a ZIP after an unhandled termination, without running an engine or any tests:

```bash
python3 -S tools/oeis_diagnostics/run.py --engine podman collect \
  --output run-state/discovery/013-oeis-observer-audit-001
```

`MATCH` means finite agreement only. `MISMATCH` beyond term 100 is a measured
negative result, not an operational failure. A baseline mismatch, executor
disagreement or incomplete execution requests investigation and produces a
nonzero diagnostic exit. None of these outcomes edits the experiment.

## Development verification

```bash
.venv-progressive/bin/python -m unittest discover \
  -s tools/oeis_diagnostics -p 'test_*.py' -v
```

At introduction, `collect_snapshot('.')` was identical before and after this
change: `sha256:6e72d46d7bd237a7c18c3f31d7ae563d88906025edd459582d47e8ba34d07d9d`.
This documents absence of a production change; it is not a new qualification.
