# Implemented CLI and runtime boundary

New interfaces live in `cli/foundation.py`, registered by `cli/main.py` under
`oeis-learn foundation`. Run `foundation COMMAND --help` for the closed arguments.
Every interface supports `--json`; paths are not evaluated by a shell, unknown
flags/configuration fields reject, and diagnostic completion never implies
qualification. Exit codes distinguish invalid input (2), unavailable hardware
(3), failed gates (4), and resource/infrastructure interruption (5); a valid
recorded diagnostic/candidate rejection can return zero. Individual handlers
preserve their explicit status alongside the exit code.

| Interface | Required inputs and implemented boundary |
| --- | --- |
| `preflight --config FILE --output DIR` | Explicit preflight settings; actual requested device/image evidence. No CPU substitution for GPU readiness. |
| `conformance --profile FILE --output DIR` | Versioned profile, independent regression/differential execution; unavailable adapters cannot qualify. |
| `freeze-cohort --source DIR --config FILE --output DIR` | Explicit indexed local source/config; private truth, grouping and reserved membership. |
| `build-pool --config FILE --cohort DIR --output DIR` | Mechanically generated generic programs, exact admission and membership exclusion; no imported teachers. |
| `train --config FILE --pool DIR --run-dir DIR` | Frozen strict baseline; `--device cpu --diagnostic` explicitly unqualified; `--stop-after-update`, `--prepare-only`, `--diagnostic-small-host` retain actual settings. |
| `resume --run-dir DIR [--checkpoint MANIFEST]` | Restore newest compatible complete state; checkpoint is an assertion, no setting overrides. |
| `inspect --run-dir DIR` | Read-only validity/lifecycle/accounting/qualification report. |
| `evaluate --checkpoint MANIFEST --cohort DIR --split development --protocol FILE --output DIR` | Actual weights, visible-only generation and sealed candidates, evaluator-only private scoring. |
| `finalize --run-dir DIR --checkpoint MANIFEST --protocol FILE --cohort DIR --stopping-record FILE --output FILE` | Immutable decision lock; does not expose final truth. |
| `evaluate --finalization FILE --split final --output DIR` | Exactly the locked model/protocol/cohort; committed per-target seals reused after hidden exposure. |
| `qualify-repairs --output DIR` | Owning G7 solver/proof/optimizer and actual Rust/native suites. |
| `activate-qualified --config FILE --gate-report FILE --profiles DIR` | Separate `qualified/v1` capability contract; current G7, plus G8 where required. |
| `qualify-learning --output DIR [--hardware]` | G8 software, optional mandatory actual accelerator suite when requested. |
| `learning-train` / `learning-search` | Explicit config, G7/G8 reports, profiles, archive, output and device; CPU `--diagnostic` cannot qualify hardware. Train supports `--updates`, `--seed`, `--resume`; search needs checkpoint/prompt/portfolio. |
| `qualify-discovery --output DIR [--native]` | G9 logical/streaming/discovery/certificate/native acceptance. |
| `discover --archive DIR --discovery-gate-report FILE --output DIR [--holdouts FILE]` | Relations from verified program archive and original structural holdouts. |
| `check-certificate --certificate FILE --profile FILE --output DIR` | Fresh bounded scoped replay, independent of neural weights; no automatic WAT theorem. |
| `census --source FILE --output DIR` | Exact JSON records; independent 20/50/100/120 denominators with offsets/tags and exclusions. |
| `prepare-release --manifest FILE --config FILE --output DIR` | Hashed indexed source files, authoritative metadata, affine cohort; real structural archive audit still required. |
| `coordinated-smoke --config FILE --pool DIR --image ID --output DIR` | Host-owned Docker strict GPU updates/containment; `--seconds` and `--resume-run` keep actual budgets/state. |
| `qualify-baseline --checkpoint FILE --run-dir DIR --preflight FILE --preflight-config FILE --suites FILE --output DIR` | Actual run-bound G0–G6, native parity, executed GPU containment and non-skipped owning suites. |
| `diagnose-overflow --strata FILE --output DIR` | Sealed bounded wider-reference traces; optional provenance/archive/holdouts/development; unchecked bytes cannot enter training. |
| `prepare-measurement` | Explicit archive/cohort/holdouts, G7/G8, baseline report/checkpoint, resolved image and output; `measurement/v2` roles contain no final truth. |
| `measure --manifest FILE --output DIR [--dry-run]` | Frozen capped diagnostic/live protocol; records explicit failure/unavailable/inconclusive decisions. |
| `handoff --manifest FILE --output DIR` | Closed enumerated hashed file inventory and readiness selections; copied evidence replay, atomic publication, no inferred authorization. |

The root Dockerfile owns `qualification`, `gpu` and default `smolvm` targets.
Existing device nodes/numeric groups only; no host driver installation,
privileged mode, host IPC/network or architecture spoofing. Host coordinator owns
Docker socket/deadlines; roles have read-only enumerated inputs, bounded output,
no network/socket and GPU devices only for learner/preflight. See
[operations](../../../docs/foundation.md) and [quickstart](../quickstart.md) for
resource caps, migration, recovery, actual validation and pending acceptance.
