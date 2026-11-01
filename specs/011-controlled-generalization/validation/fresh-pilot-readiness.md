# Fresh paired pilot handoff — 2026-11-01

The operator approved starting fresh. Historical E0/P0 are deferred, with their
artifacts retained. T132–T137 deliver the next controlled P1 screen through the
temporary `fresh_pilot.py` runner, `fresh_pilot_inputs.py` frozen blueprint and
`fresh_pilot_report.py` evidence projection. The complete operator command is at
the top of [quickstart.md](../quickstart.md). All changes stay on PR#62's existing
`011-controlled-generalization` branch. No hardware run was performed during this
implementation and CI was not awaited.

## Frozen comparison

| Item | Definition |
| --- | --- |
| Arms | Static versus refreshed, fresh random initialization; original initial-weight hashes must match. |
| Model | Imported original recipe configuration only; no historical weights initialize either research arm. |
| Seed / targets / replay | 20261016, literal programs, no replay. Historical seed16 is not a fresh confirmation run. |
| Training support | 32 generic descriptors, A4/B6/C2/D20, equal family mixture. |
| Static / refresh | Static repeats the first32,768-row shard. Refresh consumes12 distinct shards,1,024 examples/family/shard. |
| Capacity | Source-only encoding floor27,267,072 tokens. Actual native/reference admission, isolation, quota and token-capacity gates still have to pass. |
| Protected cohorts | 128 rows each for selection/assessment × parameter/structural; prepared before training, cross-role exclusions applied. |
| Coverage | Explicit frozen quotas; long/short boundary16 AST nodes. Observed generic sources ranged1–18 nodes; the old32-node long quota was infeasible. This is a new benchmark definition. |
| Learner budget | 25M target tokens and6 hours/arm. Retain initial/latest and4/8/16/25M milestones under the existing full-state contract. |
| Evaluation checkpoint | Frozen latest-available policy at each learner's terminal boundary; no adaptive checkpoint ranking. |
| Selection controls | Raw, rescue, equal-budget, shuffled, absent, template, shared absent-input bank. Existing matched-drain/v2 portfolios; eight bank seeds20261101–20261108. |
| Measurement | 256 selection rows × seven policies × two arms, frozen before training. One selected-program outcome per metric slot; original proposal and cost ledgers remain separate. |
| Interpretation | Single-seed exploratory. Require matched exposure and at least95% primary completion before interpreting the comparison; missing/censored outcomes retain fixed denominators. No automatic advancement. |
| Assessment | Sealed; no target/score access by learner, generator or report. |

Preparation may stop on a genuine native admission, uniqueness, isolation, coverage
or capacity shortfall. It never silently relaxes the design. Source-only feasibility
passed all four128-row protected cohorts using the reference evaluator and actual
cross-role exclusions. That check is neither native agreement nor completed research
preparation; the full training corpus has not been admitted on the target machine.

## Reliability and resources

Complex preparation/persistence changes were completed together in reopened Phase B:
lazy deterministic quota selection, bulk immutable exclusion indexes, full-shard
manifest bounds, bounded payload-free schema errors, and verified reuse of byte-identical
checkpoint components. Every debit-before/update-after checkpoint remains durable.
Generation-local links preserve exact bytes and survive predecessor pruning; unsupported
links fall back to copying. Actual serialization/write time remains charged. This does
not promise better model quality, different numerics or25M completion within6 hours.

The60-hour outer ceiling includes one current-source acceptance session (up to395
minutes), canonical preflight/preparation/training, isolated generation/scoring and
one hour protected for final report/verification/recovery. Image builds happen first
and are outside this clock. Each stage gets explicit host/setup/cleanup headroom;
learner and inference work retain their existing semantic limits. This ceiling is
not a runtime estimate. The last old-source acceptance took about24 minutes, which
does not predict full research preparation or training throughput.

All qualification/research consumers use64GiB/four CPUs. The helper preserves the
pinned image/lock and no-Lt policy; it does not upgrade ROCm, alter host drivers or
mount the checkout into learner/generator/scorer roles. Qualification remains bound
to current source, image IDs, device/runtime, resources and original case proofs.
Imported weights are used only by the existing software acceptance cases.

The frozen study hypothesis binds the full private protocol identity. The archive
contains the public protocol summary and exact helper sources as text. Generator
jobs receive visible prompts and frozen weights only; the scorer receives selection
targets and generated candidates through disjoint channels. Assessment stays private.

## Evidence and failure behavior

The controller stops after failed acceptance, preparation, learner, generation or
scoring boundaries. A legitimate PARTIAL learner can continue to paired diagnostic
evaluation; it is never relabeled cap-complete. Initial-weight mismatch stops before
generation. Original command results, retained metadata, measured exposures, policy
outcomes and cost ledgers are preserved. A mandatory controller receipt captures
failures that occur between command completions.

Canonical `report` packages the declared public inventory; `verify-bundle` independently
reconciles it. Success requires sealed terminal and summary records bound to the exact
archive bytes and expected scope/exit code. FAIL/PARTIAL/INTERRUPTED scopes are valid
negative evidence when integrity passes. Missing or contradictory verifier output
produces a diagnostic-only recovery ZIP. Diagnostics walk only bounded control paths,
project status/timing metadata and retain bounded log excerpts; protected target trees
are excluded. Public-copy size/member limits and symlink rejection precede packaging.

The final JSON gives the upload path. Keep private run files for later exact resume
or declared analysis. Do not rerun into the same directory or reinterpret recovery
packaging as research completion. Hard power loss or exhausted filesystem capacity
can still prevent final packaging; the preserved originals remain the recovery source.

## Local validation

- Broader generalization/helper suite:375 passed, one unavailable-native-dependency
  skip, one GPU test deselected. No CPU result substitutes for HIP evidence.
- Final temporary-helper suite:31 passed after final control/binding changes.
- Independent algorithm tests compare quota selection with the prior greedy oracle
  across80 randomized cases and bound work on32,768 rows; bulk exclusions match
  independent family/source/group/variant/window checks.
- Real CPU full-state interrupted/resumed training tests pass. Checkpoint tests cover
  changed and unchanged bytes, predecessor pruning, corrupt reused bytes, atomic latest
  publication and hard-link fallback.
- Mocked expensive process boundaries cover every failed command, both normal and
  partial completion, initialization mismatch and missing verification. These are
  controller tests, not synthetic hardware qualification.
- Actual canonical offline report/verify round trips preserve PASS, PARTIAL, FAIL and
  INTERRUPTED using explicitly synthetic original-proof fixtures. Changed archive
  binding, oversized evidence and symlinked extras are rejected.
- The host dry-run works under `python -S`, without installed third-party packages
  or container access. Actual role jobs retain canonical qualification checks.

Remaining execution gate: run the consolidated command on the HIP workstation,
then examine the returned archive for actual admission, finite training, complete
accounting and selection results. No fresh scientific outcome is available yet.
