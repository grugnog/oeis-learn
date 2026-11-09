# Proposed CLI — discovery commands

These commands are design contracts, not available commands at planning time. Implement beneath `python3 -S scripts/foundation_run.py [global engine/resource options] discovery COMMAND`. Use existing host confinement and single-engine behavior. Global resource options precede `discovery`; accept Docker or Podman. Stdlib-only operations must work without model/native/Optuna imports.

| Command | Required arguments | Contract/result |
| --- | --- | --- |
| init | --reference-campaign PATH --output NEW_DIR | Audit retained manifest/input identities, emit blueprint and unresolved-input report. No guessed paths or executable freeze; no engine needed |
| qualify | --blueprint FILE --output NEW_DIR | Build/inspect exact images and run CPU/native + actual HIP/shape capabilities selected by blueprint; no training research |
| prepare | --blueprint FILE --qualification DIR --output NEW_DIR | Bound census, exposure grouping, witnesses, train shards, source suitability and measured feasibility; never score model outcomes for cohort choice |
| freeze | --blueprint FILE --prepared DIR --qualification DIR --output NEW_DIR | Resolve exact sources/rosters/images/task DAG/budgets, validate scope and seal campaign/runtime. Any unresolved field blocks; no engine needed for verified input assembly |
| acceptance | --campaign FILE --output NEW_DIR | Run shortened full-feature CPU/native/HIP/recovery/isolation and export cases, label DIAGNOSTIC_ONLY; produce compatible feature receipt |
| run | --campaign FILE --acceptance DIR --output NEW_DIR | Execute selected finite DAG, declared gates and conditional continuations. No --start-step; never infer permission to change study choices |
| status | --run DIR | Read-only stage/task/remaining/omissions; no engine |
| pause | --run DIR | Request owned safe update-boundary checkpoint; wait for acknowledgement or report pending/external deadline, never kill unrelated jobs |
| resume | --run DIR --output NEW_INVOCATION_DIR | Same campaign/task compatible state and remaining allowance; original state and chronology preserved |
| report | --run DIR --output FILE.zip | Bounded success/partial/failure export; no device needed |
| verify-bundle | --archive FILE.zip --output NEW_DIR | Stdlib independent reconciliation and verified/incomplete/invalid terminal record; no engine |
| assess | --campaign FILE --selection FILE --acceptance DIR --output NEW_DIR | Explicit locked final-model assessment after selection freeze; evidence cannot reenter adaptive stages |

Blueprint `selected_stages` selects a closed dependency-respecting subset; defaults to capability + diversity rather than launching the maximum full-path reservation. Full-path blueprint explicitly includes all stages, finite budgets and any concrete contingency branches. No reserve implies permission to invent a new experiment after results. Status/report distinguish confirmed adoption from a bounded exploratory action. Automated trial choices are deterministic children of frozen distributions, not unrestricted edits to immutable parents. Expanded primitive or source decisions requiring new runtime compatibility stop at a proposal boundary and require a new frozen child study.

Paths must be workspace-confined and exact. New outputs must be empty/nonexistent and ownership-tagged. An existing completed output is never overwritten. `init` reports missing workstation artifacts together in one inventory, rather than repeatedly asking for files; final freeze refuses guesses. Input archives are bounded/hash-verified before use.

## Results and exit semantics

Every operation emits a sealed result record and one final JSON line containing campaign/stage/task IDs where applicable, mechanical_status, scientific_status (evidence_verdict), next_action, target_policy, reason, evidence path and missing-evidence list. Progress lines are bounded and distinguish liveness from committed work. First producer failure and local traceback remain available; “private detail omitted” cannot be the only diagnostic.

- Exit 0: operation completed mechanically, including a correctly reported NO_MATERIAL_GAIN/REGRESSION or an eligible bounded diagnostic action. `scientific_status` remains decisive; exit 0 alone is not a positive gate.
- Exit 1: technical execution or publication failure; preserve the original cause plus collector failures.
- Exit 2: blocked/incomplete prerequisites, support, budget or required evidence; do not launch the blocked dependency. A separately frozen diagnostic action must validate its own prerequisites and reservation before it can run.
- Exit 3: integrity/contract mismatch or quarantined lineage; reject mutation/reuse.
- Exit 130: operator interruption with explicit durable recovery status.

These are new CLI semantics; do not change existing generalization/PAC exit codes. `verify-bundle` returns 0 only when its claimed scope is independently reconciled, and can verify a negative scientific result. Missing authority produces 2, contradictory evidence 3. Packaging success cannot upgrade the underlying stage.

## Commands versus role operations

Host commands dispatch explicit installed operations: census, prepare-stream, audit-language, propose-trial, train, generate, admit-training, score, extract-components, verify-components, source-pilot, review-claim and publish/verify. Each operation's input channel is defined by the execution contract. The host launches no model, imports no Optuna, and receives no private truth in ordinary reports. Arbitrary operation names or supplied Python callbacks are forbidden.

### Installed predecessor outputs and bounded study alternatives

A frozen request channel may contain a literal permitted envelope path or a closed
producer slot, for example:

```json
{"task":"candidate-generate-1","operation":"generate","output":"candidates","kind":"candidate"}
```

Slots refer to earlier tasks in the same slot or declared ancestor slots. The host
resolves only original committed outputs under that campaign and checks the
paired seed, role, purpose, checkpoint and output type. Supported outputs include
training `weights`, generated `candidates`/public `evaluation`,
scored `scores`, census `census` and audit `coverage`. Public census deficits may
be consumed as explicitly partial evidence. Private witnesses and continuations
never become learner or generator channels. Full-state continuation uses the
frozen parent-task/slot initializer or the verified full-state input closure; a
checkpoint metadata receipt is not an additional learner input. Public output envelopes retain their
producer task and original artifact reference. A decision template binds task ids
and derives its table from these original outputs; absent paired cells become
unresolved and cannot authorize advancement.

`prepare` consumes the blueprint's named `preparation_plan` reference, with version
`progressive-preparation-plan/v1` and a closed prepared payload. Omit `blueprint`
from that payload to avoid a circular identity: preparation inserts the enclosing
blueprint identity. All runtime, qualification and input references are verified;
unresolved names remain preparation deficits. This assembly never selects a
cohort using model outcomes or manufactures a future producer.

A `progressive-study-policy/v1` policy freezes screening caps, the 128-task
outcome-blind development panel, startup/training/three-evaluation/finalization
feasibility, original producer bindings for each evaluation, and separate
confirmation panels/seeds. Task `trial` metadata binds its study, ordinal, phase
and charged-token evaluation coordinate. `trial_choices` is a finite catalog of
parameters, learning compatibility, operation policy and (when needed) exact
runtime and iteration-calibration references. Proposals select from that catalog
without changing any task id, resource ceiling or campaign allowance. Unsupported
proposals terminate INFEASIBLE before learning; baseline and fixed controls retain
their unpruned designation.

A catalog can share a frozen training-only input template. The host expands only
the registered model shape/seed/dropout and update-policy choices over unchanged
rows, bins and stream law. Freeze verifies every resulting compatibility and
qualified scope. There are at most 512 installed task plans, 648 alternatives per
plan and 32 MiB in a campaign/prepared/supervisor envelope; other individual JSON
limits remain unchanged. These limits accommodate the sixteen trials and their
three separate learning/evaluation cuts without creating a general scheduler.

Research screening children reserve one shared finalization envelope of at least
30 minutes per logical trial, included once in the frozen slot and two-hour total;
each child also retains its own operational finalization bound. The sampler sees
only host-verified development scalar history. Original outcomes, retention,
checkpoint and supervised phase costs remain in the durable study journal. Only
completed eligible equal-token trials can win; partial, failed, pruned, infeasible
and interrupted trials retain their charges. Pruning requires five completed
comparable observations and never applies to the registered controls or sizing
roster. Short pruning coordinates exist only in explicitly acceptance-purpose
rehearsals and cannot enter research. Cancelled later tasks are owned SKIPPED
slots with unused entitlement, never completed learning evidence.

Finalist locking includes the feasible baseline and at most three distinct
nonbaseline configurations. Confirmation uses separately frozen fresh model and
optimizer starts, paired seeds and the 400M schedule horizon; its outcomes never
enter sampler feedback. All actual-HIP calibration and qualification requirements
remain separate from CPU/native software acceptance.
