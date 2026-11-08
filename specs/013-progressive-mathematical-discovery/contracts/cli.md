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

Blueprint `selected_stages` selects a closed dependency-respecting subset; defaults to capability + diversity rather than launching the maximum full-path reservation. Full-path blueprint explicitly includes all stages and finite budgets. Automated trial choices are deterministic children of frozen distributions, not unrestricted edits to immutable parents. Expanded primitive or source decisions requiring new runtime compatibility stop at a proposal boundary and require a new frozen child study.

Paths must be workspace-confined and exact. New outputs must be empty/nonexistent and ownership-tagged. An existing completed output is never overwritten. `init` reports missing workstation artifacts together in one inventory, rather than repeatedly asking for files; final freeze refuses guesses. Input archives are bounded/hash-verified before use.

## Results and exit semantics

Every operation emits a sealed result record and one final JSON line containing campaign/stage/task IDs where applicable, mechanical_status, scientific_status, reason, evidence path and missing-evidence list. Progress lines are bounded and distinguish liveness from committed work. First producer failure and local traceback remain available; “private detail omitted” cannot be the only diagnostic.

- Exit 0: operation completed mechanically, including a correctly reported scientific STOP_NO_GAIN/STOP_REGRESSION. `scientific_status` remains decisive; exit 0 alone is not a positive gate.
- Exit 1: technical execution or publication failure; preserve the original cause plus collector failures.
- Exit 2: blocked/incomplete prerequisites, support, budget or required evidence; do not launch dependent work.
- Exit 3: integrity/contract mismatch or quarantined lineage; reject mutation/reuse.
- Exit 130: operator interruption with explicit durable recovery status.

These are new CLI semantics; do not change existing generalization/PAC exit codes. `verify-bundle` returns 0 only when its claimed scope is independently reconciled, and can verify a negative scientific result. Missing authority produces 2, contradictory evidence 3. Packaging success cannot upgrade the underlying stage.

## Commands versus role operations

Host commands dispatch explicit installed operations: census, prepare-stream, audit-language, propose-trial, train, generate, admit-training, score, extract-components, verify-components, source-pilot, review-claim and publish/verify. Each operation's input channel is defined by the execution contract. The host launches no model, imports no Optuna, and receives no private truth in ordinary reports. Arbitrary operation names or supplied Python callbacks are forbidden.
