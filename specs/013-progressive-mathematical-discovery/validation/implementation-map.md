# Spec 013 implementation map

Scope: T001–T023. Read spec.md, plan.md, tasks.md, research.md, data-model.md,
quickstart.md, hardware-validation.md, planning-review.md, all five contracts,
the requirements checklist and adopted constitution in full before implementation.
Requirements checklist: 16 checked, zero unchecked. No extension hooks registered.

## Reused owners and preserved guards

| Boundary | Existing owner | 013 adapter |
| --- | --- | --- |
| Portable host | scripts/foundation_run.py → foundation_local.py::parser/main | progressive_cli.py → progressive_commands.py; stdlib host |
| Installed role | generalization_roles.py::invoke / project_and_invoke → generalization_role.py | progressive_roles.py; reuse process supervision and engine isolation |
| Real learning | generalization_training.py::run_torch → torch_update → rl/sft_trainer.py::supervised_step | later progressive_training.py; no second learner in this increment |
| Full checkpoint | fresh/bigint_state.py::GeneralizationCheckpointStore.save/load | later progressive state adapter, preserving atomic generations |
| Actual checkpoint evaluation | generalization_evaluation.py::load_retained | progressive/baselines.py; no invented checkpoint weights |
| Native and reference | synthesis/client.py, reference.py, reference_parser.py; decoder/bigint_codec.py | progressive/coverage.py with independent witness checks |
| Qualification | generalization_qualification_io.py::produce/consume_original/current_facts | discovery runtime binds original producers and exact scope; actual new qualification later |
| Evidence | generalization evidence/reporting and bundle.py; supervision.py::launch/reclaim | progressive_evidence.py; full independent bundle reconciliation T079–T080 |
| Source closure | experiments/source_snapshot.py → pipeline_validation/closures.py | include all 013 code/tests/contracts/assets; no new exclusions |

Preserve generic 21,600-second guards in generalization_training.py and
generalization_roles.py, the PAC envelope validator, PAC fresh-random/no-replay
checks in run_torch, execution.py::_pac_policy's fixed 25M policy, mechanical
replay_admission.py's oracle/shorter-target restrictions and legacy interval
requirements. Discovery authority uses its own versions. The existing
rl/progressive.py and legacy discovery commands are unrelated and unchanged.

Proposed new files are the progressive package modules and
pipeline_validation/progressive_{cli,commands,roles,preparation,evidence}.py listed
by T002–T023. Canonical seams above resolve; generalization_bundle.py does not
exist: the actual shared archive owner is pipeline_validation/bundle.py.

## Validation

Source/test implementation is complete for T001–T017 and T019–T023. T018 remains unchecked and BLOCKED: the required container/native execution has not run. This environment has no Docker or Podman binary
and no packaged native worker or HIP device. Local CPU/stdlib checks cannot
establish container/native/HIP qualification. No training/research is launched.


## Delivered boundaries

- T001–T005: reuse/guard inventory, model-free namespace, exact Optuna 4.5.0
  dependency and lock closure, installed schemas/policies/contracts, hand-checked
  diagnostic fixtures with separate public and private files. The pinned uv 0.8.22
  resolver adds six dependencies; no existing locked package version changes.
- T006–T013: closed envelopes/references, non-executable blueprints, finite task
  plans/DAG and storage graph, exact original producer matching, append-only
  exposure and finite-window/structure relations, typed projections, classified
  targets, durable reserve/commit/reconcile and boot-bound active clocks. Unknown
  supervisor termination consumes the unproven remaining task clock. Policy caps
  keep the 24h task limit and the non-additive 168h/24h scaling split.
- T014–T017: installed operation registry, canonical campaign accounting and task
  leases, existing watchdog/reclamation, conservative runtime authority, complete
  input assembly and immutable freeze, all 12 command argument surfaces, initial
  independent-status diagnostics and local original tracebacks. Working host
  commands: init/freeze/status. Commands owned by later tasks fail explicitly
  with exit 2. Full-state continuation intentionally rejects until T049/T050.
- T019–T023: independent metric/gate specifications, fixed evaluation profile,
  four-stratum census and separate public/truth publication, private witness audit,
  retained/random/earlier checkpoint adapters, deterministic conditioning controls,
  existing generic templates and bounded exact polynomial/linear-recurrence
  baselines with one shared portfolio and per-method measured costs.

Task plans exclude the campaign hash until issuance, avoiding a circular content
identity. The frozen campaign contains the complete immutable roster; issued tasks
must exactly match it and bind the final campaign hash. Learning compatibility
excludes task paths, elapsed entitlements and seed choice. Recipe model scope is
the canonical hash of resolved model fields excluding seed. Original qualification
must explicitly cover this model scope and the selected language; old unrelated
receipts do not automatically qualify discovery work.

`init` can consume a sealed `discovery-retained-inputs/v1` retained-inputs.json
with explicit named references at the original campaign root. It verifies supplied
bytes and reports all unresolved original inputs together. No search outside that
root or supplied references substitutes guessed workstation paths. The default
blueprint has capability+diversity selected and zero executable allocation until
measured preparation resolves and freezes actual caps.

The explicit `progressive_cli.CPU_TESTS` inventory is included in the existing
bigint-cpu workflow. It excludes the unrelated historical rl/progressive test
names; no source/test bytes are excluded from qualification closure. The later
T085 task still owns complete feature-wide CI/hardware inventories.

## Validation results and limits

Local interpreter: Python 3.12.14; actual model test uses torch 2.9.1+cpu.
The container pins and native compiler/runtime closure remain unchanged.

- Failing-first authority check: `PYTHONPATH=src .venv-progressive/bin/python -m
  pytest tests/contract/test_progressive_authority.py -q` initially fails collection
  with the missing contracts module; the implemented authority tests now pass.
- New increment tests use the explicit CPU_TESTS list, including real canonical
  encoder/decoder weights save/load and exact tensor comparisons, real subprocess
  hang cleanup, filesystem role projections, successful diagnostic freeze and
  Python `-S` isolation for both successful and blocked host paths.
- The existing `tests/contract/test_foundation_packaging.py` builds a real wheel,
  extracts it outside the source tree and validates the complete installed source
  closure including the copied 013 markdown contracts and JSON assets.
- Affected historical selection: generalization authority/roles/contracts/budgets/
  operational budgets/execution/qualification/exclusions/metrics/legacy/host,
  registration and foundation packaging: **137 passed** before final combined run.
- `uvx --from uv==0.8.22 uv lock --check`: PASS. Existing locked versions unchanged.
- `python3 -S scripts/foundation_run.py discovery init ...` and read-only `status`:
  mechanically complete; missing original campaign explicitly reported. All 12
  parser surfaces are tested; unimplemented endpoints do not contact an engine.
- `python3 -S scripts/foundation_run.py --engine docker validate --suite bigint-cpu
  --output run-state/discovery/013-t001-t023/cpu-native`: **BLOCKED**, exit 1,
  `docker CLI is unavailable; select an installed container engine`. Podman is
  also absent. T018 is not marked complete. The native witness integration test
  is skipped locally and becomes mandatory in the existing native-required image.
- T019 has **8 strict expected failures** solely because T024/T025 modules are
  absent. These are independent future behavior specifications, not claimed
  passing metrics/gates. Their strict markers must be removed when those owners
  are implemented; an unexpected pass fails the suite.
- Broad `test_progressive_*` globs also select old rl/progressive tests. Three
  failures reproduce identically on unmodified PR head 7829e13 in a detached
  worktree: two Wasmtime fallback TrapCode failures and the legacy tier-2 shaping
  rejection. They were not changed by this increment. Use the explicit new
  inventory to distinguish these from spec 013 tests.

No actual HIP, sustained run, retained D evaluation, training campaign, source
acquisition or scientific advancement was executed or inferred. T024 onward,
including the independent metric reducer, gate decisions, executable scoring and
readiness MVP publication, remain outside this requested increment. Original
historical evidence and qualification scope are unchanged.

Final combined increment + affected historical run: **206 passed, 1 skipped,
8 xfailed** (9.33 seconds). The skip is the actual packaged native-service witness
case; the eight expected failures are T019's future T024/T025 specifications.
The final run includes all 17 explicit CPU_TESTS files plus the historical selection
listed above, with foundation wheel extraction/source verification included once.
No container, native or HIP status is upgraded by these local results.

Final whitespace and new-source/test Ruff checks pass. No extension pre/post hooks
are registered. Research and later task boxes remain unchanged. This increment is
reviewable independently; workstation execution of the canonical bigint-cpu gate
is still necessary to complete T018.
