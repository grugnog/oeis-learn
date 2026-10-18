# Contract: Portable Commands, Recipes and Retirement

This is a planned command extension. Only existing commands retain their current behavior today. New flags, suites and configuration files become runnable after implementation Phases 3–5; the quickstart marks these explicitly.

## Entry point and extensions

Continue `python3 scripts/foundation_run.py --engine docker|podman [global-options] COMMAND`. Host Python stays standard-library-only. Resolve YAML/data/model dependencies inside the container and emit canonical JSON contracts for the host coordinator. No Docker/Podman socket is mounted into a learner, solver or candidate container.

| Command | Planned extension / semantics |
| --- | --- |
| `image ensure` | Build/install pinned bigint WASM and native fitter; inspect packaged identity; no runtime downloads |
| `validate --suite bigint-cpu|bigint-gpu|bigint-retirement --output PATH` | Add suites to existing choices. CPU arithmetic/fitting/reference; GPU large-input/cache/update; retirement installed imports/callers. Write per-stage logs and concise result JSON |
| `data fetch --config PATH` | Existing bulk fetch with immutable snapshot/digest and workspace cache; remains the only networked data role |
| `prepare --config PATH --output PATH` | Existing command with new recipe schema; emits paired literal/parameterized generic archive, split manifest, private scoring payload and failure counts |
| `check --inputs PATH` | Validate exact bytes, identities, provenance and complete split sizes; never fit constants or modify input data |
| `learnability --inputs PATH --run PATH --config PATH [--seed N]` | Add recipe loading for new numerical/input/model profiles; use diverse32 gate. Existing old-profile flags remain supported; conflicting explicit config/flags are rejected |
| `train --profile bigint-bootstrap --config PATH --inputs PATH --run PATH [--seed N]` | Add the profile and recipe controls; freeze contract first. No implicit 12h run or inherited mutable defaults |
| `evaluate --run PATH --split development --fitting raw|exact|paired --config PATH --output PATH` | New analysis options create a linked child evaluation identity. Checkpoint/input numerical semantics stay explicit; original training contract/ledger is never edited. Parent hidden final cohort cannot be reused as a fresh final experiment |
| `diagnose --run PATH --output PATH --ablate-inputs all` | New optional input-ablation report on declared diagnostic data; does not retrain or overwrite parent weights |
| `compare --runs PATH... --output PATH` | New CPU report command joins completed contracts/results by seed/split/identity; rejects mismatched comparisons and lists missing/incomplete arms |
| `resume`, `status`, `cancel`, `pack --run PATH --output PATH` | Existing command shape, new contract/ledger fields and priority bundle policy; resume cannot silently adopt a new image/solver/model profile |

All paths must resolve within the configured workspace except approved engine/device paths. Existing global resource overrides are recorded. Source data download role has network; solver/generation/scoring do not. Errors name the phase, concise cause, result/status path and owning stderr/stdout paths. Summary absence is itself a failure, not just a giant subprocess command echo. Unsupported old options produce a migration message; no legacy solver invocation.

## Recipe schema `bigint-bootstrap/v1`

Top-level sections: `data`, `execution`, `inputs`, `model`, `learning`, `fitting`, `search`, `evaluation`, `track`, `archive`, `budgets`, `reporting`. Each rejects unknown keys and resolves defaults to explicit values before hashing.

- `data`: pinned bulk source ID/cache, generic sampler revision/seed, concrete/parameterized pairing, parameter/structure split manifest and exact source indices.
- `execution`: language/codec/runtime hashes and numerical/resource profile from runtime contract.
- `inputs`: `hybrid-chunks/v1` or named retained control, exact bit/chunk limits, mathematical features and four group masks.
- `model`: full architecture, decoder gain/dropout, initialization and precision/attention choices; seed override permitted only before run creation.
- `learning`: literal/parameterized target view, learning rate, update/time caps, complete validation cadence, selection/patience and curriculum/replay policy.
- `fitting`: one native solver profile, coefficient/parameter/bounds/deadlines, ambiguity and ranking policy; `disabled` is an explicit raw arm, never fallback Python.
- `search`: candidates/tokens/portfolio ordering and aggregate visible-only budget.
- `evaluation`: fixed denominators, held-out scope/horizon, prompt controls, template baseline and separately capped hidden scoring.
- `track`: `generic_only` or `assisted`, optional shared-foundation checkpoint and reviewed assisted-manifest digest.
- `archive`: alternative policy/cap, exact evidence retention, byte/disk limits and omissions.
- `budgets`: preparation, training, fitting/evaluation, cleanup and cumulative ledger policy. No recursive stage budget multiplication.
- `reporting`: integer-size/structure/token buckets, selected example caps and bundle limits. Minimum bundle16MiB, core reserve8MiB, default bundle64MiB; bound core report schemas themselves.

Commit recipes `bigint_smoke.yaml`, `bigint_literal_screen.yaml`, `bigint_parameter_screen.yaml`, `bigint_assisted_screen.yaml` and `bigint_encoder_study.yaml` when integrated. Configs freeze the values in the experiment contract. Assisted configs fail validation until an explicit reviewed demonstration manifest and compatible generic foundation are supplied; do not silently download or invent teachers.

## Retirement disposition

| Surface | Disposition at feature completion |
| --- | --- |
| `decoder/constant_solver.py`, `grounding.py`, `dixon_solver.py`, `qfnia_solver.py` | Delete after replacing all active callers with the thin native transport/admission facade |
| `decoder/recurrence.py` fitting/recognition; `mersenne61_filter.py` | Remove solver-only implementations; independently required reference fixtures are rewritten in the independent checker, not copied as a second solver |
| `fresh/search.py::grounded_variants` | Remove competing literal/operator repair advertised as fitting; portfolio calls one fitter |
| `grounding_evidence.py`, `grounding_reference.py` | Separate independent reference and old-record readers; native scoped constraint replay replaces Python solving. Delete obsolete implementation modules when migrated |
| Python `z3-solver` dependency | Remove with lock/build updates after every active verifier consumer is migrated; native Z3 remains pinned separately |
| CLI/evaluation/qualification/RL active solver callers | Migrate supported entry points to the same service; remove explicitly unsupported methods with migration messages/tests |
| Old run008/009/010 launch scripts and exclusive configs/examples | Retire after reachability audit; preserve historical specs/results and reproduction commit IDs |
| Signed-256 strict evaluator, actual used reference/certificates and old checkpoint readers | Retain with explicit identities; no automatic new-profile reinterpretation |
| Other unused code | Enumerate consumers and future supported purpose; remove only entries classified retired, not whole subsystems by age |

The disposition manifest records old path/import/CLI, consumers, replacement or removal, rationale and regression fixtures. Clean installed-image audits must demonstrate no callable old fitting algorithms, no dynamic import fallback and no old solver package. Thin Python request serialization and independent final arithmetic verification are permitted, not Python trial search.

## Logs, recovery and export

Durably record contract and started budget before launching work. On timeout/crash, cancel/reap native workers and role containers, preserve consumed allowance and interrupted status, and publish summaries before optional packaging. Worker's generation invalidates late responses. Verify completed-stage identities before skip/reuse; only exact same contract can resume.

Core bundle records: `run.json`, compact `status.json`, `summary.json`, checkpoint identity, input/split/track manifest hashes, aggregate `comparison.json`/size/complexity metrics and `bundle-manifest.json`. Detail files are bounded separate chunks; include selected incomplete tokens, sources, constants and verification examples with hashes. No unbounded nested training result is copied into core status. Omissions name path/reason/hash/bytes; pack either retains all core records or fails clearly. Weights and private source/holdout payloads need separate explicit requests.
