# CLI and runtime boundary

These are **planned commands**, implemented by tasks in 007. Only the Spec Kit commands in the validation guide exist as part of this documentation change. Existing `oeis-learn` commands are retained as legacy interfaces; new commands live under `oeis-learn foundation` in `src/oeis_learn/cli/foundation.py` and are registered by `cli/main.py`.

## Common behavior

Every command accepts `--json` (one summary JSON object on stdout, progress on stderr). Paths resolve without shell evaluation. Unknown flags/configuration fields fail. Exit codes: 0 success including a successfully recorded candidate rejection; 2 invalid input/configuration; 3 hardware unavailable; 4 correctness/provenance/readiness gate failed; 5 resource or infrastructure interruption. A partial result cannot be returned as qualifying success. Summaries include command, status, run/artifact identities and result paths.

| Command | Required inputs | Output / boundary |
| --- | --- | --- |
| `foundation preflight --config FILE --output DIR` | Frozen candidate Docker/image settings; explicit requested device. `--hardware-only` permits standalone pre-adoption feasibility and does not require cohort/pool. | Host/container report, dependency/image identities, actual-model GPU update evidence. No CPU fallback. Does not start a training experiment. |
| `foundation conformance --profile FILE --output DIR` | Language/resource/runtime profile | Regression and differential report; all active adapters qualified or explicitly unavailable. No checkpoint/benchmark needed. |
| `foundation freeze-cohort --source DIR --config FILE --output DIR` | Pinned local indexed source and seed/counts | Separate private truth, split manifest, exclusion census and reserved-prefix membership artifact. No corpus download without explicit source command. |
| `foundation build-pool --config FILE --cohort DIR --output DIR` | Generic generator configuration and admission service's private reserved-prefix membership | Immutable admitted pool and rejection/evidence archive. No legacy auto-generation fallback. |
| `foundation train --config FILE --pool DIR --run-dir DIR` | Admitted immutable pool, matching profiles, device/preflight evidence | Strict random initialization, verified SFT, ledger and complete checkpoints. `--device cpu --diagnostic` explicitly creates an unqualified CPU fixture; cannot satisfy GPU readiness. |
| `foundation resume --run-dir DIR --checkpoint MANIFEST` | Complete compatible manifest/blob and durable ledger | Continue existing run after hash/state validation. Configuration overrides forbidden. |
| `foundation evaluate --checkpoint MANIFEST --cohort DIR --split development --protocol FILE --output DIR` | Actual checkpoint; frozen manifests; protocol budgets | Prefix-only generation process and sealed candidates, private scoring, full fixed-denominator report. |
| `foundation finalize --run-dir DIR --checkpoint MANIFEST --protocol FILE --cohort DIR --stopping-record FILE --output FILE` | Locked checkpoint/decision, adopted contract, ledger state | Immutable finalization record. This does not open final truth or change model state. |
| `foundation evaluate --finalization FILE --split final --output DIR` | Exactly one valid finalization manifest; do not also accept overriding checkpoint/protocol | Generate visible-only proposals and commit per-target seals under the decision lock, then score each sealed target; resume those same seals. No regenerative retries following hidden feedback. |
| `foundation inspect --run-dir DIR` | Existing run | Qualification, active/charged/downtime accounting, measured/unavailable metrics, artifact validity and remaining budgets. Read only. |

For experiment data, `--cohort DIR` is resolved by the evaluator/controller. Only the derived visible-prompt view is passed to the generation process. The learner has no mount of this directory. Admission obtains only reserved-prefix membership; full evaluator truth is not sent to training workers.

## Configuration and container outputs

`configs/foundation/wat_smoke.yaml` will name the exact profiles from [execution.md](execution.md), strict track, FP32 small model, seed 20260913, an explicit device, pool identity, complete checkpoint cadence and resource budgets. Initial smoke: three updates, batch four, maximum body context 1,024; no learning-rate/model-size/worker sweep. The larger first comparison is configured by 008.

`docker/foundation/Dockerfile` will derive from the digest resolved for `rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1`. This is a conservative documented baseline, not a claim that it is latest or NixOS-certified. Build captures the tag, digest, Dockerfile hash, exact wheel locks and final image identity. `scripts/foundation/build_image.sh` writes `runtime.lock.json` with real values. An unresolved placeholder or tag-only identity blocks qualifying preflight; image resolution occurs during implementation and is not falsely claimed in these specs.

`docker/foundation/compose.yaml` will expose only the existing `/dev/kfd` and selected render node to the learner/preflight process, using existing numeric device group IDs. No privileged mode, host networking/IPC, driver installation, architecture-spoofing override or broad host mount. It constrains the learner/controller to 88 GiB and each of up to eight worker containers to 1 GiB, for at most 96 GiB total declared container memory; identical memory/swap limits disable swap. Shared APU memory is monitored separately as described in the execution contract. GPU workers and evaluator/generator/trainer mounts remain distinct.

Workers use bounded local request/result IPC, not an external service. Controller creates workers, schedules requests and kills/replaces a stuck worker. Docker socket access, if the host launcher uses it, remains with that launcher; it is never mounted into learner or candidate containers. Build may access the network to resolve dependencies; execution containers have networking disabled.

## Qualified extension commands (implemented in 007)

All use the common exit codes, explicit profile hashes and role isolation above. Qualified commands load `qualified/v1` contracts; they cannot reinterpret a baseline schema in place.

| Command | Contract |
| --- | --- |
| `foundation qualify --config FILE --output DIR` | Run owning solver/proof/optimizer, native parity, cache/policy and richer-profile gates; emits passed/pending per capability. No disabled subsystem is counted repaired. |
| `foundation census --source DIR --config FILE --output DIR` | Independent 20/50/100/120 denominators and source/finite/table/growth metadata; read-only source. |
| `foundation train --qualified-config FILE --pool DIR --run-dir DIR` | Mutually exclusive with --config; explicit repaired objective/model/replay variants, same information boundary and complete qualified checkpoint. |
| `foundation search --checkpoint MANIFEST --prompts FILE --config FILE --output DIR` | Prefix-only bounded proposals and seals. No hidden-truth path argument accepted. |
| `foundation discover --archive DIR --config FILE --output DIR` | Analysis-role program-derived hypotheses and exact independent relation validation; immutable proposals before optional novelty lookup. |
| `foundation check-certificate --certificate FILE --profile FILE --output DIR` | Fresh checker process, no weights, explicit domain/semantics/result; finite match cannot invoke proof promotion. |
| `foundation measure --manifest FILE --output DIR` | Execute frozen engineering/pilot decision graph with cumulative caps, selected branch and stop records; --dry-run validates identities and schedule without training. |

Legacy launch/census/evaluation wrappers delegate to these shared services only after parity fixtures. Preserve a machine-readable historical config and `legacy_unverified` status; corrected services must never rewrite old results as strict successes. CLI aliases beyond `oeis-learn foundation` are optional and are not a required improvement.
