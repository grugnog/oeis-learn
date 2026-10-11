# Foundation operations and evidence

Spec 007 implements the shared WAT foundation through phase 11. Its CPU, native,
installed-wheel and isolation tests establish bounded software behavior. Actual
Ryzen/HIP updates, real indexed release-source acceptance and the capped
measurement/pilot protocol still require evidence. Run lifecycle completion,
archive creation and checked task markers never authorize a full release.
See the [quickstart](../specs/007-experiment-foundation/quickstart.md),
[CLI contract](../specs/007-experiment-foundation/contracts/cli.md) and
[gate handoff](../specs/007-experiment-foundation/validation/handoff.md).

## Choices and frozen profiles

User-confirmed choices are twenty observed terms, generic bootstrap learning,
shared WAT/LODA comparison preparation, the available Ryzen machine, KV caching,
repair of actionable non-LODA findings, and bounded experiments. The 100-term
horizon (20 visible + 80 private), model shape, seed, corpus counts, numerical
thresholds and experiment caps are explicit planning defaults, not additional
user approvals. Changing a frozen profile requires a new identity/run.

| Profile or mode | Implemented boundary |
| --- | --- |
| `foundation/v1`, `wat_i256_checked_v1` | Checked signed-256 logical results; declared i32/i64 bit-vector semantics; independent exact reference; fresh calls. |
| `wat_body_decimal_v1` | Lossless signed-decimal literals and distinct PAD/BOS/EOS; body cap 1,024 tokens. Unknown/truncated tokens reject before admission. |
| `ryzen_foundation_v1` | 1M fuel per term, 50M aggregate, two-second candidate/reclaim deadlines; quotas frozen in `wat_profile.yaml`. |
| `prefix20_total100_v1` | Historical indexed cohorts; prefix rebased to model index zero with original offsets preserved as provenance. |
| `prefix20_total100_affine_v2` | New release cohorts keep exact affine/shift families and reserved prefixes together; select before admission. |
| `strict_generic`, baseline smoke | Random initialization; admitted generic programs; SFT FP32, d=256, four encoder/decoder layers, four heads, feed-forward 1,024; three updates, batch four. Baseline input features can be lossy. |
| `qualified/v1` learning | Separate closed learning contract, exact-byte conditioning, cache/policy/replay/curriculum evidence, full active-state checkpoint; owning G7/G8 prerequisites. No in-place reinterpretation of baseline checkpoints. |
| Richer WAT/streaming/discovery profiles | Versioned bounded arrays, wide helpers and streaming state; generated archive and structural holdouts; owning G9 validation. |
| `cpu_fixture` measurements | Actual CPU timings with shortfalls/unavailable observations; diagnostic decisions cannot adopt a GPU optimization. |
| `qualified_device`, `measurement/v2` | Run-bound baseline, G7/G8, generic archive, private development cohort and resolved image; coordinated live execution without final truth. |

`configs/foundation/` and [execution.md](../specs/007-experiment-foundation/contracts/execution.md)
contain the exact effective settings and hashes. Optional disabled baseline solvers,
optimizers, RL and proving do not imply that their qualified repairs are complete;
their owning executable gates are separate.

## Roles and information boundaries

The host `RoleCoordinator` owns Docker launch, IPC, deadlines and cgroup
reclamation. At most one main role and eight workers are active: the main role
has 88 GiB/four CPUs/512 PIDs; workers each have 1 GiB/one CPU/64 PIDs, within the
96 GiB declared container cap. Equal memory/swap limits disable swap. GPU exposure
is limited to learner/preflight roles; workers have no GPU capability.

| Role | Supplied view |
| --- | --- |
| Controller/admission | Original sources, membership/holdout policy, exact execution evidence and durable accounting. |
| Learner | Copied admitted IDs, twenty visible values and complete body tokens; no evaluation truth or imported solution archive. |
| Generator | Visible prompts and pinned weights/settings; candidates are sealed before hidden scoring. |
| Evaluator | Sealed candidates and private development/final values; final access requires the decision lock. |
| Execution worker | Enumerated request and bounded output, CPU-only native cache; no truth, Docker socket or network. |

The root filesystem and input mounts are read-only, networking disabled, with
nonoverlapping output mounts and Docker default seccomp. Host kill deadlines
reclaim stuck roles/workers. Shared development files alone are not containment
evidence. `scripts/foundation/run_container.sh` takes named `--role`, `--image`,
`--inputs`, `--output`, `--seconds` arguments and a command after `--`; it uses an
existing immutable `sha256:...` image ID. The root Dockerfile remains the sole
build, with no host ROCm/compiler installation or driver changes.

## Migration and recovery

Historical wrappers use shared validation where supported and preserve historical
configurations/status. `legacy_unverified`, threshold-only readiness and
weights-only checkpoints do not become strict training or release evidence.
Random initialization from an admitted pool creates a new strict identity;
conversion cannot reconstruct unrecorded optimizer/RNG/provenance state.

CPU smoke requires `--device cpu --diagnostic`. The explicit
`--diagnostic-small-host` freezes lower floors of 1 GiB free disk and 256 MiB
available memory instead of the ordinary 100 GiB/16 GiB floors. Inspection and
reports preserve this deviation; it cannot qualify. `--prepare-only` freezes
inputs without starting updates, and `--stop-after-update 1` creates a clean pause.

`foundation resume --run-dir DIR` restores the newest valid complete update-boundary
state without model, seed, device, pool or budget overrides. Optional `--checkpoint`
asserts that newest state; it cannot force rollback. Checkpoints include model,
AdamW, explicit disabled scheduler/scaler, Python/NumPy/CPU/all-active-GPU and named
RNGs, permutation/cursor/next IDs, counters and complete qualified active state.
Package sources/resources, runtime/device and profiles are pinned. Drift requires
a new run even when tensor shapes match. FP32 precision is `highest`, TF32 is
disabled and deterministic algorithms enabled; unsupported HIP operations fail
visibly. Same-environment CPU continuation is tested; cross-device bitwise equality
is not claimed.

Blobs are fsynced/hashed before immutable manifest publication. The latest pointer
is a hint; recovery scans newest-to-oldest and quarantines corrupt/incomplete
generations without reusing numbers. Retain two valid generations and a pinned
final, with quarantine inside the 40 GiB checkpoint quota and twice the estimated
largest blob reserved for writes. Finalized runs cannot resume training.

A separate controller writes durable start/phase/heartbeat events while a deadline
process enforces limits. Clean pauses exclude downtime. Unclean gaps are charged
through recovery; monotonic same-boot and UTC cross-boot accounting fail closed
on inconsistent clocks. Model rollback never rolls back charges. Events have
64 MiB segments and separate 1 GiB budget/metric quotas. Retried updates remain
visible; the checkpoint's committed update counts progress. Unavailable metrics
are null, measured zero remains zero, and overlapping APU/RSS allocations are not
summed as a purported total.

## Finite evidence and certificates

Matching a prefix or all 100 terms establishes only that finite match. Wrong
values, incomplete output, overflow, timeout, unsupported syntax and infrastructure
failure remain separate outcomes. Any-of-budget and selected top-one scores use
the fixed denominator and pre-hidden selector. Final evaluation requires a locked
checkpoint/protocol/stopping decision and per-target seals; repeated evaluation
reuses those seals rather than drawing again after hidden feedback.

Grounding assignments and optimizer rewrites pass final exact execution. Solver
UNKNOWN differs from scoped UNSAT. Discovery starts from actual admitted programs
and sealed relation proposals. Certificates distinguish fitting, independent
validation, bounded machine checking and class-specific polynomial-program, linear-recurrence,
finite-state, loop-invariant and bounded-machine evidence. Unsupported families return UNKNOWN.
A checked mathematical certificate does not prove the WAT compiler/lowering:
`wasm_theorem` remains false without a formal link proof. Novelty lookup occurs
only after sealing; no finite match automatically claims a theorem or novelty.

## Measurement decisions and handoff

One capped engineering session (two hours) precedes at most one six-hour
diagnostic-plus-paired learning contrast. The frozen trigger order, practical
effect rules and three paired seeds come from the experiments contract. Every
attempt, recovery and evaluation is charged; there is no Cartesian sweep.
Reports distinguish adopted, baseline_retained, not_triggered, inconclusive,
unavailable and failed outcomes. Missing source, trained-model, device or thermal
observations cannot become zero, a speedup promise or an experiment winner.

`foundation handoff` copies only explicitly enumerated hashed files into an atomic
archive (4,096 files/40 GiB cap), replays G0–G10 from copied dependencies and records
identities, decisions and pending/failing gates. Replays check current package
implementation; stale evidence rejects. The 68-item disposition inventory is bound
to the original source and actual task checklist. Neither archive completion nor
that reconciliation grants release authorization. Preserve earlier phase evidence
as history rather than rebinding its validation to changed code.

Only native LODA runtime, paired language comparison and conditional transcoding
questions move to 008. Remaining GPU/source/measurement acceptance remains in 007;
008 experiments must wait for applicable 007 gates. GPU acceptance uses the existing
`lab-gpu image ensure` / `lab-gpu exec --` workflow and actual HIP tests, including
`test_foundation_gpu.py`, `test_foundation_learning_gpu.py` and the GPU coordinator
test with `OEIS_GPU_ROLE_IMAGE`. Skips leave those tasks half-checked.

## Phase 13 evidence identities

New preflight evidence uses `preflight-report/v2`. The preflight-only configuration
requires a resolved `base_image_digest`, the observed built `final_image_id`, and
confined `dockerfile`/`dependency_lock` file paths relative to that configuration.
The emitted `config_digest` hashes the complete effective preflight configuration;
it no longer stands for the parent-image digest. The report separately hashes the
Dockerfile and dependency lock and records observed runtime/device details.
`foundation qualify-baseline` requires `--preflight-config` in addition to the
report, actual checkpoint, run events and owning suites. G0 copies/replays those
selected inputs and checks image/runtime/device against checkpoint provenance and
containment. Old preflight reports remain historical and cannot qualify G0.

G8 reports use `g8/v2`: software JUnit plus optional hardware process/JUnit
evidence. Unrequested hardware is `PENDING`; nonzero exit, launch failure or
rejected/missing hardware JUnit is `FAIL` with durable diagnostics. Even diagnostic
replay rejects a recorded hardware failure. Actual accelerator acceptance remains
a separate phase-14 run. CPU input diagnostics can be replayed with
`python scripts/verify_phase13.py --output DIR`; missing indexed source and trained
artifacts stay explicit and cannot qualify T113/T114.
