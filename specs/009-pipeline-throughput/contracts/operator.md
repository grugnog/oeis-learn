# Contract: consolidated pipeline validation CLI v1

**Status**: Planned interface; not implemented by this planning PR. All durations below are finite operational ceilings, not predictions. Long validation is explicitly acceptable to the user.

## Public interface

Extend the existing command, preserving all current suites:

```bash
python3 scripts/foundation_run.py --engine podman --cpus 16 validate \
  --suite pipeline --phase all-ready \
  --manifest configs/foundation/pipeline_validation.json \
  --run runs/EXISTING_COMPLETED_RUN --inputs tmp-fresh-inputs-host \
  --output tmp-validation/009-validation
```

| Option | Contract |
| --- | --- |
| `--suite pipeline` | Consolidated entry point. Engineering phases1–5/`all-ready` cannot train, resume a learning attempt or evaluate a final cohort; separately admitted phase6 follows the research contract |
| `--phase all-ready` | Default: execute implemented/manifest-authorized engineering stages whose prerequisites can be satisfied, including across phases; no pause at document phase boundaries |
| `--phase 1` through `5` | Select that phase and missing prerequisites, deduplicated; no manual subordinate commands |
| `--phase 6`, `--study-manifest PATH` | Explicit research mode; study manifest required and mutually exclusive with engineering `--manifest`. See [research-gates.md](research-gates.md); final access additionally requires `--final-plan PATH` |
| `--manifest PATH` | Immutable `pipeline-validation/v1` manifest; default committed path above. Read/validate with host standard library before engine access |
| `--run PATH`, `--inputs PATH` | Read-only completed bigint run and compatible prepared inputs for neural measurements; missing/invalid inputs block dependent stages while eligible CPU-only checks may proceed |
| `--output PATH` | New workspace-confined directory, must not exist; create initial ledger before engine work; write sibling `PATH.zip` on completion/partial failure |
| `--reuse PATH` (repeatable) | Prior invocation directory or extracted manifest-verified bundle; import compatible successful stage receipts, never inherit a whole-run PASS blindly |
| `--engines available\|selected\|both` | Default `available`: primary `--engine` plus usable other supported engine for portability checks. `selected` limits execution but cannot qualify both engines alone. `both` explicitly requires both; missing capabilities are BLOCKED. GPU performance runs only on the manifest's primary engine |
| `--max-seconds N` | May lower the frozen invocation ceiling, never silently raise it; inability to reserve a selected stage yields PARTIAL/BLOCKED with a reason |
| `--scope-report` | Print resolved selected stages, dependencies, inputs and ceilings, without executing work or exposing private inputs |

Global foundation options retain their current meanings. `--image` pins the primary engine's image and still requires packaged-source/artifact checks. Secondary engines resolve their own digest/target; never assume a local image ID is portable across stores. CPU qualification and GPU image targets are each ensured at most once per engine per invocation, with all build work sharing that engine's build allowance.

After implementation, the default handoff is the single `all-ready` command. No separate build, HIP test, native benchmark, export or cleanup invocation is required. Relative example paths are resolved inside the configured workspace, with symlinks/path escapes rejected consistently with existing mounts.

## Static stage graph

A versioned registry in `pipeline_validation` declares stage ID, implementation availability, dependencies, capabilities, input closure, cost reservation, execute/read/verify adapters and affected gates. Unknown stage IDs, cycles, duplicated IDs and missing mandatory fields fail before engine access. Do not infer implementation availability from file existence alone.

| Stage family | Required predecessor | Behavior |
| --- | --- | --- |
| `manifest`, `inputs` | None | Scope/authority/path/hash checks; initialize ledger; validate public selected-weight/data identities without mounting final data |
| `engine.<name>` | Manifest | Availability, quotas, readable mounts/output and free workspace; missing capabilities recorded |
| `image.<name>.<target>` | Engine | Ensure once, inspect exact installed source/native/guest/policy artifacts; preserve cached dependency layers |
| `cpu.semantic`, `cpu.fit`, `cpu.containment` | Qualification image | Existing mandatory JUnit suites; zero mandatory skips/empty suites; disagreements fail closed |
| `operations.retirement`, `operations.export`, `operations.resume` | Applicable image | Clean installed retirement/readers, bounded normal/interrupted/resume/permissions fixtures; no real learning authority |
| `hip.preflight` | GPU image | Device/mount audit, actual finite parameter update, current supported HIP; trace capability only when a trace was requested |
| `baseline.calibration` | Semantic/device/input prerequisites for each subworkload | Baseline-only noise/cost/memory measurement, freeze derived comparison reservations before candidate observations |
| `inference.compare` | CPU/HIP parity and frozen workload | Isolated cache parity/timing; early v1 pipeline observations are diagnostic only. End-to-end adoption confirmation belongs to `overlap.compare` after common v2 policy qualification |
| `policy.compare` | Semantic/admission/archive tests | Serial v1/v2 selection and archive comparison; explicit policy-change ledger |
| `native.compare` | CPU suites, policy v2 tests | Fixed candidate 1/2/4 comparisons and bounded fault/stress cases; v2 on both sides |
| `overlap.compare` | Qualified policy v2 plus selected components or explicitly selected baseline substitutes | Combined work, fixed-budget coverage, timing-mode and resource checks; serial measurement execution |
| `qualification`, `cost.diverse32`, `cost.transfer` | Applicable current-source receipts | Engineering status and independently scoped schedule forecasts; a missing cost component blocks admission |
| `report`, `archive` | Always attempted | Mandatory ledgers/summary/omissions first; preserve partial/failure outcomes |

A correctness failure blocks its dependants, not unrelated safe suites. Global interruption, exhausted disk/memory, contradictory role audit or global deadline stops further work. No timing measurement overlaps another CPU/GPU measurement or build on the same host; ordinary correctness suites may be serialized too for simplicity.

`all-ready` reports three distinct facts: selected stage scope and its verdict; unavailable/not-yet-implemented/unrequested stages; feature-level remaining gates. A PASS for currently ready phases is labelled with those phases and cannot claim phase 5, both-engine qualification or whole-feature completion. An explicitly requested unavailable phase is BLOCKED. Any missing mandatory prerequisite inside selected scope prevents PASS.

## Initial budgets and resources

All limits are recorded in the resolved manifest. Lower limits may be requested; larger values require a new prospective manifest identity. The initial combined engineering ceiling is **86,400 seconds (24 hours)**, including builds/startup and packaging. This is not authorization for long training.

| Reservation | Maximum seconds |
| --- | ---: |
| Build/ensure allowance per engine, all image targets combined; at most two engines | 14,400 each |
| Phase 1 baseline/calibration and host/fixture checks | 3,600 |
| Phase 2 inference comparisons | 10,800 |
| Phase 3 policy/native comparisons and fault tests | 7,200 |
| Phase 4 overlap/joint comparisons | 18,000 |
| Phase 5 current-source operations/qualification/cost | 14,400 |
| Final report/archive | 900 |
| Invocation-wide reserved cleanup/emergency reporting | 300 |
| **Worst-case declared reservation total** | **84,000** |

Shared prerequisite stages are charged once, to their owning phase. Each phase manifest subdivides its total before execution; ordinary mandatory suites are capped at 3,600 seconds each and HIP preflight at 600 seconds, within those totals. A stage cannot borrow time from another phase without a prospective manifest revision. Before starting it, require remaining global time to cover the stage reservation plus cleanup/archive reserve. Unused time does not enlarge any per-prompt or learning allowance. Build cap exhaustion does not prevent export of the failure.

The default workstation policy has16 CPU equivalents and an aggregate memory ceiling of `min(floor(0.90 * physical_memory_bytes), effective_engine_or_ancestor_limit_bytes)`, rounded down to MiB. The user explicitly made90% of the128 GB machine available. Resolve this manifest policy before constructing Host so its older default16 GiB limit does not silently win. Record nominal physical, engine/ancestor, currently available and requested/enforced byte limits. The90% value is an enforcement ceiling, not an allocation or a minimum-free-memory requirement. Freeze each stage's expected active peak reservation from declared allocations and baseline calibration; add safety headroom of the larger of1 GiB or10% of that peak. Require peak plus headroom to fit the effective ceiling and currently available capacity before dispatch. Record both the peak reservation and maximum separately. If that reservation cannot fit, block the stage rather than silently shrink its comparison or trigger swapping. Unknown peak requirements cannot admit a large-capacity trial; a bounded preflight may establish them under its own declared limits. When device allocations are not fully charged to the CPU cgroup, monitor the global shared-memory envelope as well; unavailable accounting prevents an aggregate-memory qualification claim.

All overlapping CPU/GPU/native/cache allocations share this one budget. At most one compute role receives the full allowance; a separate scorer runs after the GPU role drains. Keep at least10% of physical memory outside the workload's authorized maximum. The initial1 GiB per-native-slot/4 GiB total native RSS watchdogs remain fault limits, not a16 GiB ceiling for the whole pipeline. Report shared-page accounting limitations.

Named resource profiles are `historical16` (16 GiB), `capacity64` (64 GiB) and `workstation90` (the resolved90% ceiling). The latter is the operator default; the others are controlled comparisons. If64 GiB exceeds the authorized/effective maximum, that capacity trial is unavailable. Existing `--memory-gib` may request a smaller explicit profile; it cannot exceed the authorized ceiling or be presented as the same resource contract. A lower-resource diagnostic does not silently replace target qualification.

Output limits: 8 MiB per log stream with a bounded tail/omission counter; 8 GiB transient invocation artifacts excluding read-only inputs and engine-owned images; 128 MiB optional traces total; 512 MiB combined ZIP. Reserve at least 16 MiB for core summary/receipt/compact job records and calculate a larger mandatory allowance if the selected manifest needs it. Reject caps that cannot hold mandatory evidence before executing expensive work. Optional traces/examples are omitted first, with exact reasons. Build/image storage is separately reported and free-space checked; engine-owned caches are never automatically deleted to recover space.

Emit a phase/stage start and completion message, log path, elapsed time and remaining global allowance. Heartbeat every 30 seconds while running, satisfying the specification's maximum 60-second interval. User interruption must reach the supervisor during builds and container startup as well as during model work.

## Receipt reuse and failures

Every stage has an immutable dependency identity; see [data-model.md](../data-model.md). Reuse only PASS receipts with complete mandatory artifacts, matching stage version and prerequisite receipt identities. Timing/admission receipts also bind hardware, engine, resources, workload and measurement policy. A changed decoder invalidates inference/overlap/timing descendants; changed native arithmetic/fitting invalidates semantic/native/admission descendants. Unknown closure changes invalidate conservatively. Current installed-package/source inventory always runs even when an expensive pure test receipt is reusable.

A documentation-only change need not invalidate runtime timings, but the report must record full current source plus the exact tested closure and previous producer identity. No claim that an old binary ran the new source. Native/guest/package locks, workload manifests and validation adapters are explicit closure inputs; filenames alone are insufficient.

A rerun creates a new invocation and starts a failed bounded stage afresh. Successful prerequisites can be imported; previous failures, output identities and costs remain linked. Do not checkpoint an arbitrary half-finished test or restore a diagnostic optimizer as a learning run. A comparison interrupted after three of five pairs is PARTIAL; the default rerun restarts its whole confirmation block rather than selecting favorable old pairs.

Stage states: `PENDING → READY → RUNNING → PASS|FAIL|PARTIAL|INCONCLUSIVE`; `PENDING|READY → BLOCKED`; compatible imports use `REUSED` with the original PASS receipt hash. `NOT_SELECTED`/`UNAVAILABLE` describe the registry scope, not successful execution. A stopped stage never returns to PENDING inside the same ledger.

Top-level process exit: 0 for PASS of the explicitly reported selected scope, 1 for FAIL, 2 for invalid invocation, 3 for BLOCKED/PARTIAL/INCONCLUSIVE, 130 for user interruption. A report-generation error cannot erase the prior failure. Different research/feature gates remain separate fields, not an overloaded single boolean.

## Packaging and privacy

One archive contains `summary.json`, `manifest.json`, resolved public contract, `stages/<id>/receipt.json`, compact per-job/per-prompt outcomes, scope/qualification decisions, file hashes, omissions and bounded useful logs. Use the current role audits: network disabled for execution, read-only inputs/root, output-only writes, no runtime socket, requested device only, CPU scorer with no accelerator visibility. Build/data preparation network access remains confined to existing explicit preparation operations; validation never silently fetches a new evaluation cohort.

Before packaging, verify all mandatory files are regular, confined and readable by the operator, including arbitrary-UID/rootless outputs. Fail with exact paths when unreadable; do not silently omit mandatory evidence. Exclude weights, private/final truth, full input payloads, caches and runtime sockets. Hash-check archive members and sizes, preserve source-bound provenance and create one top-level ZIP rather than nested phase ZIPs.

If disk failure prevents a ZIP, leave the smallest possible readable status/manifest and print surviving paths plus the precise write failure. SIGINT, build failure and unavailable engine must still take the reporting path. Host process death beyond supervisor control can leave an incomplete ledger; the next read must classify it as interrupted, never infer PASS from missing terminal records.
