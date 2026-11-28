# Implementation Plan — Learning velocity on new sequence families

**Feature branch** `016-measure-learning-velocity` · **Date** 2026-11-27 · **Status** Phase 1 design complete  
**Input** [Feature specification](spec.md) · **Source baseline** `4a13ec34560e7cbeffcd2dad2b24a25b07ee0dd5`

## Summary

Build round029 as a versioned study on the existing discovery campaign. The first deliverable is a preflight that tells the operator which round028 ordinary checkpoints can be reused, how many eligible families and training pairs exist, and what one matched probe and its follow-up would cost. The research comparison then measures gains in exact solves per million useful tokens and per charged workstation hour. Previously unseen real OEIS families are the transfer endpoint. D remains the direct operational reference.

Two pieces of work can proceed independently after their inputs are ready. The seven-recipe screen needs a batch-aware, recoverable learner and the same 0/1M/4M probe at every look. The CPU study first tests the fitter with a supplied structure, then tests whether ordering legal constructions improves the single selected answer. Both use the same rules for visible input, exact admission, sealed scoring and recorded cost. The launcher follows only the branches registered and funded before their scores. [Experiment contract](contracts/experiment.md) · [CPU search contract](contracts/search.md)

## Technical Context

| Area | Design |
| --- | --- |
| Runtime and dependencies | Python ≥3.11 standard-library host, installed Python workers from `src/`, locked PyTorch learner, qualified native Rust bigint runtime/synthesis service and pinned Z3 fitter. Keep Optuna 4.5.0’s existing trial semantics. Build from the root Dockerfile and committed locks. |
| Platform | Existing Linux operator interface with explicit Docker/Podman engine. The research target is the Ryzen AI Max+ 395, 128 GB shared RAM, NixOS, Podman/HIP workstation. Use CPU for preparation, diagnostic search and independent scoring, and the existing lease for each device operation. |
| Persistent state | One physical workspace and 029 session. Content-addressed input projections, full checkpoints and receipts are private. Publish aggregate report and pre-score qualification ZIPs for an archive-only reader. |
| Scale and opportunity | Seven recipes share a control across two 2×2 comparisons. Six new 4M arms imply about 24M useful tokens before overshoot if ordinary is reusable. The full comparison proposes 128 development and 32 retention cases per method, with fresh reservations for confirmation. D is evaluated at matched inference cost. |
| Performance and uncertainty | Measure verified family yield per additional million useful tokens and per additional charged hour. Record wall to decision, paid history, marginal cost, GPU lease/activity and CPU work. Price each new workload on the physical machine. A late learner can earn a paired 8M continuation. |
| Validation | Existing unit, contract, integration, installed-wheel and detached-donor entry points. Add targeted tests for false matching labels, exposure/clock reuse, CPU search attribution, recovery, independent population claims and archive replay. |

## Constitution Check

The adopted [constitution v2.0.0](../../.specify/memory/constitution.md) applies. At entry to Phase 0, the spec selected the strict track, existing exact profile, isolated scoring and recoverable resource accounting. The Phase 1 review found a design path for all six principles without an amendment. Evidence required before the corresponding production dispatch is listed below.

| Principle | Phase 1 design and required evidence |
| --- | --- |
| I. Exact values and numerical semantics | Authenticate indexed integer text, codec, precision and bigint resource semantics. Qualify finite gradients for each changed objective and independently check final-source arithmetic at nonzero/negative offsets. |
| II. Supported language and independent acceptance | Typed search emits only the declared grammar. Round-trip sources, verify legal grounding and replay exact continuations independently. Exercise a false negative that can be regrounded and a real full-scope negative. |
| III. Bounded and recoverable execution | Keep the registered service and drain rules. Audit hard guards by owner, measured peak and margin. Exercise real exhaustion, lost worker, missing telemetry and two identical resumes without refreshing allowances. |
| IV. Workstation reproducibility | Pin controller, worker, image and grant. Demonstrate actual HIP parameter updates, full restore and equivalence after interrupted backward. Measure storage, lease and actual time on the donor machine. |
| V. Strict provenance and isolated generalization | Audit linked families and all historical reservations before scoring. New seeds start from eligible random initialization. Separate training, teachers, privileged oracle, live roles and private suffixes. Test both feature channels and exported archives for leakage. |
| VI. Scoped scientific claims | Preserve all frozen denominators, unresolved bounds, conditional uncertainty and independent replication. Report finite 80-term agreement. Real OEIS adoption requires a replicated direct comparison with D. Final assessment has separate authority. |

## Project Structure

### Documentation for this feature

| Artifact | Development use |
| --- | --- |
| [Research decisions](research.md) | Phase 0 evidence, rejected alternatives and verified interfaces |
| [Data model](data-model.md) | Identities, ownership, state transitions and public sufficient statistics |
| [Experiment contract](contracts/experiment.md) | Seed/donor reuse, seven recipes, matched curves, spending and formal confirmation |
| [Search contract](contracts/search.md) | Privileged structure recovery, fixed/guided/shuffled controls and complete-bank evidence |
| [Operator contract](contracts/operations.md) | CLI, grant scopes, measured limits, single-root recovery and independent readers |
| [Quickstart](quickstart.md) | Current baseline checks, pinned future start/resume invocation and end-to-end acceptance scenarios |

### Source code and integration boundaries

Paths labeled **add** are proposed. All other paths identify code at the source baseline. Extend shared functions narrowly where a versioned request can preserve existing behavior.

| Boundary | Integration |
| --- | --- |
| Study and host | In `pipeline_validation/progressive_cli.py` register `discovery round029`. Dispatch from `progressive_commands.py` to **add** `pipeline_validation/progressive_round029.py`. Extend `generalization/progressive/{campaign,graph,lifecycle,trials}.py` with round029 registration, eligible nodes and observations. Preserve older study policies and sampler feedback. Source paths in this table are under `src/oeis_learn/`. |
| Preparation and exposures | Reuse `pipeline_validation/{progressive_research_inputs,progressive_preparation}.py` and `generalization/progressive/{generator,prepared_panels,program_search_panels,training_rows}.py`. Add a versioned panel allocator and keyed sidecars for training-pair proofs and composition selection. |
| Learning | **Add** `generalization/progressive/velocity_trials.py` for recipe/seed and useful-milestone compatibility, and `pipeline_validation/progressive_velocity_worker.py` for typed role requests. Reuse the canonical [supervised logical update](../../src/oeis_learn/rl/sft_trainer.py), `progressive_training.PolicyUpdate` orchestration where compatible, `generalization/checkpoints.py` and interrupted-update accounting. Add batch16, optional matching and an explicitly bridged 8M horizon through a 029 policy. Preserve old hard-coded 028 validators on old request versions. |
| Search and role isolation | **Add** `generalization/progressive/family_search.py` for typed constructions, guide features and action ordering. Reuse `synthesis/{fitting,admission,evidence,replay,reference}.py`, `decoder/bigint_codec.py`, portfolio admission/selection in `fresh/bigint_diagnostics.py` and private bank replay. Register versioned 029 operation/channel identities. Keep the role-policy digest verifiable for earlier archives. |
| Metrics and publication | **Add** `generalization/progressive/velocity_metrics.py` for joint transition counts, cost axes, factorial contrasts and branch selection, and `velocity_report.py` for public bundle recomputation. Reuse `metrics.compare` and `decisions.recompute` for their supported yield claims. Reuse `budgets.BudgetJournal`, `freeze_progress.FreezeProgress` and the historical staged-publication pattern. |
| Profile, launcher and distributions | **Add** `configs/foundation/progressive/learning-velocity.json` and `specs/016-measure-learning-velocity/validation/launch-round029.sh`. Bind the operator helper/source cache to its full Git pin. Extend `setup.py`, `MANIFEST.in` and `src/oeis_learn/experiments/source_snapshot.py` to bind and ship 016 contracts/profile in source and wheel, plus any new runtime schemas. Authenticate host-only helpers and prospective grants separately to avoid self-referential hashes. |

## Implementation sequence and acceptance

| Delivery slice | Depends on / first observable result | Regression or release gate |
| --- | --- | --- |
| 1. Reuse and affordability (FR-001–003, FR-015–016) | Read-only donor inventory, exact eligibility per reuse layer, current physical capacity and cost of a matched probe plus reserve. The same trial registration owns all later work. | Exercise a changed score profile, weights-only cut, historical reservation, shifted duplicate and cross-panel composition link. Report precise affected work and a smaller feasible tranche when possible. |
| 2. Comparable learning (FR-004–010) | From the frozen seed/stream, run one ordinary and one changed logical update, reopen the complete cut, then construct the two factorial blocks and explicit cheap/full probe entries. | Check zero/1M/4M on the common roster, 16/32 accounting, withheld row collisions, positive-only exposure, shared-zero reuse and synthetic rate examples. Restore a mid-update kill on real HIP. |
| 3. Diagnose search (FR-010–012) | From disjoint practice data run the oracle first. A separate live CPU role can proceed independently of slice 2. Price D inference and any concurrent resource contention. | Test ambiguous vs unsupported vs witnessed failure, shuffled feature isolation, negative offsets, empty/partial banks, same-bank passed-over solve and every unit of charged work. |
| 4. Price follow-ups and confirm (FR-013–014) | Join only registered evidence. Route a direct OEIS gain even with a flat structural proxy, preserve late learners and condition a paired continuation or fresh independent comparison on the proper grant. | Replay missing outcomes that flip ranks, one influential family, unchanged shuffled performance and retention loss. Two seed pairs use the same untouched panel. D has one qualifying receipt and one cost. |
| 5. Operator delivery (FR-015–018) | One pinned command recovers the earliest eligible boundary, produces a verified report and preserves the first error in a diagnostic. | Two launches contend for one lease. Repeat a resume after staged export. Recompute a negative result and a confirmed result from copied archives, and reject altered parent, count or multiplicity. Exercise actual byte/inode exhaustion and the largest supported report on the candidate installation. |

Run the new tests alongside the current [recovery tests](../../tests/unit/test_progressive_recovery_transactions.py), [native-fit conformance](../../tests/contract/test_native_fitting.py), [metrics tests](../../tests/unit/test_progressive_metrics.py) and [packaging checks](../../tests/contract/test_progressive_packaging.py). Qualify the actual CPU/native and HIP workers and reproduce archive-only verification outside the checkout before long dispatch. The scale measurements and hard-guard policy are in the [operator contract](contracts/operations.md).

## Planning outcome

The design resolves seed reuse, variable batch sizes, admissible matching labels, search controls, velocity accounting, formal confirmation and recovery. Current private checkpoint availability, matched-pair supply, fresh family counts, safe resource margins and funded hours are preflight outputs. The implementation release must supply its actual full Git SHA. Phase 2 task breakdown can proceed from the five slices and their acceptance gates.
