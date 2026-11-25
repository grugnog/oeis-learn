# Implementation Plan: Remove Unused Repository Content

**Branch** `014-repository-cleanup`

**Date** 2026-11-24

**Input** [Feature specification](spec.md)

**Stack** PR #73 targets `013-progressive-mathematical-discovery`. The audited runtime baseline is `a2618bbd` on PR #72.

## Summary

Start with the checked-in HPO outputs and old E2E report. They total 31,222 bytes, sit outside the qualification source map and image context, and have identifiable producers. Review each group against its remaining readers. Hold the 309-line retired-teacher method and the old HPO launcher. Removing either causes the previously passing round009 donor check to reject its source path. The output groups can proceed independently.

Trace who reads, executes, packages or authenticates each candidate. Compare affected behavior on its parent and on the proposed change. Keep each group independently reversible. Document an entangled obsolete boundary in a surviving tracked guide or index outside authenticated files. Missing evidence holds the affected group. [Candidate evidence](research.md) · [Acceptance contract](contracts/cleanup-acceptance.md)

## Technical Context

| Topic | Planning baseline |
| --- | --- |
| Project and platform | Python 3.11+ package and portable host CLI, Rust native crates, Linux Docker/Podman, CPU/native `qualification` and ROCm `gpu` targets. The host launcher uses the standard library. Packaging and toolchains use the existing locks and Dockerfile. |
| Research and storage | Existing spec 013 code, spec 012 checkpoints and original producers, task leases, budget journals, byte/inode limits, full-state checkpoints and public verification bundles. Actual workstation data stays outside Git and outside the deletion set. |
| Scope | 2,037 tracked baseline paths, including 612 under `src`, 514 under `tests`, 581 under earlier specs, 50 under `scripts` and 44 under `crates`. Inspect both files and symbols, with explicit coverage of config, docs, CI, packaging, examples and checked-in output. |
| Analysis and testing | Git tree/text/blob queries, Python AST, Grimp import edges, Vulture leads, real callers and `cProfile`, bounded pytest fixtures, clean vs source-distribution wheel comparison. Grimp and Vulture are disposable reviewer tools. No new runtime dependency or analysis service is proposed. |
| Acceptance scale | Stop the round027 rehearsal after authenticating 47 outputs and four checkpoints and recording the comparison. For publication, use four generator outputs spanning the observed 45.5–52.2 MB each, their four matching scores, roughly 49 MB of calibration and roughly 290,000 journal events. Measure original bytes, the encoded ZIP entries before compression and the final ZIP separately. The campaign keeps its frozen limits. |

## Constitution Check

Reviewed against the adopted constitution v2.0.0 before research and again after designing the contracts. [Project constitution](../../.specify/memory/constitution.md)

| Principle | Before research | Post-design gate |
| --- | --- | --- |
| I–II. Exact semantics and independent acceptance | The change concerns retirement. Integer semantics, language profiles, independent oracles and validation still have consumers. | If a group touches an oracle, grammar, policy or native dependency, compare its existing positive and negative conformance cases. Retire a test only with the exclusively retired behavior it covered. |
| III. Bounded, recoverable operation | Ownership, time, byte and inode limits and journals are treated as inputs. | Interrupted and repeated recovery retain original charges and full checkpoint state. Test the affected bound at realistic size and retain a failing boundary case. |
| IV. Reproducibility and actual-device qualification | Treat qualified source, package, images and device as distinct identities. | Compare checkout and installed resources. Preserve the pinned accepted checkout. Record a supported prospective transition for identity-changing groups and the actual device evidence required for that transition. |
| V–VI. Provenance, isolation and scoped claims | Reuse original spec 012/013 producers with private evaluation inputs isolated. | Locked selection gates assessment. An independent verifier must reject altered evidence and preserve incomplete verdicts. Diagnostic fixtures give software evidence only. |

The design passed the pre- and post-design constitution reviews. The independent review at `83684256` reproduced the saved receipt from the original round027 ZIP, which contained 352 accounting rows. The receipt-backed private recovery rehearsal, combined full-size publication and physical HIP qualification remain open. The baseline wheel packaging difference is recorded separately. [Review evidence](research.md#publication-review-probes)

## Project Structure

### Documentation for this feature

| Artifact | Purpose |
| --- | --- |
| [research.md](research.md) | Sources, measured tool output, manual call paths, historical controls and first candidate dispositions. |
| [data-model.md](data-model.md) | Minimal review record with consumer evidence, source identity, disposition and reversible group boundaries. |
| [contracts/cleanup-acceptance.md](contracts/cleanup-acceptance.md) | Observable contracts at the operator, installed, recovery and verifier boundaries. |
| [quickstart.md](quickstart.md) | Repeatable baseline checks, candidate comparisons, known differences and representative-data acceptance. |

### Source code and assets at the repository root

| Area | What to trace |
| --- | --- |
| `src/oeis_learn/`, `scripts/`, `tools/` | Public exports, CLI parsers and defaults, legacy aliases, subprocesses, string- and filename-based dispatch, diagnostics and saved-state selectors. Follow a symbol through its consumer even when the consumer sits outside the package. |
| `crates/`, `docker/dev/`, root Dockerfile/locks, `.github/` | Native identity inputs, build scripts, optional targets, CI path/partition selection, isolated installed-wheel behavior and both engines’ existing contracts. |
| `configs/`, packaged policies/schemas, `tests/`, `data/` | Config-selected branches and resources, globs, negative controls, fixtures and CI paths. Equal bytes at two locations can satisfy different contracts. |
| `docs/`, `specs/`, `.specify/`, `.agents/`, `.pi/`, `reports/`, `runs/` | Current operator and maintainer entry points, manifests, evidence cited by a current claim and historical references. Distinguish a checked-in example from live state and a directory reader from a reader of a particular file. |

## Implementation sequence

1. **Baseline the group.** Start from the full tree, record its parent SHA, inventory the affected roots, and find direct, indirect, external/operator and identity consumers. Use exact paths, basenames, symbols, digests, wildcards and package aliases. For each potential deletion record a finding or an explicit unanswered question. Existing spec 013 paths in the spec are positive controls for the search.
2. **Review generated outputs first.** Trial the two HPO outputs as one change and the E2E report as another. Compare surviving producer defaults, references, resource maps, public claim handling and relevant tests. Show counts/bytes and the exact removed paths. Keep the still-executable generators and controls available to their consumers.
3. **Check each source change against a passing donor control.** Derive the expected source-map changes from the group’s parent and tip. Validate the original round009 donor against clean qualified `77667e67`, then apply only that group’s patch to a qualified copy with its own Git root. Require the observed changed keys to match the expected set before rerunning donor validation. Record the exact rejection or the accepted result. The reporter at `a2618bbd` already fails on `budgets.py`, which masks the independent `rl/elite_buffer.py` rejection when removing `_populate_canonical_defaults`. Hold a group that fails, cannot be applied faithfully, or lacks transition evidence for required reuse. Keep the caller and installed-behavior comparisons for every affected interface. [Donor procedure](quickstart.md#test-each-source-changing-group-on-the-passing-qualified-control)
4. **Sweep remaining areas by dependency.** Review scripts with their exclusive assets, old guides with their incoming links, tracked runs with provenance consumers, and native/build/test assets with their actual installation or execution paths. A receipt referring to a Git pin can remain reproducible through Git. An active pathname or recorded hash required by a supported invocation needs a working resolution. Summarize areas examined, candidates found and actual disposition. History and deletion quotas do not determine scope.
5. **Validate and integrate.** Run each group’s checks and the four operator demonstrations on the combined candidate. Freeze a one-task CPU fixture, verify the first census receives its frozen inputs, and check the rejection cases have zero dispatches. For SC-003, guard both recovery attempts and exercise the proposal and the full-state child on separate bounded fixtures. Record the genuine private replay independently. For current reports, compare the baseline and all three negative cases at the observed campaign sizes. Verify the immutable earlier ZIP separately. Use one reversible commit per group and repeat its checks after rollback. [Acceptance procedures](quickstart.md#5-close-the-affected-evidence-gaps)

| Spec coverage | Where it is exercised |
| --- | --- |
| FR-001–004, SC-001/006 | Review records, candidate groups, reference and packaging comparisons, selective rollback. |
| FR-005–007, SC-002/003/006 | Fresh gates, installed interfaces, accepted replay, continuation, screening and separate final assessment. |
| FR-008, SC-002/004 | Read-only progress snapshots and waiting/terminal collector behavior. |
| FR-009, SC-002/005 | Candidate verification of a previously published ZIP, current full-size publication and verification, altered/incomplete controls and the encoded-entry package boundary. |
