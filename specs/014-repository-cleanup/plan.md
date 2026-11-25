# Implementation Plan: Remove Unused Repository Content

**Branch** `014-repository-cleanup`

**Date** 2026-11-24

**Input** [Feature specification](spec.md)

**Stack** PR #73 targets `013-progressive-mathematical-discovery`. The audited runtime baseline is `a2618bbd` on PR #72.

## Summary

Start with the checked-in HPO outputs and old E2E report. They total 31,222 bytes, sit outside the qualification source map and image context, and have identifiable producers. Review each group against its remaining readers. The best source candidate is a 309-line private method for seeding a retired teacher. Its class still serves the installed application. Removing the whole class or the old HPO launcher would affect live contracts.

Use the same rule for the rest of the tree. Trace who reads, executes, packages or authenticates each candidate. Compare affected behavior on its parent and on the proposed change. Keep independently reversible groups small. A named dependency can justify an external deprecation note, and a missing piece of evidence can hold one group while another ships. [Candidate evidence](research.md) · [Acceptance contract](contracts/cleanup-acceptance.md)

## Technical Context

| Topic | Planning baseline |
| --- | --- |
| Project and platform | Python 3.11+ package and portable host CLI, Rust native crates, Linux Docker/Podman, CPU/native `qualification` and ROCm `gpu` targets. The host launcher uses the standard library. Packaging and toolchains use the existing locks and Dockerfile. |
| Research and storage | Existing spec 013 code, spec 012 checkpoints and original producers, task leases, budget journals, byte/inode limits, full-state checkpoints and public verification bundles. Actual workstation data stays outside Git and outside the deletion set. |
| Scope | 2,037 tracked baseline paths, including 612 under `src`, 514 under `tests`, 581 under earlier specs, 50 under `scripts` and 44 under `crates`. Inspect both files and symbols, with explicit coverage of config, docs, CI, packaging, examples and checked-in output. |
| Analysis and testing | Git tree/text/blob queries, Python AST, Grimp import edges, Vulture leads, real callers and `cProfile`, bounded pytest fixtures, clean vs source-distribution wheel comparison. Grimp and Vulture are disposable reviewer tools. No new runtime dependency or analysis service is proposed. |
| Acceptance scale | Preserve all four operator workflows, including the 47-result/four-checkpoint interruption and reporting with a roughly 49 MB calibration input, four large comparison projections and roughly 290,000 journal events. Keep the campaign’s frozen numerical and resource limits. |

## Constitution Check

Reviewed against the adopted constitution v2.0.0 before research and again after designing the contracts. [Project constitution](../../.specify/memory/constitution.md)

| Principle | Before research | Post-design gate |
| --- | --- | --- |
| I–II. Exact semantics and independent acceptance | The change concerns retirement. Integer semantics, language profiles, independent oracles and validation still have consumers. | If a group touches an oracle, grammar, policy or native dependency, compare its existing positive and negative conformance cases. Retire a test only with the exclusively retired behavior it covered. |
| III. Bounded, recoverable operation | Ownership, time, byte and inode limits and journals are treated as inputs. | Interrupted and repeated recovery retain original charges and full checkpoint state. Test the affected bound at realistic size and retain a failing boundary case. |
| IV. Reproducibility and actual-device qualification | Treat qualified source, package, images and device as distinct identities. | Compare checkout and installed resources. Preserve the pinned accepted checkout. Record a supported prospective transition for identity-changing groups and the actual device evidence required for that transition. |
| V–VI. Provenance, isolation and scoped claims | Reuse original spec 012/013 producers with private evaluation inputs isolated. | Locked selection gates assessment. An independent verifier must reject altered evidence and preserve incomplete verdicts. Diagnostic fixtures give software evidence only. |

The design passes both planning reviews. The outstanding baseline packaging difference, private replay and hardware checks have named owners in the validation contract. A planning pass records the design’s compatibility with the constitution. Runtime evidence remains attached to the revision and environment that produced it.

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
3. **Review isolated symbols and defaults.** Trial removal of `_populate_canonical_defaults` in a disposable candidate checkout. Compare the rejection, empty-buffer and public canary behaviors, the source map and the donor-equivalence result. Apply the same process to other private symbols, configs or duplicate instructions. A public default, an authenticated helper or a missing transition moves the affected item to hold or documented deprecation.
4. **Sweep remaining areas by dependency.** Review scripts with their exclusive assets, old guides with their incoming links, tracked runs with provenance consumers, and native/build/test assets with their actual installation or execution paths. A receipt referring to a Git pin can remain reproducible through Git. An active pathname or recorded hash required by a supported invocation needs a working resolution. Summarize areas examined, candidates found and actual disposition. History and deletion quotas do not determine scope.
5. **Validate and integrate.** Run the group-specific baseline comparison and the independent protected workflow checks for the resulting cleanup. Compare live contracts, costs, version dispatch, verdicts and any required source transition. Save evidence outside authenticated inputs. Use one revertible commit per independent group. If a regression appears, revert only its group and re-run the same witnesses. Independent groups retain their own evidence.

| Spec coverage | Where it is exercised |
| --- | --- |
| FR-001–004, SC-001/006 | Review records, candidate groups, reference and packaging comparisons, selective rollback. |
| FR-005–007, SC-002/003/006 | Fresh gates, installed interfaces, accepted replay, continuation, screening and separate final assessment. |
| FR-008, SC-002/004 | Read-only progress snapshots and waiting/terminal collector behavior. |
| FR-009, SC-002/005 | Public producer provenance, independent verdicts, incomplete/altered controls and campaign-sized verification. |
