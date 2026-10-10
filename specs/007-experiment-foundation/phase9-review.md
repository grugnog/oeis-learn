# Spec 007 phase 9 review and remediation

Base: `933a180` (phase 8 merged). Scope: T080–T092; later release/experiment phases unchanged.

## Findings and repairs

1. The baseline lacked logical IR, full wide arithmetic, bounded arrays, streaming and richer certificate modules despite historical partial marks. Added a separate typed logical WAT profile and decimal codec. Native limb-stack WAT remains a separate control. Arithmetic is checked signed256; literal logical shifts are 0–255, native shifts retain their existing semantics. Trusted lowering emits no imports and exactly one fixed 64-KiB page. Python-integer expectations do not use helpers or production limb algorithms.
2. Full products retain all 512 product bits. Division uses unsigned long division plus independently checked sign/range handling. Division/remainder reject zero and min/-1. Checked casts, comparisons, bit operations and distinct right shifts have boundary fixtures.
3. Streams commit only successful next transitions. Failed transitions roll back index, slots and arrays. Checkpoint restoration verifies history and charges replay; a durable single-writer controller journals issued work before execution. Orphaned work conservatively spends the remaining allowance. Per-next and aggregate fuel/reference-step caps remain separate, under an external process deadline.
4. Retrieval previously returned the first traversal matches and allocated no honest search accounting. It now uses bounded neighborhood pairs, blockwise distances and globally ranked retained proposals with deterministic IDs. Scalar one-index PSLQ is replaced by exact multi-index matrix/nullspace fitting, primitive coefficients and bounded basis search; incomplete search is UNKNOWN. Approximate-real PSLQ requires an explicit separate contract.
5. Numerical validation accepted empty/overlapping partitions and published fabricated rank/nullity/source metadata. It now rejects malformed partitions/zip lengths, checks actual transformed indices, computes exact rank/nullity and real source identities, and leaves unproved minimal support null.
6. Added typed structural/dataflow indexing over verified training archives, bounded transforms, and a program-provenance discovery entry point. Generation genealogy remains attached to each implementation; finite fingerprints do not prove equivalence. Legacy embedding discovery is explicitly diagnostic and does not establish program provenance.
7. Added exact polynomial extraction, full linear-state recognition/Cayley-Hamilton checking, reachable-state closure, restricted-loop SMT obligations and tiny-domain compiled/reference enumeration. A fresh worker replays each certificate without weights. Domain, coefficients, sources/IR, assumptions and compiler identities are bound; compiler linkage is **finite conformance only**, never a universal WASM theorem. Each certificate carries independently replayed compiled/reference observations for its own programs, including successful full state or matched trap classification; a mathematical domain outside machine indices explicitly has unavailable machine conformance. Unsupported general programs remain explicit.
8. Added typed macro sharing with identical expansion, acyclic dependencies and positive description-length benefit. Value-guided source changes remain separately labelled hypotheses with bounded cost and training-only targets; no equivalence is inferred from their scores. Activation of conditional measured pilots belongs to the later experiment protocol.
9. External knowledge lookup is analysis-only after immutable proposal sealing. Reports distinguish rediscovery, new implementation and novelty_unknown; finite/certified relation status remains separate. No external-knowledge argument exists in the strict program-discovery worker.
10. Added G9 qualification and CLI replay/discovery commands. A missing native capability is unavailable/pending, distinct from a failed mandatory case. Current complete source/build-bound G9 evidence is required for discovery activation.

## Validation and current status

The initial eight discovery regressions failed on the baseline. The review now covers all thirteen phase-9 software tasks, with owning legacy expectations reconciled to exact finite verification and disjoint partitions. Duplicate operands combine before canonical identity; incomplete numerical evidence carries null rank/source/PSLQ metadata. Actual pipeline output is checked against the updated owning report contract.

CPU CI passed the full foundation suite (643 passed, 51 documented capability skips on the reviewed code). G9 requires every mandatory case to run without skips. Root Docker qualification passed G7, G8 and G9 at `9ba3be2`, including 47 logical Python, 37 discovery/certificate/CLI, 16 owning discovery and 32 native cases. The last report/semantics review adds three owning cases, bringing G9 to 135 required tests. Each code update is requalified before this PR is made ready; the final source-bound reports and public CLI acceptance are published in the [PR checks](https://github.com/grugnog/oeis-learn/pull/10) and workflow artifacts.

This workspace lacks Docker, the native toolchain and `lab-gpu`. Broad local accounting tests also require an unavailable Linux boot identity; those cases pass in CI. The local public qualification command emits clean JSON, passes every CPU G9 suite and reports native capability as unavailable/PENDING. Native evidence comes from the root Docker image, rather than a local skip. No GPU is necessary for phase 9. Earlier HIP and production-container partial markers are preserved, and phase 10 is unchanged.

Reproduction:

```sh
python -m pytest -q tests/unit/test_foundation_extended_wat.py tests/integration/test_foundation_discovery.py tests/contract/test_foundation_discovery_cli.py
python -m oeis_learn.evaluation.discovery_gate --native --output /evidence/discovery
python specs/007-experiment-foundation/validate_artifacts.py
```

For native qualification, build `Dockerfile --target qualification` and run G9 **inside that same image**. A copied PASS report cannot authorize a different source/build/runtime. CPU fixtures do not authorize accelerated learning or establish throughput.

Further review added external deadlines for exact fitting and certificate construction/template proposal, a bounded driver-index domain for finite-state certificates, checked false-guard arithmetic in loop exit obligations, independent macro expression types, and nonoverlapping description-length accounting. Indexing, macro mining and learned-value targets reject historical execution evidence after runtime/helper changes until independent readmission. Structural retrieval now proposes ranked two-program relations as well as numerical triples; actual archive tests retain both generated and model-hypothesis implementations, their separate genealogy, and exact finite evidence.

## Capability limits

Polynomial extraction supports straight-line exact integer polynomial operations, including exactly divisible polynomial division with declared intermediate poles. Other integer division/rational/array/loop shapes are unsupported by that extractor. Recurrence recognition supports affine slot-only transitions and exact initial states at origin zero. Finite-state certificates require index-independent streams and closure within 10,000 full states. Restricted-loop certificates support one slot-assignment loop with exact affine/polynomial invariants and ranking obligations. General-purpose proof and formal compiler verification are not claimed. Bounded machine evidence remains finite.
