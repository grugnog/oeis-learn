# Existing matched-checkpoint rescue — T093

Status: **DEFERRED by the user, 2026-10-22.** This historical-model comparison
is outside the current Phase5 completion gate. The project continues from
scratch; no conversion/import of historical model weights is required. The
read-only diagnostic and retained-profile compatibility tests remain supported.
No checkpoint evaluation, rescue rate or passing experiment is claimed. T093
stays unchecked with an explicit deferred disposition, rather than PASS.

The user explicitly chose to defer T093 while planning whole-pipeline resource
profiling. This removes the missing historical weights as a Phase5 prerequisite,
including the dependent execution item in T097. It does not waive current-model
correctness, independent fitting admission, provenance or other qualification
gates. A current bigint checkpoint used as a frozen profiling input is a
separate analysis artifact, not completion of T093 or continuation training.

## Historical request (superseded as a current gate)

Status: BLOCKED — this checkout contains no tracked trained `.pt`, `.pth` or
`.ckpt` bytes and no complete matched-run directory was supplied. The earlier
foundation input review also records missing checkpoint bytes. No checkpoint
hash, rescue rate, unsupported fraction or learned continuation result is
invented; arithmetic fixtures are not model evaluation.

Supply a complete retained matched-run directory in this checkout's workspace,
including the actual selected weights, original contract/model/codec/runtime,
input identity and checkpoint manifest. Add `--checkpoint-run runs/ACTUAL_RUN`
to the Phase 5 helper. It invokes the supported read-only
`diagnose --fitting paired --output PATH` child evaluation, leaving parent weights,
contract and ledger unchanged. Original numerical semantics are retained;
unsupported wrapping/scalar/control candidates remain unsupported. The resulting
diagnostic identifies the real checkpoint and reports same-candidate raw/exact
rescue, full continuation, supported fraction and fitting/reference costs.
The new bigint tiny-gate checkpoint cannot substitute for this existing-run gate.
