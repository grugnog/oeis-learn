# Replay viability accepted for its registered scope

`009-replay-20261024T172754Z.zip` passes the actual bundle verifier: all 201 listed members plus the canonical manifest, file hashes/sizes, safe paths and consistent summary. The 3,031,092-byte original ZIP is unchanged. The full source closure matches the clean `c9be16a` checkout exactly: `sha256:69807d38ae62e91406f1d02e9afa284302156ba89cf3d26ebf27ff9f5808a20f`. [Machine-readable review](replay-operator-20261024T172754Z.json) retains source/archive/reference/workload/receipt identities and all trial observations.

The complete invocation took 921.27 seconds (15m21s) including prerequisites/images. Selected scope **engineering-replay PASS**, engineering PASS, portability PENDING, feature incomplete. Learning is NOT_ADMITTED, research NOT_EXECUTED and longer-run admission NOT_ADMITTED. All 13 selected stages are PASS; none was reused, failed or omitted.

## Registered contract checks

The [replay contract](../contracts/replay.md), six-phase plan and original comparison/scheduling rules were used to assess these results. Host tests pass 99 cases. The target CPU suite passes **613 tests** and actual-HIP suite **16 tests**, with zero failures/errors/skips. Both current images pass whole-package inspection. The new real reference/role fixtures pass, including the cache mutation-version guard that rejected the earlier `inference_mode` implementation. No decoder guard or numerical tolerance was weakened.

The CPU reference completes all 12 registered boundaries and seals the expected tensor shape/dtype/hash records before GPU timing. Every GPU role uses that same reference and workload. All six roles retain complete 12-boundary terminal reports, with exactly one PLAN and 12 queued/unverified plus 12 verified events each. All 60 GPU boundary checks pass the registered finite-logit, CPU/HIP tolerance, exact top-one and row/finished/inactive invariants in the source-verified role code. Raw tensors are deliberately absent from the ZIP; this review checks identities/progress and the source-bound role results, rather than recomputing private logits externally.

| GPU role | Verified replay interval (seconds) | Completed boundaries |
| --- | ---: | ---: |
| Cold | 29.6734 | 12/12 |
| Excluded warmup | 29.6845 | 12/12 |
| Calibration 0 | 29.6696 | 12/12 |
| Calibration 1 | 29.7764 | 12/12 |
| Calibration 2 | 29.6049 | 12/12 |

The three calibration intervals have median **29.6696 s**, minimum 29.6049 s and maximum 29.7764 s. The registered rule gives **60 seconds**: `ceil(max(60, 2*29.7764))`. This is a conservative observational reservation for this replay component interval. It is not a statistical p95/confidence statement or a complete container/encoder/generation/solver/scorer/exported-study schedule. Model/role startup and host cleanup remain separately measured; the GPU host role intervals were about 31.31–31.56 seconds. A future scope must separately reserve its external startup/cleanup and work, rather than treating 60 seconds as an all-inclusive pipeline budget.

All roles have qualified aggregate-resource reports and no breach under the 120,744,574,976-byte ceiling. GPU-role conservative peak upper bounds are about 1.78 GB; these are observed bounds with the report's shared-memory accounting caveats, not capacity or allocation recommendations. Every GPU role records two intraop/one interop Torch threads and the same actual HIP/device identity. CPU reference isolation expects no GPU devices; GPU trial audits match the requested render/KFD devices. All role processes complete with exit zero, and reclamation remains within five seconds. The repeated roles are fresh contexts/KV caches, not proof of persistent-cache warming.

Both CI workflows for the producer were already complete and successful when reviewed: Foundation `40114981190`, Container/native `40114981189`. They provide affected-source contract/packaging/engine checks alongside the actual target CPU/HIP run. No CI completion was waited for. CI machine coverage is not merged into this target archive or used to change its immutable portability PENDING result.

## Scope closure and remaining work

Supplemental `009:R005` closes: the registered baseline replay collection/review succeeded. Original T110 closes for affected-source contract/packaging verification of these last code changes. Neither checkbox closes a missing original measurement or research task. The original 111-task counts are now 78 complete, 18 runtime/evidence pending and 15 research blocked; all five supplemental replay items are complete.

The report explicitly denies optimization adoption and research-cost admission. These five single-arm baseline roles are **not five paired confirmations**. No hoisted/native/overlap candidate was timed, no generated-program success denominator was supplied, and no full-training speedup is measured. The original empty 32-prompt fixed-work baseline, incomplete native calibration, full-control costs and no-adoption decision remain unchanged. Docker portability, original collection/adoption gates and all separately admitted research/final/long-run decisions remain open. `008:T093` stays deferred and original `008:T096` stays negative at7730/14400 seconds.

No repeat of this baseline-only run is requested and no runtime code changes are needed from this archive. The next measurement decision can prepare a separately registered baseline-versus-hoisted **component** comparison with exact frozen inputs/reference, both-arm resets, cache/telemetry mode, order, finite role/startup reservations and acceptance rules fixed before candidate timing. Such a comparison still cannot replace free-generation equivalence or whole-pipeline confirmation/cost. It is outside the current baseline-only manifest and was not launched by this review. No new target command or research execution is introduced here.
