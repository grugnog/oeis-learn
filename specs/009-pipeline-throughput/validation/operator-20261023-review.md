# First operator archive: review and repair handoff

The uploaded `009-20261023T232536Z.zip` is intact and matches the clean producer commit `f7ecc720f5cada14820add0d74d099b6d3a038e4` exactly. Its full source snapshot is `sha256:a36ac23965bf07da59b02c3d32dd33110d1fd4c767434b2578d14d154ef5aec1`. The canonical manifest, every listed member hash/size and terminal summary passed verification. The 507-member, 9,688,496-byte ZIP ran for 1,765.05 seconds; this failed scope's 29m25s is not a prediction for a complete run. [Machine-readable review](phase-1-operator.json) retains archive and stage identities. The original ZIP is unchanged and is not committed.

| Observed gate | Producer result | Review |
| --- | --- | --- |
| Podman CPU semantics | 593 tests, zero failures/errors/skips | Historical source PASS |
| Actual HIP preflight | 14 tests, zero failures/errors/skips | Historical source PASS; no CPU fallback |
| Podman image, retirement and supported workflow | PASS | Historical source evidence retained |
| Calibration and diverse32 cost | FAIL | Diagnostic checkpoint compared GPU originals with deliberately CPU-loaded tensors |
| Transfer cost | FAIL | A local native-service expiry aborted the bounded diagnostic |
| Native paired comparison | INCONCLUSIVE, selected workers=1 | Two serial arms incomplete; no speedup/adoption claim |
| Docker | PARTIAL; descendants blocked | Resource facts unavailable; raw Docker info absent, so its cause is not established |
| Overall | selected scope FAIL, engineering BLOCKED, portability PENDING | Feature incomplete; learning/research/long-run authority absent |

The Podman workstation exposed 32 CPUs and 134,161,076,224 physical RAM bytes. The invocation resolved 16 CPUs and a 120,744,574,976-byte memory ceiling (90% of measured physical, rounded down). This does not qualify a separate Docker engine.

## Repairs and available verification

1. Diagnostic checkpoint equality now compares dtype, shape and exact CPU tensor values. `CheckpointStore.load` keeps its CPU loading behavior and the source model/snapshot keeps its device. New fixtures exercise CPU and actual HIP with an optimizer update and the real checkpoint store. They are registered with the existing pinned-runtime suites; they have not run in this workspace because Torch is unavailable.
2. Sequential and overlap owners retain explicit incomplete results for positively completed/queued local deadline expirations. Late candidate data is discarded; an unstarted job has no debit and started work is charged once. Other prompts retain their original parent allowances. An unacknowledged running task still requires cancellation/escalation and quiescence. Callers without an explicit safe adapter still raise. Deterministic owner regressions pass.
3. A completed inconclusive native investigation now explicitly retains the unchanged serial component so it does not itself block independent joint work. Its sealed report and original outcome remain visible; failed correctness does not become a substitute. Experimental workers require complete comparison and qualified resource evidence; joint correctness/resource/five-pair gates remain mandatory. Docker probes now explain missing memory facts or a Podman-shaped compatibility response without inventing resources or qualifying two engines.

The real public `python3 -S` all-ready/both-engine invocation passes **85 stdlib tests** and verifies its 54 manifest-listed archive members. It exits 3 with unavailable engines, selected scope PARTIAL, engineering BLOCKED and portability PENDING. [Current local evidence](operator-repair-local.json) records its exact source/archive identities. Python compilation and `git diff --check` pass. Current Torch/native/engine/HIP repair validation is pending. The review binding was refreshed while preserving all 68 dispositions and incomplete acceptance.

There is a separate honest completeness limit: the producer's cold baseline scheduled 576 attempts but decoded only 376, hit generation deadlines on 32 and left 168 unstarted at the parent deadline. **Zero of 32 prompts completed the whole frozen workload.** After the checkpoint repair, that observation can yield INCONCLUSIVE calibration and leave dependent inference/overlap comparisons unstarted. Workload membership, attempts, policy limits and phase budgets have not been expanded or reduced in response. No partial arm is promoted to equivalent completed work. Cost reports still require complete exact-scope controls.

Both CI workflows at the producer `f7ecc720` completed successfully. This is historical producer evidence, not validation of the repairs. New CI is not awaited. Task counts remain 67 implementation/local tasks complete, 29 runtime/evidence tasks pending and 15 research tasks blocked; this upload cannot close the full operator tasks or current-source acceptance. Original008:T093/T096 and research decisions remain unchanged.

## One target rerun

After updating `009-pipeline-throughput`, run from the repository root on the HIP/ROCm workstation:

```bash
python3 scripts/foundation_run.py --engine podman --cpus 16 validate \
  --suite pipeline --phase all-ready --engines selected \
  --manifest configs/foundation/pipeline_validation.json \
  --run tmp-validation/008-phase5-diverse32-20261021T160815Z/learnability-run \
  --inputs tmp-validation/008-phase5-hip-20261020T202907Z/inputs \
  --max-seconds 86400 \
  --output "tmp-validation/009-$(date -u +%Y%m%dT%H%M%SZ)"
```

The existing input paths are retained; substitute their workspace-relative locations if moved. Return the single sibling ZIP, including a non-PASS result. `--engines selected` collects remaining Podman checks while independent Docker is unavailable; portability remains PENDING. Use `--engines both` when an independent Docker engine with measurable resource facts is available. No separate manual build/test/profile/archive sequence is needed. This command includes the new CPU/HIP checkpoint fixtures.

The hard cap remains 24 hours; complete current-source duration is unknown. A fresh output is required. Optional `--reuse` accepts only compatible completed evidence; changed source closures invalidate affected stages, image inspection and timing rerun, so prior skips are not promised. No research study or final exposure is launched by this engineering command.
