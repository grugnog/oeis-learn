**Current implementation/handoff:** [Learning-pilot preflight](learning-pilot-implementation.md) now implements all unblocked preparation code and supplies one finite command/ZIP. Actual larger-corpus/current-source HIP/restart/full-control costs await that collection. T011 closes and new-source T110 reopens; original counts remain87/9/15. The accepted hardware result below remains historical evidence; do not repeat its soak/size ladder.

**Current preparation:** [Sequence-learning pilot plan](learning-pilot-plan.md) and [L001–L015 tasks](../tasks.md#long-run-learning-pilot-preparation) now include actual held-out100-term learning/control curves and cautious further-training estimates. The proposed profile is48 hours inclusive with up to24 learner-role hours, subject to corpus feasibility, implementation and measured cost/admission. A short test validates plumbing without any learning-success threshold. No executable new HIP command or long-run admission exists yet; accepted hardware evidence is preserved.

# Hardware readiness and remaining work

The [accepted primary soak](soak-operator-20261025T232558Z.md) establishes a workable **104.5M-parameter / training-batch-16 / inference-batch-16 / four-native-worker / ordinary-observation** configuration. Keep FP32 math, Torch 2/1 threads, native one thread each, one GPU owner, bounded overlap and drain barriers. This is the largest model tested, not proof that every larger model is infeasible or that filling unused memory improves useful throughput. Keep serial rollback and historical v1 defaults.

## Reuse first

No repeat HIP command is needed now. Reuse the exact-source qualification, full-data calibration, stress/parity, two persistent windows, state recovery, native health, UID/ENOSPC/interruption and isolated scorer evidence. Both work windows ran under the original 7,800-second parent; documentation updates do not change the runtime source. The complete invocation cost was 9,452.064 seconds and the persistent parent cost was 7,213.195 seconds. These measured costs describe this engineering workload only.

Remaining original engineering items are separate scopes:

| Tasks | Remaining evidence or work | Relationship to longer-run preparation |
| --- | --- | --- |
| T011 | Exhaustive archive fault-test catalog: current evidence covers UID, symlink/missing mandatory files, ENOSPC recovery and interruption, but does not establish every unreadable/traversal/capped-detail/contradictory-summary case in the named test task | Complete missing host-side tests against existing implementation; do not repeat the primary soak to close a test-catalog gap |
| T034, T041 | Original frozen baseline and isolated inference comparisons, with material-workload coverage | Needed for original optimization/adoption claims; accepted soak is not a substitute |
| T059, T060, T062 | Original common-v2 queued 1/2/4 native comparison and full policy/counterexample matrix | Original native calibration remains incomplete; accepted slot waves and health pulses have narrower meanings |
| T073 | Original complete-work joint comparisons | Actual joint safety is accepted, but original five-pair adoption coverage remains missing |
| T086 | Complete prospective costed next-run command, finite limits and archive path; current reconciliation is partial | Code/planning preparation; no duplicate primary soak |
| T084 | Independent Docker qualification | A Podman alias cannot supply it; no primary-soak rerun for this capability |
| T093–T107 | Separately admitted research preparation/execution | Held; no final exposure or multi-day run follows automatically |

## Next code and planning package

Prepare a prospective finite longer-run recipe around the accepted configuration, with one consolidated future invocation/archive after its code and host checks are complete. Do not launch it as part of this review. Avoid more size/batch ladders unless a concrete new resource constraint requires one.

1. Freeze purpose, maximum duration and complete cadence before execution. Distinguish a time-limited engineering extension from a scientific learning attempt. Existing v2 research controls use batch 8 and a 3,000-update bound; the experimental batch-16 engineering choice does not silently change that recipe.
2. Define cumulative time/update/job/attempt/frontier bounds and bounded checkpoint/event retention. Existing engineering ceilings are 7,800 seconds, 128 cycles, 49,152 updates, 32,768 jobs, 65,536 frontier entries and 36,864 attempt events. They cannot simply be reused for a multi-day run. The two-hour run produced 57 cycles, 14,646 updates, 5,370 generated native jobs and 16,416 attempt events; naive 48-hour scaling would exceed these ceilings. Scaling illustrates a capacity problem, not a complete cost forecast.
3. Budget storage explicitly. The accepted run reached 7.546 GiB of the 8 GiB transient cap. Specify checkpoint pruning/publication, rolling evidence, export reserve, interruption recovery and archive limits prospectively. Do not expand limits or renew the existing lease during execution.
4. Preserve complete own-state recovery and original clocks/debits, exact source/image/config/data lineage, read-only inputs, rollback and cleanup. Engineering checkpoints remain engineering-only unless a separately reviewed contract grants another use.
5. Make inclusive wall cost authoritative and add any needed component accounting prospectively. Final-save component counters are incomplete; 535.994 seconds of this parent's cost remains unattributed. Whole diverse32 and full 128+128 controls/reference/independent-scoring costs are UNKNOWN. No 80%-of-budget scientific admission is satisfied by training tokens/s.
6. Carry forward the 7.36% between-leg slowdown investigation trigger, native queue/resource incomplete outcomes, memory/cache growth and storage headroom. Evaluate them in the next independently admitted run; do not request an identical two-hour rerun solely to remove a watchpoint.

Preparation is the next ready action. A future command, finite expected complete-schedule cost and archive path belong to that prospective recipe once its unblocked implementation is complete. No new target command is prescribed today, and no new HIP handoff is needed to accept the existing results. Learning quality, organic/stochastic versus curriculum decisions, final access and a multi-day launch remain separate later decisions.
