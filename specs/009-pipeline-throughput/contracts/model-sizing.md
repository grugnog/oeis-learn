# Bounded model sizing v1

The user requested a larger model before the long-run decision and authorized
continuing the proposed small hardware screen. Register a separate
`pipeline-model-sizing-validation/v1`, scope `engineering-model-sizing`, registry
`pipeline-model-sizing-registry/v1`, phases1/2/4, workload `sizing-real32/v1`,
comparison `model8-32-100/v1`. Historical manifests/results stay immutable.

Three exact profiles: width256, four encoder/four decoder layers, heads4, FF1024;
width512, four/four layers, heads8, FF2048; width768, six/six layers, heads12,
FF3072. Keep hybrid-chunks input features, all four input groups, FP32 eager,
sinusoidal positions,1024-token body cap, dropout0, seed20261016, AdamW/LR and
training batch8. Record actual encoder/decoder/active/total parameters. New models
start from seeded fresh initialization, including the8M control; parent trained
weights are lineage only, never resized or inherited. No model-quality comparison.

The existing checked completed parent/prepared inputs supply the same128 training,
32 teacher-forced validation and32 distinct visible prompt records. No hidden truth
is read or mounted. Freeze all row/token/data/recipe/profile/source identities and
CPU initialization weights/oracles before GPU timing. CPU full-prefix oracle uses
two actual visible inputs and positions1/32/128; HIP cached logits and greedy IDs
must match at the retained tolerance before disposable updates. Full1024 grammar
generation is separately measured, not asserted numerically by this short oracle.

Exactly five fresh roles per profile: cold, excluded warmup, calibration0/1/2.
Each performs32 actual AdamW updates at batch8, the same32 validation rows, and
one real grammar sample for each of32 unique prompts at each batch8/32. Fixed
batch order:8 then32 for small/large,32 then8 for medium. Seeds are mapped to
prompt IDs independently of batching. Each actual generation group retains its
60s production clock and1024-token cap; no retry, padding, replicated work, larger
generation limit or extra attempts. Every planned row retains decoded/partial/
unstarted state and all emitted tokens/source hashes. Timeouts are reported as
incomplete generation, never useful program completions. Sampling cannot mutate
model parameters. Do not use fresh-model generation to infer learned quality.

After each generation arm, independently fit/verify only complete emitted programs
with four bounded native workers, at most four admitted requests per wave and an
immutable4s admission/dispatch/owner clock. Preserve exact native evidence and
conditional decoded/verified denominators; zero generated programs means zero,
not missing data. No template substitution or native fixture work fills omissions.
GPU/native work is serialized here to attribute size costs; accepted overlap remains
the direction for later integration. Each profile keeps one fixed workload, not a
search for favorable samples or batch adoption. Different models are not equivalent
optimization arms: report absolute throughput/cost, not five-pair speedup/adoption.

All15 GPU slots and64 generation-row slots per trial are planned before work. Use
one1110s external/1080s internal parent per role; CPU references600s each. Required
work reserves600s training/validation/parity,300s generation,64s native and116s
setup/export/drain inside1080s; no component renews its trial parent. Resource
failures/nonquiescence/interruption stop globally, and partial work cannot qualify
a profile. Record all five roles even when descendants are blocked. Reference
roles are CPU-only; one GPU role owns one shared at-most16CPU/90%-physical-or-
ancestor/engine memory envelope, Torch2/1 and native numerical-thread policy1.
Conservative prospective profile reservation plus headroom must fit known current
availability with10% unreserved capacity. It is an admission estimate, not measured
memory proof. Actual aggregate watchdogs remain mandatory.

Use one-Hz whole-role observations, including cgroup memory.stat, device residency,
CPU quota, disk, component durations and checkpoint-free disposable update costs.
Telemetry overhead is uncalibrated; all models use the same collection mode.
Weights/oracles are hard-linked into read-only role inputs rather than copied for
every trial. Full generated/native evidence stays bounded by unchanged512MiB ZIP,
8GiB transient, stream/trace/reserve/global/build and5s cleanup caps. Fifteen1110s
GPU reservations plus30s report =16680s, below phase4's18000s. Three600s references,
30s preparation and normal host/CPU readiness fit phase1's3600s (builds separately
share14400s per engine). HIP600s stays phase2. No new dependency or host toolchain.

Report complete/partial profiles, all absolute training tokens/s, updates/s,
validation rows/s, generated tokens/s, decoded and independently verified programs
per whole-role minute, batch/prompt denominators, native clocks, full costs and
observed memory. Calibration aggregates require all three complete calibrations
and all five role audits; no favorable-subset medians. Parameter-count ranking of
hardware-eligible profiles is descriptive, not a selected learning configuration.
Generation incompleteness stays explicit and cannot establish useful-throughput
readiness even if finite updates/memory qualify. Original complete-work/five-pair,
persistent-state/operations/cost/learning gates remain separate.

One command shares CPU/HIP qualification, all references/roles and one archive.
The next persistent-state soak depends on these measurements and an explicit model
choice. No multi-day run, research cost admission, resume or final authority is
created by this hardware screen.
