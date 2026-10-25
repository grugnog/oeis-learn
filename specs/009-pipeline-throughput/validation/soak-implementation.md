# Full-scale engineering soak: corrected target handoff

The [first target review](soak-operator-20261025T211131Z.md) verifies all twelve full-data batch trials and selects32, but UID operations block every soak leg. The permission/image repair passes192 host tests and is pushed; corrected actual target qualification remains pending.

Code and available local checks are complete on `009-pipeline-throughput`,
PR [#60](https://github.com/grugnog/oeis-learn/pull/60), stacked on #59.
The [integrated plan](../plan.md) and [closed contract](../contracts/soak.md)
govern this collection. Actual current-source CPU/HIP/engine qualification and
sustained hardware readiness are **pending**, not inferred from host simulations.

Run once from the existing checkout on the HIP/ROCm workstation:

```bash
git pull --ff-only origin 009-pipeline-throughput &&
python3 scripts/foundation_run.py --engine podman --cpus 16 validate \
  --suite pipeline --phase all-ready --engines both \
  --manifest configs/foundation/pipeline_soak.json \
  --run tmp-validation/008-phase5-diverse32-20261021T160815Z/learnability-run \
  --inputs tmp-validation/008-phase5-hip-20261020T202907Z/inputs \
  --max-seconds 86400 \
  --output "tmp-validation/009-soak-$(date -u +%Y%m%dT%H%M%SZ)"
```

Return the resulting `tmp-validation/009-soak-*.zip`, including on failure.
No separate build/test/profiler/manual batch choice/resume/package command is
needed. The two existing input paths remain read-only; adjust only their locations
if they moved. The completed parent establishes lineage and compatible prepared
data; this run initializes a fresh104,496,662-parameter model and imports no trained
8M weights. It requires the full approved512-or-more training and128-or-more
records in each development partition. Missing/incompatible inputs block dependent
work explicitly; there is no silent diagnostic-subset fallback.

The38-stage graph runs current host, pinned CPU/native/HIP, installed-command,
retirement and both-engine operations prerequisites; creates one full-model CPU
reference; checks HIP parity; runs batch8/16/32 warmup plus three equal two-pass
calibrations each; freezes the smallest batch within5% of the best complete median;
checks long-context actual updates and complete-state recovery; measures optional
instrumentation overhead; runs two3600s persistent work windows with one controlled
container/process restart; then launches the isolated public CPU scorer and reports.
Incomplete batch arms stay visible; resource breaches or failed quiescence stop
further compute. A missing secondary engine leaves portability pending while an
independently qualified primary can continue.

The existing trainer owns actual AdamW updates, full development validation,
fixed periodic96-prompt generation, native fitting and checkpoints. Cadence is
eight complete training passes in actual target tokens. Read-only GPU probes and
at most twelve18-case native health pulses exercise coexistence even if the fresh
model produces no usable programs. Probe jobs remain separate from generated work.
Skipped pulse intervals stay UNSTARTED. Each soak leg must include a full training
traversal, a fully dispatched diagnostic/validation/checkpoint cycle, a native pulse
and real updates. Restart verifies all seven checkpoint state payloads, exact weight
continuity and the original cumulative budget; no leg receives a renewed parent.

Planning estimate is3–6 hours with warm caches, uncalibrated. Two3600s work windows
share600s for setup/checkpoint/restart/drain within a7800s parent. The invocation's
hard ceiling is86400s; existing maximum reservations total84000s, including up to
14400s shared builds per engine and archive/emergency reserves. Actual non-image
stage reservations for phases1–5 are1440/5820/3600/12600/11340s. Streams8MiB,
transient disk8GiB, traces128MiB, public ZIP512MiB and report reserve16MiB remain.
Roles retain5s cleanup, native4s admission-through-owner deadlines, at most16 CPU
equivalents, Torch2/1 and measured aggregate memory/headroom admission. Batch32
long-context stress reserves a conservative69,915,886,272-byte live upper bound
before headroom; the selected batch's own bound governs the soak.

The report separates correctness/recovery, ten steady five-minute windows per leg,
resource behavior, useful generated work per whole soak minute, portability,
default adoption and scientific costs. Missing matched windows/sensors or missing
work stay inconclusive. More than10% matched steady rate decline prevents a
sustained PASS;5% is an investigation trigger. Anonymous/file/driver growth and
checkpoint/launch/cleanup costs stay visible. Hardware qualification has no learned
success threshold, and zero generated-to-native work is explicitly NOT_EXERCISED.

Repair local evidence:192 stdlib tests pass and the actual public CLI exports a
verified88-member PARTIAL archive with zero omissions when both engines are
unavailable. The full UID exporter/recovery code executes locally under restrictive
umask077, with only its read-only mount simulated. Both images now explicitly
provide searchable0755 mount parents, the fixture explicitly publishes readable
public proof files and logs UID/path modes, and primary UID operations gate
preparation/candidate timings. A pinned CPU image-path regression is registered.
Actual repaired-image foreign-UID/rootless access and the unstarted soak still
require the command above; no host permission/account changes are needed.

At prior producer d80bf576,716 CPU tests plus31 subtests,21 HIP tests and all12
batch roles pass; all four scoped full-size parity records have zero mismatches.
Batch32 achieves2543 median training-component tokens/s versus2042 at8, with7.41GiB
maximum calibration role upper memory. Short whole-role GPU means near63% include
setup, validation and checkpoint/drain and do not establish steady training duty.
No joint/stress/overhead/soak/scorer role starts. Docker is a Podman compatibility
alias and remains a portability blocker only. Both already-completed PR workflows
at d80bf576 pass; they do not qualify repair source1f3bba26, and no new CI is awaited.
See [operator review](soak-operator-20261025T211131Z.md) and
[repair local evidence](soak-uid-repair-local-evidence.json).

Current original tasks:77 complete /19 runtime pending /15 research held. S001/S003/S007–S010 close for implementation/local scope. S002 has actual model/count/parity/update evidence at d80bf576. S004–S006 retain incomplete runtime clauses, S011 awaits corrected full-scope evidence and S012 its review. Original adoption remains INCONCLUSIVE/RETAIN_BASELINE with
all20 cold/warm confirmation slots UNSTARTED: this disposable model/batch scope
does not establish the original complete-work baseline/strata for adoption.
Diverse32/full128+128 whole-schedule costs remain unknown.008:T093 stays deferred,
008:T096 stays NEGATIVE7730/14400s, final data stay sealed and learning/multi-day
admission remains separate. A further HIP loop requires a concrete blocker; finish
unblocked repairs first and reuse only proven-compatible evidence.
