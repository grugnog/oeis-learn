# Execution and operator contract

All names introduced here are proposed interfaces, to be implemented and checked
against actual `--help` before publishing an execution command. Existing host entry
is `python3 scripts/foundation_run.py --engine docker|podman COMMAND`; host helpers
remain stdlib-only. Reuse canonical generalization preparation, training,
evaluation, scoring, qualification and archive verification handlers. Extend their
closed registered contracts for PAC; do not implement these operations in a wrapper.

## Temporary wrapper boundary

Proposed location: `specs/012-parameterized-adaptive-curriculum/validation/run.py`.
Subcommands `prepare`, `run`, `resume`, `recover` are planned, not currently present.

- `prepare`: explicit engine, original input roots/inventory, output, control
  disposition, optional ablation selection and campaign mode. Perform read-only inventory/control audit and
  construct a finite dependency/budget graph. Display scientific configuration,
  calibration inputs and operational ceilings before costly work. No unbounded
  directory walk; bounded search exhaustion requests an explicit root/inventory.
- `run`: consume a sealed plan. Default scope is software acceptance. Research
  requires a manifest already marked/frozen for the operator's chosen research
  scope; no prompt or automatic scope change halfway through a long invocation.
- `resume`: verify existing plan/receipts and execute the smallest unfinished or
  incompatible dependency; never overwrite immutable prior receipts. Model resume
  requires complete compatible state, not imported weights alone.
- `recover`: read-only evidence collection, verification and export. Never launches
  training or rewrites original failure status.

A normal session executes serially: inventory and capacity gate; CPU/native full
structural/probe admission; required compatible qualification; representative
calibration and small installed HIP acceptance; retained structural baselines;
optional frozen new arms and final evaluation; case analysis; verified export.
Freeze all scientific choices before baseline inference. Failure suppresses
its dependent stages but still triggers bounded diagnostic packaging. Independent
completed stages are not automatically repeated. GPU roles are serial; only
bounded deterministic CPU preparation may use existing worker parallelism.

## Relaxed finite timeout policy

Use the maximum of an existing justified allowance and the calculated new allowance.
Do not shrink a limit on the evidence of one fast/warm run. Store requested AND
effective values at wrapper, canonical CLI, schema, supervisor and role boundaries.
A hard-coded legacy ceiling must cause preflight rejection, not silent clipping.

For production preparation/learning, use a floor of **24 hours for preparation**
and **12 hours per new learner**, then raise either when `3 * forecast_work + 2h`
plus measured startup/publication costs is larger. These are conservative planning
floors, not claims that work takes this long. Preserve at least the prior 6.5-hour
per-arm generation allowance and enlarge it if the full nested graph requires
more. Inner work and outer process limits are separate: the containing process
adds startup/exit and cleanup reserves rather than subtracting them unexpectedly
from the promised inner work. Preserve longer existing relevant limits.

For startup, image inspection, model load, checkpoint serialization/hash/copy,
proof and reporting, reserve at least three times the **largest representative
observed** cost, plus explicit fixed overhead. Use the previous justified limit
when measurement is absent; absence is a calibration gap, not a zero. Cold startup
and real production-size model save/load are measured separately from tiny-model
correctness. Synchronize asynchronous device work where timing requires it.

The calibration plan samples short/long target families and cold/warm native
execution, early and later preparation shards, actual learner updates and a real
production-sized save/load. Report sample counts, worst observed throughput, scope
and extrapolation uncertainty. Do not claim a worst-case execution-time guarantee
from a sample. If the projection plus margin cannot fit, stop before research and
export the measured plan; do not begin hoping the deadline is sufficient.

Build enclosing case/command/campaign ceilings by summing all sequential child
reservations, copy/publication overhead, protected cleanup and finalization. Use
the critical path for explicitly bounded parallel CPU work, with shared resource
contention included. Reserve **at least one hour for final reporting/recovery**
at campaign level, and preserve each existing canonical cleanup floor. The outer
campaign has no fixed inherited 5-, 6-, 30- or 96-hour ceiling: compute and freeze
a sufficiently large **finite** total from the selected arms and full graph.
An extra fresh A or optional bandit-only D must add its complete preparation,
training, diagnostic, checkpoint and evaluation reserve. No optional arm is added
without updating both the frozen manifest and enclosing operational graph.

Small acceptance also uses relaxed canonical startup/proof/cleanup ceilings; it
is small in actual updates/data, not forced into a tiny wall timeout. Existing old
recipe limits keep their historical meaning; add a new versioned PAC authority
where needed. Test every limit at exact-boundary and over-boundary values,
including schema maximums, resumed proof, failure paths and final publication.
No automatic extension during research and no silent budget reset on resume.
Spent time and discarded exposure remain charged. A revised operational envelope
is separately recorded without altering frozen scientific per-task budgets.

The 60-second structural portfolios and five-second training semantic probes are
**scientific measurement budgets**. Container startup, model loading and report
writing do not eat those budgets; each runs inside its larger containing envelope.
Censoring at a genuine scientific budget is a reported outcome, not evidence that
a generic timeout should automatically change the experiment.

## Acceptance and reuse

Tiny acceptance must reach initial/final snapshots, one valid parameterized and
one fallback row, a known pre/block/post reward and valid kernel update, anchor
acquisition followed by a graded due review with an optimizer update, clean
pause/resume inside an unfinished block and after gradients/before feedback, and
a fault with recovery export. Derive encoded batch/token minima
before launch; never assume a nominal 4,096-token cap can reach all boundaries.
Use an explicitly reduced acceptance policy with shorter measurement/review cadence
and block length, retaining the exact kernel and grade/commit rules; record it as
such. The fixture must exercise a nondegenerate reward-scaler update, not only its
warmup zeros. Compare uninterrupted/resumed scaler and queue state as well as model
tensors. Production 8-update blocks and the 32-update cycle receive independent
CPU clock/update fixtures. Acceptance artifacts cannot be research initializers.

Require actual changed parameters, finite loss/gradients, HIP device/RNG evidence,
checkpoint identity and installed-handler full-state comparison. Test no-Lt process
startup without intrusive tracing. Measure the complete recurring reward/grade/
common-panel overhead on representative
production shapes; tiny correctness timing is not its forecast. Include GPU
synchronization and eval/train RNG-mode restoration in reserves. A source/image/
backend change requires affected
qualification; old successful acceptance is not current evidence by filename.
Cache by full dependency identity, never path or a bare PASS marker. A source-only
structural helper change can retain compatible producer qualification while binding
its own plan/source closure separately. A changed training policy cannot.

## Disk, supervision and evidence

Compute peak live bytes from original data, permitted references/copies, old/new
full snapshots, milestone pins, proof projections, runtime scratch, logs and both
success/recovery packages. Reject insufficient capacity before work. Reuse verified
immutable data without duplicate private trees where existing isolation permits;
never prune original inputs automatically. Use the existing disk-backed private
scratch and containment/watchdog implementation.

Every stage emits bounded progress at most 60 seconds apart, or explicitly marks
progress unavailable/stalled. Heartbeat elapsed time is not completed work. Export
whole-run counters and measured ETA ranges, save overhead, required/available
space, requested/effective deadlines, source/image/policy/helper hashes, original
producer references, application/process/cleanup outcomes and ordered errors.

Canonical finalization must bind original report bytes, verifier result, archive
hash and terminal completion receipt. Verify success, negative scope, interruption,
missing nested evidence and exporter failure paths using the real packer. Preserve
private weights/state/truth and host paths outside the public archive. A minimal
fallback records first and subsequent errors and how to invoke read-only recovery.
No archive or quota score is synthesized when evidence is unavailable.
