> Historical handoff: the pilot execution is complete. Detailed audit data remain
> private. No repeat run or calibration is requested by the status update.

# Full-clock learning-pilot implementation and handoff

Implemented009:J001–J005 in [the v3 contract](../contracts/long-run-pilot.md).
The accepted calibration is retained; no repeat calibration/reference/corpus/size/
batch/soak is requested. New current CPU/HIP/installed-package/operation checks
run together in `pilot-admit`. Code/local verification is separate from target PASS.
No long study has been launched by this implementation.

## Main program CLI

From the existing checkout, `oeis-learn pipeline ...` exposes the same persistent
portable interface as `python3 scripts/foundation_run.py ...`. An installed
`oeis-learn` can use `pipeline --workspace /path/to/oeis-learn ...` from elsewhere.
The commands below use the main Python module with `PYTHONPATH=src`, requiring
only host Python's standard library and the existing Podman connection. No host
package install, separate script, image build, qualification command or hash editing
is needed. A wheel alone is not a workspace: the engine builds/inputs remain in
the selected checkout.

First review the frozen scope:104.5M FP32/math, fresh seed20261016, unchanged32768
training corpus,296prompts/six arms, six0/1.5/3/6/12/24learner-role-hour snapshots,
24learner-role hours within48inclusive hours, evaluation36000s/safety41400s.
There is no short-run learning-success gate. The command below's `--approve`
explicitly accepts that exact design and the compatible source/data/image/cost
bindings generated after all gates pass. Omit both `--approve` and `--reviewed-by`
for a DRAFT-only preparation; that draft cannot run.

The full original producer and full successful calibration directories must still
exist at these paths. Their public ZIPs alone do not contain private inputs/state.
They are read-only; no trained diagnostic state is used to initialize learning.

```bash
git pull --ff-only origin 009-pipeline-throughput &&
pilot_admission_output="tmp-validation/009-longrun-admission-$(date -u +%Y%m%dT%H%M%SZ)" &&
pilot_run_output="tmp-validation/009-learning-pilot-longrun-$(date -u +%Y%m%dT%H%M%SZ)" &&
PYTHONPATH=src python3 -m oeis_learn.cli.main pipeline --engine podman --cpus 16 pilot-admit \
  --inputs tmp-validation/009-learning-pilot-preflight-20261026T213151Z \
  --calibration tmp-validation/009-pilot-calibration-20261027T141405Z \
  --approve --reviewed-by 'Owen Barton' \
  --max-seconds 43200 --output "$pilot_admission_output" &&
PYTHONPATH=src python3 -m oeis_learn.cli.main pipeline --engine podman --cpus 16 validate \
  --suite pipeline --phase 6 \
  --study-manifest "$pilot_admission_output/stages/study.report/decision.json" \
  --max-seconds 172800 --output "$pilot_run_output"
```

This is one operator command block with conditional sequential admission and run;
there is no result handoff between those steps. Neither command starts the other
by default. Failed qualification/import/export returns nonzero and prevents the
second command. Source, images, resource envelope, CPU, data, filesystem and
review authority are rechecked before actual training. The independent Docker
portability gap remains explicit; the passed primary-engine scope is used.

Optional read-only scope display (no inputs resolved, engines or training):

```bash
PYTHONPATH=src python3 -m oeis_learn.cli.main pipeline --engine podman --cpus 16 pilot-admit \
  --inputs tmp-validation/009-learning-pilot-preflight-20261026T213151Z \
  --calibration tmp-validation/009-pilot-calibration-20261027T141405Z \
  --output tmp-validation/009-longrun-scope --scope-report
```

## Costs and outputs

The source-bound calibration projects **126780s =35h13m warm study**, including
86400s learner-role reservation,31380s evaluation and9000s qualification/setup/
cleanup/archive. The threshold is138240s =38h24m (80% of48h), margin11460s =3h11m.
Actual learner updates stop150s before each nonzero role reservation for bounded
validation/checkpoint/supervision; reports retain actual time/tokens/updates.
These are finite envelopes from early models, not guaranteed later coverage/cost.

New admission's build/tests/import/export are separately charged before that
48h ledger begins. The prior calibration's corresponding stages observed about
**23m27s including builds** (1411.13s; most recent base for planning, not a current
measurement). New code/image cache/added fixtures can change it; hard cap12h,
shared build cap4h, no hidden unlimited retry. Previous preparation, reference
production and calibration remain paid history, with original offline reference
total UNKNOWN. They are not folded into or relabeled the warm35h13m estimate.

Admission outputs `$pilot_admission_output/summary.json`, `.zip` alongside the
folder, `stages/pilot.compatibility/compatibility.json`, and
`stages/study.report/{cost,qualification,environment,decision,authority,admission}.json`.
`authority.json` exists only for explicit passing approval. The long run outputs
`$pilot_run_output/summary.json` and a sibling `.zip`, per-milestone
`stages/pilot.evaluate-N/snapshot-N.json`, and
`stages/study.report/{learning,debits,report}.json`. The archive's exact integrity
manifest records all public evidence, including missing/interrupted work.
The durable ledger is `run-state/pipeline-pilots/<decision identity hex>/ledger.json`.

Retain the complete admission, producer and every run output directory. Public
ZIPs exclude private checkpoints. Full-state latest/selected remain separate;
future hidden outcomes never choose training state. Bounds remain16GiB transient,
32GiB cumulative durable,5GiB free floor,512MiB archive/4097members,16invocations,
finite work/metric/evidence caps and the original boot/monotonic48h lifetime.

## Interruption and recovery

After SIGINT, wait for the CLI's supervised cleanup/archive to finish. Review any
fatal error. Do not repeat `pilot-admit` to obtain a new decision/lifetime after
research starts. Resume only the **same original decision file** with a new output
path and unchanged source/images/resources on the same boot:

```bash
PYTHONPATH=src python3 -m oeis_learn.cli.main pipeline --engine podman --cpus 16 validate \
  --suite pipeline --phase 6 \
  --study-manifest "$pilot_admission_output/stages/study.report/decision.json" \
  --max-seconds 172800 \
  --output "tmp-validation/009-learning-pilot-recovery-$(date -u +%Y%m%dT%H%M%SZ)"
```

The ledger reuses complete slots with original producer identities; interrupted
slots stay charged and are never retried. Only acknowledged nonfatal cleanup and
a full-state CPU audit permit later unstarted roles. Initialization cannot be
silently replaced; missing milestones stay missing. Fatal/late/orphan/boot changes
block continuation. Reports preserve fixed denominators, verified matches,
coverage/failure bounds, train/dev gaps and UNKNOWN further-compute estimates
when comparability/coverage/backtest requirements are unmet. Actual L014 execution
and L015 scientific review remain outstanding.
