# Matched decoder input comparison on ROCm

This is step 2 after the [hybrid checkpoint investigation](hybrid-checkpoint-probe.md).
It tests the candidate from scratch; the successful CPU continuation does not
establish from-scratch learnability or transfer to OEIS.

## Supported controls

`learnability` now accepts `--decoder-token-scale`, `--decoder-position-scale`,
`--dropout`, and `--seed`. Supplying any of these opts into `bootstrap-model/v2`
and requires an explicit `--encoder`. Without them, the frozen v1 profile,
defaults, and five-key learnability settings remain unchanged. These controls
are currently exposed for bounded learnability experiments, not production
`train` runs.

The token multiplier is applied on top of the existing `sqrt(d_model)` factor.
Position gain multiplies unit sinusoidal values at use, including after buffer
growth or checkpoint reload. It does not change the stored sinusoidal buffer.
Dropout applies to both encoder and decoder, matching the checkpoint probe.
The seed controls model initialization and sampling; it never re-prepares or
changes the sealed examples. Omitting `--seed` retains the input configuration's
seed. Numeric scales must be finite: token gain is in (0, 256], positional gain
in [0, 256], dropout in [0, 1), and seed an integer in [0, 2**32).

The profile hash includes all model controls. Versioned decoder state also
binds the gains and dropout: loading into a different configuration fails rather
than silently reinterpreting weights. Existing v1 state dictionaries retain
their original keys. Each changed experiment needs a new run directory; resume
retains the issued settings, code/image, checkpoints, and budget.

## Run the hardware checks and matched pairs

After updating to this implementation, run from the checkout on the ROCm host.
The commands below reuse `tmp-bootstrap-inputs`; do not prepare different inputs
for the two arms. `docker` may replace `podman` consistently.

```sh
python3 scripts/foundation_run.py --engine podman image ensure
python3 scripts/foundation_run.py --engine podman validate \
  --suite gpu --output tmp-validation/decoder-inputs-v2
```

The GPU suite includes real updates, reconstruction from checkpoints, and
cached/full repeated-literal decoding for the candidate as well as v1 controls.

If validation reports a failure in
`test_complete_effective_constructors_and_real_forward_backward`, inspect the
failure section of `stdout.log` rather than only the final summary:

```sh
sed -n '/FAILURES/,$p' \
  tmp-validation/decoder-inputs-v2/gpu-regressions/stdout.log
```

An assertion listing missing `token_scale`, `position_scale`, and `input_version`
was a constructor-declaration regression: it ran before any GPU forward/backward.
The frozen foundation constructor now explicitly declares the original values
`1.0`, `1.0`, and `None`. This preserves v1 behavior and the full-signature test.
Pytest's cache also lives in the writable private runtime cache; it is excluded
from validation evidence receipts.

After pulling the fix, rerun the two commands above using the same output path.
The runner checks code/image identities and reruns affected stages. No validation
directory, training run, or prepared input needs to be deleted. GPU environment
checks always execute again.

Run the following comparison after those hardware checks pass:

```sh
for seed in 20261016 20261017 20261018; do
  for arm in control candidate; do
    case "$arm" in
      control) gain=1; dropout=0.1 ;;
      candidate) gain=16; dropout=0 ;;
    esac
    rc=0
    python3 scripts/foundation_run.py --engine podman learnability \
      --inputs tmp-bootstrap-inputs --encoder hybrid --view shortest \
      --decoder-token-scale 1 --decoder-position-scale "$gain" \
      --dropout "$dropout" --seed "$seed" \
      --max-updates 3000 --validation-every-updates 100 \
      --budget-seconds 7200 --diagnostic-seconds 180 \
      --run "runs/decoder-v2-${arm}-${seed}-001" || rc=$?
    case "$rc" in
      0) ;;
      2) echo "Recorded learnability gate failure: $arm seed=$seed" ;;
      *) echo "Execution failed: $arm seed=$seed exit=$rc" >&2; exit "$rc" ;;
    esac
  done
done
```

There are six sequential runs, each capped at two hours including diagnostics
and orchestration. They may finish sooner because of update limits, patience,
or remaining evaluation budget. Exit 2 means the completed learnability gate
failed, and is retained as comparison evidence; other errors stop the loop.
Use a new suffix if these names have already been issued. Re-running a complete
unchanged run republishes it rather than starting another experiment.

The same seed gives the two arms identical initial parameter tensors and an
independent, matched sampler order. Dropout changes training randomness as part
of the intended intervention. Both explicit arms use v2 profile/checkpoint
identities; the control's numerical defaults match v1.

## Inspect stability and send reports

Each run automatically writes a reports-only archive under `tmp-analysis/`.
Inspect the entire committed checkpoint history, not just the selected gate:

```sh
python3 - <<'PY'
import json
from pathlib import Path

lines = ['run,seed,gain,dropout,updates,exact,total,evaluated,coverage_complete']
identities, examples = set(), set()
for run in sorted(Path('runs').glob('decoder-v2-*-001')):
    record = json.loads((run / 'run.json').read_text())
    cfg = record['settings']['learner']
    identities.add(record['inputs_identity'])
    examples.add(cfg['examples_sha256'])
    model = cfg['model']
    for stage in sorted((run / 'roles').glob('train-*')):
        path = stage / 'result.json'
        if '-attempt-' in stage.name or not path.is_file():
            continue
        result = json.loads(path.read_text())
        measured = result.get('diagnostic', {}).get('splits', {}).get('training')
        if measured is None:
            continue
        lines.append(','.join(map(str, [
            run.name, cfg['seed'], model['decoder_inputs']['position_scale'],
            model['dropout'], result['completed_updates'], measured['exact'],
            measured['denominator'], measured['evaluated'],
            measured['coverage_complete'],
        ])))
assert len(identities) == len(examples) == 1, 'Comparison inputs differ or are missing'
text = '\n'.join(lines) + '\n'
Path('tmp-analysis/decoder-input-history.csv').write_text(text)
print(text, end='')
PY
```

The existing gate requires at least 31/32, full measurement/cache coverage, and
a conditioning effect. For this comparison, seek repeated 32/32 performance
across checkpoints and seeds, correct termination of both `+11` and `-11`, and
matching cached/uncached generation. A single 31/32 gate crossing does not settle
the repeated-digit issue. Development synthesis and loss remain separate from
memorization; no final OEIS cohort is consumed.

Package the six reports-only ZIPs for analysis:

```sh
python3 - <<'PY'
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

files = sorted(Path('tmp-analysis').glob('decoder-v2-*-001.zip'))
assert len(files) == 6, f'Expected six comparison archives, found {len(files)}'
output = Path('tmp-analysis/decoder-input-comparison-001.zip')
history = Path('tmp-analysis/decoder-input-history.csv')
assert history.is_file(), 'Run the history extraction above first'
with ZipFile(output, 'w', compression=ZIP_STORED) as bundle:
    bundle.write(history, history.name)
    for file in files:
        bundle.write(file, file.name)
print(output)
PY
```

Upload `tmp-analysis/decoder-input-comparison-001.zip`. These reports contain
controls, runtime/GPU evidence, examples, per-checkpoint diagnostics and token
traces. Weight payloads are unnecessary for the initial comparison. If further
weight inspection is needed, use the existing `pack --checkpoint` command on
the specifically selected run.
