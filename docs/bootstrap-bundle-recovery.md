# Recovering a bootstrap diagnostic bundle

A completed learnability command returns exit 2 when its gate did not pass.
Inspect `learnability.json` to distinguish accuracy, conditioning, cache-parity
and evaluation-coverage failures. This does not by itself indicate a crashed
training worker.

Detailed token traces can fill the analysis bundle cap. Packaging must retain
run metadata, the gate report, selected training examples and the selected
checkpoint's diagnostic before optional historical role reports. Omissions must
remain explicit and source artifacts must remain unchanged.

After updating the checkout, repackage an existing run without retraining or
rebuilding the GPU image:

```bash
python3 scripts/foundation_run.py pack \
  --run runs/bootstrap-step1-hybrid-control-001 \
  --output tmp-analysis/bootstrap-step1-hybrid-control-001-repacked.zip
```

This host-only operation requires Python's standard library and reads the
existing published run. Do not rerun training to recover an analysis package.
