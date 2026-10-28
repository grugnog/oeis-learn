# Completed pilot review: public status

Supplemental009:L014 (actual admitted pilot execution/publication) is complete.
L015 (the subsequent learning-regime decision) remains open. Original research
comparison, independent-engine, adoption and final-exposure gates retain their
previous dispositions. Independent review of the separate admission evidence is now complete: T086
and T110 close. No new HIP run is requested.

The reusable [stdlib review utility](pilot_longrun_review.py) accepts the operator’s
public run and producer archives, verifies integrity/lineage/clock/science/resource
records and reproduces the learning report. It runs locally and never publishes
its input or result.

```bash
python3 -S specs/009-pipeline-throughput/validation/pilot_longrun_review.py \
  /path/to/run.zip --producer /path/to/producer.zip --output /path/to/review.json
```

Detailed run-derived paths, hashes, metrics and resource observations are retained
privately and are not included in this public update. No new experiment, strategy
switch or default promotion is authorized by the review.

The [admission audit utility](pilot_admission_review.py) additionally revalidates
public authority and prerequisite artifacts against the exact completed-run
decision and ledger, checking original data/calibration and actual image bindings.
It reads operator-provided archives locally and writes its detailed result locally.

```bash
python3 -S specs/009-pipeline-throughput/validation/pilot_admission_review.py \
  --admission /path/to/admission.zip --run /path/to/run.zip \
  --producer /path/to/producer.zip --calibration /path/to/calibration.zip \
  --output /path/to/admission-review.json
```

Only this reusable code and task disposition are published; detailed run-derived
metadata remain private. Independent-engine coverage and the next research
decision remain separate.
