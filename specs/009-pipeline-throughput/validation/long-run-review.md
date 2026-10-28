# Completed pilot review: public status

Supplemental009:L014 (actual admitted pilot execution/publication) is complete.
L015 (the subsequent learning-regime decision) remains open. Original research
comparison, independent-engine, adoption and final-exposure gates retain their
previous dispositions. Independent review of the separate admission evidence
remains pending; no new HIP run is requested.

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
