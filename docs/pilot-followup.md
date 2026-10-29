# Running the pilot follow-up

Use the full retained run, admission and producer directories on the HIP workstation, not the public ZIP exports. The selected checkpoint must still exist under the final trainer's checkpoints directory. Failure to find or verify it stops the invocation.

```bash
python3 scripts/pilot_hip_serial.py --engine podman \
  --run tmp-validation/009-learning-pilot-longrun-20261027T180510Z \
  --admission tmp-validation/009-longrun-admission-20261027T180510Z \
  --inputs tmp-validation/009-learning-pilot-preflight-20261026T213151Z \
  --output tmp-validation/010-followup --dry-run
```

Remove `--dry-run` to execute. The helper runs `foundation_run.py` commands sequentially: bigint CPU/native qualification, HIP qualification, snapshots 2/5/selected diagnostics, matched evaluation, matched evaluation with native draining, then the three-arm fresh screen. `--skip-training` runs only qualification and saved-weight work. `--seeds 20261016 20261017 20261018` adds independent confirmation screens; every screen has its own bounded invocation. The maximum study invocation is 36 hours including three six-hour learner ceilings, evaluation, qualification and reserves; completion can be earlier at the 25M-token cap. Expect substantial target time. Logs and exact commands stay under the new output directory. The helper is temporary and should be removed after target review.

Each command can also be run separately with the portable host CLI. `pilot-diagnose ... --scope-report` and `pilot-followup ... --scope-report` describe sealed scope without launching an engine. Study outputs contain best/latest weights and optimizer/RNG state privately and public counters/hashes. They do not authorize automatic continuation or final exposure. Match inclusive compute as well as training tokens when comparing arms.

The old six-snapshot diagnostic contract and original portfolio policy retain their names and defaults. New saved-weight evaluation requires an explicit operation and new policy binding. New target runs must requalify current source; prior engineering receipts remain historical.

Local validation: standard-library follow-up tests and CLI scope/dry-run checks are runnable without an engine. Torch/native/HIP qualification must be performed on the target through the helper; do not interpret host checks as GPU evidence.

After an earlier serial attempt stops, `--start-step N` can omit completed top-level commands while writing to a **new** output root. Original command numbers and log names are retained. For the six-command default roster, `--start-step 4` runs matched evaluation, matched evaluation with native draining, then the training screen. This skips the standalone qualification commands and model probes; each evaluation/study invocation still qualifies its current installed source. It neither reuses a qualification receipt nor extends a failed invocation's clock. `execution-plan.json` explicitly records the omitted steps. Use `--dry-run` to inspect the selected command suffix before executing.

Matched evaluations retain strict native service deadlines. If an active operation crosses that deadline, the scheduler pauses new work and permits up to five seconds for cleanup; late candidates remain excluded and the slot remains charged. Work that fails to quiesce within that interval aborts the role. This cleanup interval does not extend portfolio or service eligibility. Generation and scoring use separate immutable settings descriptors, while the learner's validation-history file is a live progress journal.
