# Round027 trial 00 exploratory continuation

Trial 00 was selected after reviewing six 1M results. All six were formally infeasible. This follow-up measures whether more training improves the saved baseline on the same visible development and retention cases. It has its own frozen task identities, hardware acceptance, checkpoint store and budget journal.

| Checkpoint | Evidence and decision |
| --- | --- |
| Six original 1M looks | Recompute the saved frozen and companion measurements from the original producers. The registered result is zero frozen passes and one companion pass, trial 00. Verify before training. |
| Trial 00 at 2M | Continue the original full training state and save a new checkpoint. Generate and score both panels. Publish both readings, verified yield, paired case changes, unresolved reasons and settled costs. Continue automatically if the companion rule passes. |
| Trial 00 at 4M | Continue from the verified 2M checkpoint. Repeat the measurements and publish the final report under either outcome. |

Both readings require at least 95% completion in each of the two 32-case primary development strata and at most two percentage points of conservative retention loss against the original 128-case paired baseline. The companion rule counts a proven no-EOS token cap or an independently reproduced parameter-independent control-range failure as a spent, unsuccessful draw. A case changes only when every remaining gap is accounted for. Verified successes require the original scorer. Timeouts can remain unresolved. There is no yield threshold for the exploratory 4M decision.

The qualified checkout is `77667e67176ae32fa16457d52565bae1dbe41afc`. The shell entrypoint extracts the registered host files from Git. Raw producer evidence and resumable checkpoints stay private. The reports in `run-state/discovery/013-round027-trial00-followup/share/` contain aggregate results and evidence identifiers. Rerunning with `--run` checks saved boundaries and continues eligible work. Use `--verify` to reconstruct published results from their producer files.

The original round027 study keeps its six terminal screening decisions. This follow-up authorizes neither a new screening trial nor an evaluation on the locked assessment set.

## Workstation checks after the 2M report

From the root of the qualified checkout, fetch the host tools and run them directly from Git. These commands use Bash.

```bash
set -o pipefail
git fetch origin 013-progressive-mathematical-discovery &&
rev=$(git rev-parse FETCH_HEAD) &&
validation=specs/013-progressive-mathematical-discovery/validation &&
git show "$rev:$validation/report-round027.py" | python3 -BS - --panel
```

The panel report uses the two saved capability controls and seven checkpoint measurements on identical development cases. It prints aggregate counts. To see allocated space, a conservative estimate for disposable round026 test scratch, and the largest retained files, run the launcher in storage mode.

```bash
git show "$rev:$validation/launch-round027.sh" | bash -s -- --round026-space
```

To remove the eligible scratch, add the explicit apply flag.

```bash
git show "$rev:$validation/launch-round027.sh" | bash -s -- --round026-space --apply
```

Cleanup holds the round026 session lock and checks its original handoff against round027 before and after removal. Only worker directories recorded as qualification test scratch are eligible. Preparation products, failure receipts and training inputs keep their original paths.
