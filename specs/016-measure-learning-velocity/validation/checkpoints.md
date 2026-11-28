# Measure checkpoint costs for the four ordinary-stream treatments

Run this qualification from the physical root of the detached round027 checkout that completed the fresh-start qualification. Set `ROUND029_REV` to the full checkpoint implementation commit supplied in the handoff. The command reads the original stream, initialization and optimizer policies from that workstation.

```bash
(
  set -euo pipefail
  umask 077
  : "${ROUND029_REV:?full checkpoint implementation commit}"
  out=$(mktemp -d "$HOME/round029-checkpoints.XXXXXX")
  git fetch --no-tags origin "$ROUND029_REV"
  git show "${ROUND029_REV}:specs/016-measure-learning-velocity/validation/qualify_continuation.py" |
    python3 -I -S -B - "$ROUND029_REV" 20 checkpoints --output "$out/checkpoints.zip"
)
```

Twenty minutes is the cumulative ceiling for this separate qualification grant. A repeat with the same revision and ceiling reopens settled stages and charges any retry against the remaining time. CPU packaging and the single HIP stage each check free bytes and inodes against the physical reserve before dispatch.

| Treatment | Effective batch | AdamW weight decay |
| --- | ---: | ---: |
| 00 | 32 | 0 |
| 04 | 16 | 0 |
| 05 | 32 | 0.01 |
| 06 | 16 | 0.01 |

Each treatment starts from the shared weights, writes the initial and first-update states, resumes to a second complete update, then reopens that cut. Verification compares the saved and restored weights, optimizer, scheduler, random state, next rows and checkpoint reference. The worker releases each treatment's temporary checkpoint directory before starting the next one.

Attach `checkpoints.zip` from the printed path, including after an interruption. A completed ZIP contains one JSON per treatment, physical capacity at each dispatch, the receipt, private verification, and the refreshed forecast. Each treatment reports three storage writes, two storage reads, the enclosing phase times, charged tokens, and the largest observed on-disk allocation at the two saved cuts. The receipt also contains the device lease, peak device memory, worker CPU time and sampled GPU activity coverage.

`summary.json` records the grant balance and the SHA-256 of each enclosed file. If a device attempt stops after publishing some treatments, their filenames start with `unverified-` and identify the attempt. A capacity block records the observed deficit. A forecast failure leaves the verified treatments in the ZIP. These two-update measurements supply preparation evidence for T019 and T059. The research screen has its own registration, full-wave capacity projection and grant.

## Collect historical control costs and the supported forecast

From the same donor checkout, use the full revision supplied with the control-history handoff.

```bash
(
  set -euo pipefail
  umask 077
  : "${ROUND029_REV:?full control-history implementation commit}"
  out=$(mktemp -d "$HOME/round029-controls.XXXXXX")
  git fetch --no-tags origin "$ROUND029_REV"
  git show "${ROUND029_REV}:specs/016-measure-learning-velocity/validation/qualify_continuation.py" |
    python3 -I -S -B - "$ROUND029_REV" 0 control-history --output "$out/control-prices.zip"
)
```

This command takes zero grant minutes. It reopens the registered historical development roster and both original capability cohorts, authenticates their generation and scoring records, and groups the three conventional controls by stratum. The ZIP includes observed clocks, attempt coverage, whole-stratum means and maxima when complete, current storage capacity, and the checkpoint-verified four-treatment forecast. The historical clocks cover the recorded searches and selected choices. The isolated round029 CPU service, bank scoring and supervision still require their own cost evidence. A partial ZIP preserves independently verified sections and identifies missing partitions or failed checks.

## Measure the three isolated CPU controls together

Run from the detached donor after checkpoint qualification. Set `ROUND029_REV` to the full isolated-controls commit supplied in the handoff.

```bash
(
  set -euo pipefail
  umask 077
  : "${ROUND029_REV:?full isolated-controls commit}"
  out=$(mktemp -d "$HOME/round029-isolated-controls.XXXXXX")
  git fetch --no-tags origin "$ROUND029_REV"
  git show "${ROUND029_REV}:specs/016-measure-learning-velocity/validation/qualify_continuation.py" |
    python3 -I -S -B - "$ROUND029_REV" 15 isolated-controls --output "$out/isolated-controls.zip"
)
```

The command takes a cumulative 15-minute CPU preparation grant. It binds the checkpoint and installed wheel, reopens the original eight exposed development cases (two families in each stratum), and checks mount capacity. Template, polynomial and linear recurrence each receive a supervised CPU search and a separate scorer. Each search sees only visible prefixes, and each scorer receives its committed candidates and original 100-term truth. This qualification uses zero HIP leases and zero optimizer tokens.

The ZIP contains stage receipts, per-treatment search and scoring clocks, cold overhead, empty-case counts, private verification, charged and remaining work, capacity and the forecast. For planning, the slowest exposed stratum supplies the retention proxy. The full-wave storage registration and protected panels stay outside this measurement. A partial ZIP preserves completed roles. Repeating the same candidate and grant duration reopens those roles and uses the remaining balance for eligible retries. Attach the ZIP from the printed path even when the command reports a partial result.
