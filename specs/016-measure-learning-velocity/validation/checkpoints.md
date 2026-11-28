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

## Price matched candidate-bank scoring

Run `bank-prices` from the detached round027 root. Set `ROUND029_REV` to the published implementation commit and `ROUND029_CONTROLS` to the `receipt` in the completed isolated-controls ZIP.

```bash
(
  set -euo pipefail
  umask 077
  out=$(mktemp -d "$HOME/round029-banks.XXXXXX")
  git fetch --no-tags origin "${ROUND029_REV:?full bank-pricing implementation commit}"
  git show "${ROUND029_REV}:specs/016-measure-learning-velocity/validation/qualify_continuation.py" |
    python3 -I -S -B - "$ROUND029_REV" 20 bank-prices \
      --controls-receipt "${ROUND029_CONTROLS:?original isolated control receipt}" \
      --output "$out/bank-prices.zip"
)
```

The command authenticates the paid control costs, five original generation and scoring groups, and the previously exposed eight-case roster. It runs a separate CPU scorer for each group. The scorer executes each recoverable distinct admission on all 100 indices using the native and reference implementations. The ZIP reader checks the published bytes, case counts, receipt links and settled charges. The command uses one cumulative 20-minute qualification grant. Rerunning the same candidate and grant duration resumes individual groups.

The measured groups are ordinary and first-divergence at one and eight attempts, and retained D at eight attempts. The ZIP reports missing sources, incomplete searches, empty banks, verified outcomes and timings separately. Its cost scenario uses eight cases per group and explicitly identifies the historical learner proxy. The full trial cost still depends on the prospective sample and its observed admissions. Attach the ZIP from the printed path, including a partial result.

## Measure paired conditioning and recheck the resulting banks

Set `ROUND029_REV` to the published conditioning commit and `ROUND029_BANK` to the receipt from the completed bank-prices ZIP. From the detached round027 root run

```bash
(
  set -euo pipefail
  umask 077
  out=$(mktemp -d "$HOME/round029-conditioning.XXXXXX")
  git fetch --no-tags origin "${ROUND029_REV:?full conditioning implementation commit}"
  git show "${ROUND029_REV}:specs/016-measure-learning-velocity/validation/qualify_continuation.py" |
    python3 -I -S -B - "$ROUND029_REV" 75 conditioning \
      --bank-receipt "${ROUND029_BANK:?completed bank-pricing receipt}" \
      --output "$out/conditioning.zip"
)
```

The CPU custodian authenticates the donated ordinary 4M checkpoint and retained D, then freezes the two conditioning maps for every group. One HIP lease per group covers both input modes. Ordinary runs at one and eight attempts, and D runs at eight. Each probe uses the same inference seed, family order and true recipient for fitting. If an exposed timing pair has identical visible values, the custodian selects a previously exposed differing family in the same stratum using only its visible prompt. The ZIP reports the number of substitutions.

All six sets of choices must seal before the CPU roles score any suffix. Each scorer checks the selected choice and independently executes every recoverable admission with both implementations. The ZIP includes paired selected outcomes, unresolved cases, bank coverage, search and scoring clocks, GPU leases, sampled activity, peak device memory and charged resources. The proxy for other learner recipes uses the historical ordinary checkpoint. The 75-minute grant covers the combined qualification. Retries at the same candidate and duration reopen completed groups and charge interrupted attempts to the original balance. Attach the printed `conditioning.zip`, including a partial result.

### Recover the partial conditioning run

The returned ZIP for worker `c561004224f380d3c3b246d9b57f013a6559b2ff` has SHA-256
`b774f1b56b9a4916dfed6410dee3d7dc5c6abe9d1f9347d9cd7679006d91b29d`.
All eight member hashes match. Each of the three paired searches ended in a verifier
`ValueError`. The ZIP records the exception class. The original errors and candidate
banks are on the donor. Zero groups reached scoring.

| Original 75-minute qualification | Measured |
| --- | ---: |
| Charged time | 178.773252 s |
| Remaining time | 4,321.226748 s |
| Three recorded GPU leases combined | 175.041633 s |
| Charged attempt units | 272 |
| Optimizer tokens | 0 |

Use the recovery reader on the original detached donor. It checks the device and timing
evidence from each paid search against its original request and debit. Once all six sets of
choices verify, the three CPU scorers can run on the remaining grant. A failed check returns
a partial ZIP with the observed discrepancy. The command uses the same grant on each resume.

```bash
(
  set -euo pipefail
  umask 077
  reader=404a652c936eb7cfb4ab3303301edc4c9b5ee33c
  worker=c561004224f380d3c3b246d9b57f013a6559b2ff
  grant=sha256:ee51d1909fe93ce8928d1b930edb209d0e8582be7a20b776eab712bad539f34c
  bank=sha256:fca44fe0dc257caa17f4ea7c9e559891530794fcfba19574397684a79c9c658a
  out=$(mktemp -d "$HOME/round029-conditioning-recovery.XXXXXX")
  git fetch --no-tags origin "$reader" "$worker"
  git show "$reader:specs/016-measure-learning-velocity/validation/qualify_continuation.py" |
    python3 -I -S -B - "$reader" 75 conditioning --bank-receipt "$bank" \
      --recover-grant "$worker" "$grant" --output "$out/conditioning.zip"
)
```
