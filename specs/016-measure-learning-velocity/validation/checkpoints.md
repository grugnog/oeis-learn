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

The archive-only estimate replays the conditioning ZIP and reconciles the checkpoint, controls
and matched-bank subtotals. From a checkout containing this reader, run

```bash
python3 -I -S -B specs/016-measure-learning-velocity/validation/qualify_continuation.py \
  --estimate-conditioning-zip /path/to/conditioning.zip
```

For the returned partial ZIP, the priced work on 00/04/05/06 totals 29,026.014955 s
expected and 98,054.130824 s at the observed tail. The separate follow-up reserve is
101.545884 s. Paired conditioning, trajectory overhead and verification/publication
still have open price entries. The mount snapshot in this ZIP covers one conditioning
role. A full-wave estimate needs the simultaneous owners on the actual mounts.


## Recover conditioning and price the supported trial

From the detached round027 donor, set `ROUND029_REV` to the full commit supplied with the
launch-pricing reader. The following command uses the original 75-minute conditioning grant
and authorizes a separate, cumulative 15-minute CPU accounting rehearsal.

```bash
(
  set -euo pipefail
  reader=${ROUND029_REV:?supply the full launch-pricing commit}
  git fetch --no-tags origin "$reader"
  git show "$reader:specs/016-measure-learning-velocity/validation/qualify_continuation.py" |
    python3 -I -S -B - --review-launch "$reader" 15
)
```

The pinned repair reader reopens the three paid HIP searches and can finish their three CPU
scorers against the original barrier. Each retry uses the same worker, reader and grant.
The CPU rehearsal uses the exposed stream’s batch-16 and batch-32 row order and lengths.
It times durable update debits, cold journal reopening and stream replay at the whole updates
crossing 1M and 4M useful tokens. Its disposable ledger simulates token charges. The actual
qualification grant charges elapsed time, dispatch attempts and reserved disk with zero
optimizer tokens. Keep the same pricing commit and minute cap to resume the CPU work.

Return the printed `launch-prices.zip` even if the command exits with a partial result.
It contains the original conditioning ZIP, the independent CPU grant balance, the capacity
projection and the available launch estimate. A missing or rejected price stays open.
To verify the copy offline, run

```bash
python3 -I -S -B specs/016-measure-learning-velocity/validation/qualify_continuation.py \
  --verify-launch-zip /path/to/launch-prices.zip
```

The full projection assumes a cached image, serial roles, a dedicated eight-family paired
sample for each eligible profile and retained outputs through publication. It prices separate
destination mounts, their byte and inode reserves, four checkpoint trees, hourly saves,
a pending checkpoint, worker staging, portfolios, journals and staged and published output.
The smaller ordinary prefix has a separate mount check. The publication estimate projects
the authenticated private verification times and the historical release encodings onto the
observed output sizes and member counts. All-in time remains unavailable while any price
is missing. A completed preparation ZIP supports a funding decision. The pre-score freeze
and trial runner have separate acceptance conditions.


## Returned four-recipe launch estimate

The conditioning reader verified all 15 members at worker `c561004224f380d3c3b246d9b57f013a6559b2ff`.
The repair reused all three original HIP searches and completed their CPU scorers. The original
grant charged 182.209383 s cumulatively, including 175.041633 s of device leases. The three
eight-family paired panels yielded zero selected solves in either input condition. Common
ordinary had eight measured neither-solved pairs. Full ordinary had five such pairs and three
unresolved. Full D had seven and one unresolved. These exposed results price preparation.

The launch reader at `1f3dbc3059678f4fc92e9db3d9d23688f660bffb` verified all four payloads.
Its embedded conditioning ZIP matches the returned standalone ZIP byte for byte.
The batch-16 and batch-32 CPU bookkeeping runs together charged 19.204501 s and zero research
tokens. Each replay reached 4,000,117 useful fixture tokens, in 2,756 and 1,378 updates respectively.

| Cached four-recipe planning scope | Expected seconds | Observed-tail seconds |
|---|---:|---:|
| Screen and conventional controls | 29,020.587401 | 98,048.492302 |
| Paired search, selected scoring and banks | 438.238775 | 515.190637 |
| Additional trajectory overhead | 84.756364 | 85.782211 |
| Projected private verification and publication | 507.637462 | 771.302959 |
| Total | 30,051.220002 | 99,420.768109 |

The follow-up reserves 101.545884 s for matched ordinary and challenger OEIS scoring. The tail
including that reserve is 99,522.313993 s. Publication uses a historical scale projection.
The two preparation grants confer zero research authority.

| Returned mount, bytes | Available | Required | Reserve | Deficit |
|---|---:|---:|---:|---:|
| Full wave | 13,325,987,840 | 369,573,318,126 | 5,368,709,120 | 361,616,039,406 |
| First ordinary cut | 13,325,987,840 | 11,791,980,254 | 5,368,709,120 | 3,834,701,534 |

The mount had 118,135,552 available inodes. The wave required 127,904 and the cut required
3,082. These observations are dated snapshots. Recheck the destinations before dispatch.

| Evidence | SHA-256 |
|---|---|
| Conditioning ZIP | `d1a54ad97d21fd18dd84f362883d3d103b0523a9682a92785cde00909faea0b1` |
| Launch-pricing ZIP | `63b30739c0bfdaf968652248bd2e2afc545da1db24f057ad7800749d540ccab0` |
| Conditioning receipt | `f7bd91485c8de8e7b5e3f8576715fa638a8592a979590757e3f544546812690c` |

The installed preflight reopens these original grants and the paid search, barrier and scorer
receipts. It reports a separate zero-token initialization in an empty checkpoint directory.
The writer installs one generation before any later staged checkpoint can coexist. Keeping
the complete observed three-generation tree, its factor-of-two guard, the learner journal
and the runtime gives 5,916,677,854 bytes on the returned mount. That leaves 2,040,600,866
bytes above the recorded reserve. A new capacity observation is required at dispatch.

## Qualify the ordinary zero-token boundary

Run this block at the physical root of the same detached donor. Set `ROUND029_REV` to the full
initialization implementation commit. The 30-minute ceiling covers one scoped qualification,
including host roster preparation, offline wheel packaging, publication and the device role.
Retries with the same commit and duration use its original remaining balance.

```bash
(
  set -euo pipefail
  umask 077
  rev=${ROUND029_REV:?full ordinary-initialization commit}
  out=$(mktemp -d "$HOME/round029-init.XXXXXX")
  git fetch --no-tags origin "$rev"
  git show "$rev:specs/016-measure-learning-velocity/validation/qualify_continuation.py" |
    python3 -I -S -B - "$rev" 30 initialization --output "$out/initialization.zip"
)
```

The command reopens the paid launch prices and family census, freezes the seven-recipe look
graph and its three measured exclusions, and binds the actual wheel, image, device, seeds and
follow-up reserve. It writes `round029-qualification.zip` and a separate copy under the original
grant's authorization directory before the exposed worker starts. The worker writes one initial
ordinary checkpoint and restores it twice in the same lease. A private read rehashes the
retained state blobs and checks the original debits. The command also runs the installed
archive-only verifier on the canonical zero-cut report and the separately retained pre-score
ZIP. The funded scope ends at zero optimizer
updates. Comparative scoring and the full-wave trial need their own acceptance and authority.

Return the printed `initialization.zip`, including after a capacity stop. Its public copy of the
pre-score archive contains look digests and counts. Private case IDs and weights stay on the
workstation. The standalone reader accepts `--verify-initialization-zip RESULT.zip` and an
optional final path to the separately retained qualification ZIP for byte-for-byte verification.
