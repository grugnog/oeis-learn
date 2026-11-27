# Round029 workstation qualification

The detached donor completed the learner update and restore rehearsal on candidate `73009e595e589a1f6780cec06d5d16989ee8dcd5`. The first attempt interrupted one backward pass at each batch size. Both resumed runs reproduced the uninterrupted final weights, AdamW state and next rows. Each scheduler recorded two useful updates and retained the discarded token charge.

| Logical batch | Useful tokens | Discarded tokens | Tokens charged on recovery trajectory | Matching on the final update |
| ---: | ---: | ---: | ---: | --- |
| 16 | 391 | 200 | 591 | Recipe 04, matching disabled |
| 32 | 791 | 400 | 1,191 | Recipe 02, 16 eligible pairs and 4 matching passes |

The journal charged 2,964 tokens across both uninterrupted references (1,182) and both recovery trajectories (1,782). The native fitter could reground the literal on constant observations and accepted the varying-term control as a negative.

| Measurement | Donor result |
| --- | --- |
| Wall time and budget | 188.091933 s charged out of 1,800 s. The grant retained 1,611.908067 s and zero training tokens. |
| GPU lease and activity | 140.782421 s leased. All 61 telemetry samples were readable, with 5.356970 s busy over 59.999875 s of coverage during the 60.757373 s worker interval. |
| Peak device memory | 3,414,627,328 bytes, 1,437,761,024 bytes below the historical 4,852,388,352-byte peak. |
| Full-state checkpoint allocation | 2,937,348,096 bytes and 29 inodes for each batch, measured separately. |

The returned preflight, qualification and status documents passed the candidate's result validator. The receipt matched the last committed reference in both replies. Recomputing the saved status digest reproduced its recorded identity. The candidate revision, donor root identity, role policy and 30-minute grant reproduced the binding in the receipt. Preflight reported fresh family and matching-pair supply as unmeasured. The run dispatched zero research operations and kept the final assessment sealed.

## Donor replay

The donor reopened the seven retained artifacts twice under the original qualification grant. Both launcher replies matched the entire first result, including its receipt and remaining budget. The review helper returned the following record.

```json
{
  "candidate": "73009e595e589a1f6780cec06d5d16989ee8dcd5",
  "receipt": "sha256:2e4f54628a8a6d9d1cfb4349f333e530acdedd944e7c7f76677d392c142a6e5b",
  "verified_replays": 2,
  "elapsed_us": 188091933,
  "charged_tokens": 2964,
  "research_dispatches": 0
}
```

The remaining release checks include measuring CPU contention against a serial reference (T059) and independently verifying the copied report (T062).

## Ordinary 4M continuation on HIP

The detached donor restored the round028 ordinary checkpoint at 4,000,210 useful tokens on candidate `cfe8230a7a4427fb29cb852c32819404eed7fdf5`. Both the round028 policy and round029 recipe 00 consumed the same next 1,647 positive tokens. They reached 4,001,857 useful tokens with identical updated weights, AdamW state, scheduler counters, RNG state and next row order. The two disposable updates charged 3,294 tokens. Two subsequent replays returned the original receipt.

| Workstation measurement | Observed |
| --- | ---: |
| Cumulative qualification budget | 1,800 s |
| Charged elapsed time | 182.685033 s |
| Remaining budget | 1,617.314967 s |
| GPU lease | 126.814207 s |
| Peak device allocation | 2,912,811,008 bytes |

The helper returned this summary after verifying the private receipt and checking the copied report against the independently retained pre-score archive in a separate process. The result covers one disposable continuation of recipe 00. The seven-recipe study still requires its own registration, funding and results.

```json
{
  "candidate": "cfe8230a7a4427fb29cb852c32819404eed7fdf5",
  "receipt": "sha256:c785fdefae23440cf74b532ea75142b7d0e72b18bcf979803d6c5878f2565154",
  "report": "sha256:6da4b6691e79f675dc91ee347da3cb4c3997a254463c43cd3966adeca12d18e3",
  "parent_cut": "sha256:c8637061f54dc583d501f782a8bf33038baa2805bfc6a20f8807de4e7963daab",
  "parent_useful_tokens": 4000210,
  "next_update_tokens": 1647,
  "charged_tokens": 3294,
  "elapsed_us": 182685033,
  "remaining_us": 1617314967,
  "lease_us": 126814207,
  "peak_device_bytes": 2912811008,
  "verified_replays": 2,
  "research_dispatches": 0,
  "scientific_verdict": null
}
```

## Reopen the saved qualification

This command reproduces the receipt check from the same detached donor root using the helper and candidate commits from the verified run.

```bash
(
  set -euo pipefail
  review=19f27afc7f4821e7c46d537bf3a33caae036e6bf
  candidate=73009e595e589a1f6780cec06d5d16989ee8dcd5
  git fetch --no-tags origin "$review" "$candidate"
  git show "$review:specs/016-measure-learning-velocity/validation/replay_learner.py" |
    python3 -I -B -
)
```
