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
