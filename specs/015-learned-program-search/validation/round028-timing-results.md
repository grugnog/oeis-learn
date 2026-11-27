# Round028 workstation timing and spending decision

**Round028 closed with D retained.** The workstation checked all 40 timed portfolios and their independent scoring records. Candidate-specific timings project 10 h 21 min 37 s for the registered matched pilot, exceeding the eight-hour shift by 2 h 21 min 37 s. The close-out dispatched zero new GPU jobs. The 4M comparison has zero training runs and zero fresh scored cases, so the effect on accuracy is unmeasured.

| Forecast on the registered cases | One role per arm and look | Eight-case batches |
| --- | ---: | ---: |
| Descriptive 0 and 1M looks for both arms | 3 h 32 min 42 s | 4 h 17 min 24 s |
| Full 4M endpoint, including retention and D | 4 h 59 min 31 s | 5 h 02 min 41 s |
| Pilot including training and overhead | 10 h 21 min 37 s | 11 h 09 min 28 s |
| Pilot shortfall against eight hours | **2 h 21 min 37 s** | **3 h 09 min 28 s** |
| Independent repeat if earned | 11 h 11 min 48 s | 11 h 17 min 02 s |
| Pilot and repeat combined | 21 h 33 min 25 s | 22 h 26 min 30 s |

The earlier estimate based on historical models was 3 h 16 min 29 s for the pilot. The matched-arm training estimate remains 1 h 44 min 13 s. With the actual candidate checkpoints, evaluation alone projects 8 h 32 min 14 s for the two early looks and the full endpoint. The slower observed case per phase in each two-family stratum sets the planning rate. The forecasts include cold role overhead and independent scoring.

| Eight exposed cases per row | Draws started / reserved | Sealed choices checked | Measured search time |
| --- | ---: | ---: | ---: |
| Ordinary, one attempt | 8 / 8 | 0 | 2.10 s |
| First-divergence, one attempt | 8 / 8 | 0 | 423.49 s |
| Ordinary, full profile | 64 / 64 | 0 | 263.72 s |
| First-divergence, full profile | 17 / 64 | 0 | 477.84 s |
| Retained D, full profile | 64 / 64 | 2 | 71.21 s |

The two learned checkpoints were measured after one rehearsal update, on the same eight exposed problems. Across the five search roles, 161 of 208 reserved draws started. The first-divergence decoder spent 475.85 s decoding in its full-profile block. It reached only 17 draws across eight 60-second portfolios. In both profiles its slower case in each stratum cost about 60 s. The scorer recorded 38 empty choices and independently checked the two selections from D. These observations measure cost. Fresh-panel yields, gains, losses, retention effects and confidence bounds remain unmeasured.

The physical collection held the GPU lease for 1,301.93 s across five roles. Telemetry covered 1,244.00 s and observed 379.74 s of integrated GPU activity. The five CPU scoring roles took 58.29 s of host wall time in total. Image acquisition took 41.19 s for HIP and 22.90 s for CPU. These intervals and the per-case decoding and fitting clocks overlap. The registered 4M comparison, independent repeat and longer confirmation receive zero new GPU time. No assessment cases were opened.

## Workstation receipt and public reports

| Verified object | Identity |
| --- | --- |
| Timing producer | `bf9d223501fa3bd6281b6c767c3d963c7587afab` |
| Returned timing ZIP (SHA-256) | `9e1b326b61aebf331833badb0f6ff745b7a53ba4875bc86a5238b88cc3baccae` |
| Timing registration | `sha256:76b37c68936e85d5f692c7c572b1a65f7b02afb746e716a762f8a71f27578042` |
| Timing report | `sha256:ec4f586852826843a3315f2355426acc057e539aee0baafbf7e2eedcb48c7ce1` |
| First workstation close-out | `f0b44acaf7d27591304456f99c4f2e19151a81ba` |
| Returned close-out diagnostic (SHA-256, 548 bytes) | `8568d734338f4d5b07e22c57b4f38c6a5c9bb8d8e8ee20a3c06cc71b8c7f79a8` |
| Journal recovery implementation | `6d4c79adbff84a06bb3d2ddaa196837f4ea59f1b` |
| Final public ZIP (SHA-256, 42,882 bytes) | `af2ec6fcff0fdd32886162a6cf272249fb0adc9e0168e41659c77099000982fe` |
| Final public report | `sha256:dd4faada77e3168e3496ab885b3dbc0e86a2f140f3c098eaa38adcc9a16bbb97` |
| Returned workstation ZIP (SHA-256, 43,787 bytes) | `4b0541f95218c64234d7c44137530d839bd5ddfe1a0731baaa6ef64336c6ece3` |
| Workstation receipt | `sha256:1901e97e71d146db415743f380715f4ad9e96da07f16424bd628663379a80ce9` |
| Private replay of original weights, choices and scorer clocks | `sha256:955cdad79ada616ba7753fa13a9735a37501f44434a17d06b795d6a3895fe133` |
| Committed workstation journal boundary | `sha256:11608719819a05902aa06bfd488726be3eeeef31a4f26947c86b9d05681cb6b3` |

The nested evidence preserves the original 128-case audit, its 64-case spending decision, both checkpoint restores and all 576 fresh reservations. The historical route found 51 witnessed proposal failures and zero proved selection opportunities. The timing result closes the selected generator branch under this registration.

## What has been verified

An independent Python process with only the final ZIP and the pinned reader reproduced the historical routing bounds, the timing forecast, the `INCOMPLETE` evidence verdict and the `RETAIN_BASELINE` action. The final ZIP contains the decision and the four public JSON reports from which it was derived. Rebuilding it from the original timing ZIP produced identical bytes. The fresh-panel gain and retention fields are null.

The returned `round028-final-workstation.zip` contains the workstation receipt and the canonical `round028.zip`. The receipt records the private replay of the original weights, choices and scorer clocks against the committed journal boundary. Independently, the pinned reader verified the returned package from the archive alone. Its enclosed report matched the previously verified public ZIP byte for byte. The reader reproduced the `INCOMPLETE` scientific verdict, the `RETAIN_BASELINE` decision and zero further authorized GPU time. The final assessment stayed sealed. [Archive verification](../quickstart.md#verify-the-final-public-report-on-any-computer)

The first close-out implementation passed 262 local regression tests. Four tests require the native qualification image and skipped locally. Coverage included sdist and installed-wheel packaging, 013 Optuna feedback, original archive readers, the matched generator, existing same-list selector controls, training recovery, freeze recovery and the new workstation receipt.

### Journal accounting and recovery

The first workstation attempt checked the original visible/scoring channels, both full-state checkpoints, D's evaluation weights, all ten search/scoring roles and their budget debits. It installed and verified the final public ZIP. Its journal write then measured the whole session against limits of 64 MiB and 8,192 inodes. Each of the two retained checkpoint trees accounted for 1,257,820,683 logical bytes. The trace ended at the journal quota check, before the filesystem free-space check.

The repaired diagnostic run journal counts the session root, journal records, temporary files and unrecognized entries. Its separately owned `private`, `share` and `accounting` directories have their own storage scopes. The write still checks the original journal limits and the filesystem's available space and inodes. If a publication is pending, the normal run and timing commands can reuse only a completed timing boundary.

A cross-revision test used the authentic public timing return, a private replay fixture and sparse files matching both checkpoint sizes. The original pin installed the real final archive and stopped at its last journal write. A second Python process on the repaired pin committed exactly one transition against that same session. Repeated publication reproduced the receipt and preserved the installed archive's bytes, inode and modification time. The archive digest matched the distributed public ZIP.

The repaired release passed 340 local regression tests. Five tests require the qualification image and skipped locally. The tests exercised journal byte and inode exhaustion, simulated a shortage of filesystem bytes or inodes, and checked the numerical deficits in the diagnostic ZIP. A separate run retained 8,200 private files while committing the journal entry. The exact launcher ran twice on a detached donor checkout lacking the private campaign and returned a zero-GPU diagnostic both times. The documented independent reader fetched the repaired revision and reproduced `VERIFIED INCOMPLETE RETAIN_BASELINE` from the final public ZIP.

| Checklist | Total | Checked | Unchecked | Status |
| --- | ---: | ---: | ---: | --- |
| Requirements | 16 | 16 | 0 | Pass |
