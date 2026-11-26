# Round028 workstation timing and spending decision

**Retain D and stop before the 4M pilot.** The standalone reader verified all 40 original case clocks and their independent scorer dispositions. Candidate-specific timings project 10 h 21 min 37 s for the registered matched pilot. Its eight-hour shift is short by 2 h 21 min 37 s. The scientific comparison is unmeasured. The verified result is a resource decision.

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

| Verified object | Identity |
| --- | --- |
| Timing producer | `bf9d223501fa3bd6281b6c767c3d963c7587afab` |
| Returned timing ZIP (SHA-256) | `9e1b326b61aebf331833badb0f6ff745b7a53ba4875bc86a5238b88cc3baccae` |
| Timing registration | `sha256:76b37c68936e85d5f692c7c572b1a65f7b02afb746e716a762f8a71f27578042` |
| Timing report | `sha256:ec4f586852826843a3315f2355426acc057e539aee0baafbf7e2eedcb48c7ce1` |
| Terminal implementation | `f0b44acaf7d27591304456f99c4f2e19151a81ba` |
| Final public ZIP (SHA-256, 42,882 bytes) | `af2ec6fcff0fdd32886162a6cf272249fb0adc9e0168e41659c77099000982fe` |
| Final public report | `sha256:dd4faada77e3168e3496ab885b3dbc0e86a2f140f3c098eaa38adcc9a16bbb97` |

The nested evidence preserves the original 128-case audit, its 64-case spending decision, both checkpoint restores and all 576 fresh reservations. The historical route found 51 witnessed proposal failures and zero proved selection opportunities. The timing result closes the selected generator branch under this registration.

## What has been verified

An independent Python process with only the final ZIP and the pinned reader reproduced the historical routing bounds, the timing forecast, the `INCOMPLETE` evidence verdict and the `RETAIN_BASELINE` action. The final ZIP contains the decision and the four public JSON reports from which it was derived. Rebuilding it from the original timing ZIP produced identical bytes. The fresh-panel gain and retention fields are null.

The workstation action reopens the saved visible/scoring channels, both full-state checkpoints, D's original evaluation weights, 40 portfolios, all ten search/scoring roles and their original budget debits. The collector includes a separate receipt identifying the committed private replay and checks the copied result independently. An interrupted publication and repeated invocation were exercised with fixtures. Installation and replay on the original workstation await its receipt. [Workstation command](../quickstart.md#record-the-stop-decision-on-the-workstation)

The clean implementation checkout passed 262 local regression tests. Four tests require the native qualification image and skipped locally. Coverage included sdist and installed-wheel packaging, 013 Optuna feedback, original archive readers, the matched generator, existing same-list selector controls, training recovery, freeze recovery and the new workstation receipt. On a detached donor checkout without the private campaign, the pinned launcher returned metadata-only preflight and status and twice produced the expected zero-GPU diagnostic. A separate checkout fetched by the public verification command reproduced `VERIFIED INCOMPLETE RETAIN_BASELINE` from the real final ZIP.

| Checklist | Total | Checked | Unchecked | Status |
| --- | ---: | ---: | ---: | --- |
| Requirements | 16 | 16 | 0 | Pass |
