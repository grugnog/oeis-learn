# 010-followup-002 target evidence review

The supplied `1.log` and `2.log` end in PASS for their respective CPU/native and HIP suites. Both `models.zip` and `matched.zip` have matching payload byte counts and SHA-256 hashes. The models invocation passed all recorded stages, including current-source CPU/native/HIP qualification and complete probes of ordinals 2, 5 and 6. Each probe contains 33,024 completed, unique census rows, and records no parameter mutation, training or optimizer loading.

## Saved-model observations

Ordinal 6 is the producer's retained selected checkpoint, imported through its original checkpoint/weight hashes. Recalculated nonpadding-token-weighted losses are:

| Snapshot | Training | Parameter development | Structural development |
| --- | ---: | ---: | ---: |
| 2 / scheduled 3-hour | 0.139962 | 0.271705 | 0.300880 |
| 5 / scheduled 24-hour | 0.004035 | 0.671913 | 0.936048 |
| 6 / retained selected | 0.133334 | 0.271048 | 0.285595 |

The selected checkpoint has 5,617/32,768 exact canonical training targets, 0/128 parameter-development targets and 2/128 structural-development targets. It reproduces the best pooled development-loss region identified in the original validation journal. These remain teacher-forced canonical-target measures; new semantic continuation results are unavailable.

Token-role denominators are now available. On parameter development, the selected checkpoint has 436/1,640 incorrect integer-category tokens and 392/6,226 incorrect operator/control-category tokens. The final checkpoint has 410/1,640 and 334/6,226 respectively. Integer-token errors are more frequent per token in this split, but non-integer errors remain substantial; these observations do not establish a unique failure mechanism.

## Evaluation failure and repair

Matched evaluation passed all prerequisite stages, then all three generation roles failed after approximately three seconds with `ValueError: registered full-clock portfolio required`. No new semantic scores were produced. The successful report-writing receipt does not override the failed generation receipts or invocation FAIL.

The outer generation adapter accepted the new matched policies, while the inner `_portfolio` guard still accepted only `portfolio-clock60/v1`. Repair: share registered policy/attempt validation with the clock module, so the inner generator accepts both matched policies while continuing to reject unknown policies, mismatched attempt counts and changed portfolio durations. Move the optional Torch sampler import to the cache-comparison path so the actual portfolio state machine can also be exercised by standard-library host regressions.

New regressions drive the actual `_portfolio` generator through every mode under all three registered policies, using deterministic generated candidates and service stand-ins. A separate regression verifies independent raw evidence survives fitting expiry. These are host control-flow tests, not new HIP/native scientific measurements. All ten follow-up host tests and eight pipeline-contract tests pass locally. The implementation fingerprint was refreshed with the package's canonical serializer after the package change.

The helper now accepts `--start-step 4` in a new output directory. This omits the earlier standalone qualification/model commands; each new evaluation invocation still qualifies current source. Retain `010-followup-002` unchanged, and rerun matched, matched-drain and training against the repaired source under a fresh root such as `010-followup-003`.
