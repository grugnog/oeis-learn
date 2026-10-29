# 010-followup-001 CPU qualification review

The operator supplied `010.tar.gz`, containing the first command's logs, JUnit XML, stage receipt and validation summary. The serial runner stopped at `bigint-cpu-regressions` after pytest completed; the 300-second message was a progress update, not the reported cause of failure.

Measured result: **810 passed, 1 failed, 7 deselected, 72 subtests passed**, in 321.72 seconds. Stderr was empty. The sole failure was `Contracts.test_current_dispositions_use_the_package_implementation_serializer` in `tests/unit/test_pipeline_contracts.py`.

The package implementation identity was `sha256:60f1a6572caa5038ce7adc16040b54e25172a26d0691e0c9dc3c8834345b3ee1`; the checked-in current-dispositions register still contained `sha256:b5b9b45be0bfd8aa9e73e6ed89807fc131cfe3739d9900c02f05bd9435f440f1`. The earlier package changes had not refreshed this mandatory register fingerprint.

Repair: refresh only `implementation_sha256` using the actual `canonical_bytes`, `compute_canonical_digest` and `implementation_identity` function definitions, with the same dependency-free execution method as the failing contract test. All 68 dispositions, their task states, historical references and `full_007_complete: false` remain unchanged. This metadata maintenance grants no new readiness or scientific acceptance.

Local validation reproduced the same failure before the repair. Afterwards, all eight pipeline-contract tests and all seven follow-up host tests passed. The operator's full native/CPU suite must still be rerun against the repaired source. No HIP qualification or training result is inferred from this archive.

Retain `010-followup-001` as failed evidence. Run the serial helper with a new output root, such as `tmp-validation/010-followup-002`, after updating the branch. The helper stops on failure; it does not retry or overwrite prior evidence automatically.
