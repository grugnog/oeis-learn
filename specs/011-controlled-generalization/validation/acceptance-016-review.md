# Acceptance-016 evidence review

Reviewed 2026-11-01 Pacific against ef886a791a9dd3f9807174858f9cbd2e0ba726cf.

Archive: acceptance-016.zip, SHA256
`cd0ae831d6098a9e0655fc18b79fb81419c8c7058893415ca64ba9a4d7e39c84`.
All550 listed payload sizes/hashes match. The installed Python
`generalization_archive.verify_archive` independently returns integrity PASS,
software_status PASS, scientific_status UNMEASURED, feature_qualified true,
all five cases PASS and no missing cases. This includes final workflow/verifier
binding and the original qualification/feature evidence, not just summary text.
All905 source-snapshot entries equal the published production checkout.

| Evidence | Observed result |
| --- | --- |
| CPU/native suite log | 1196 passed,77 subtests passed,8 deselected;375.22 seconds |
| HIP suite log | 27 passed;117.51 seconds; no skipped cases in JUnit |
| Real-device proof | HIP forward/backward finite, parameters changed,2 updates, resume_equal true |
| Retained evaluation | Actual weights loaded, visible-only inputs, matched budgets |
| Refresh | Fixed support, fresh shards, finite updates |
| Targets | Literal and parameterized updates, grounding verified |
| Replay/resume | Training-only admission, paired control, finite updates, full-state resume |
| Failure packaging | Failure preserved, archive verified, private payloads excluded |
| Final workflow | All10 canonical stages exit0;1430.071 seconds cumulative (23m50s), excluding builds |

The paired resume proof records456 tokens/3 updates in each branch, identical
initial weights and final weight hash
`sha256:211583257ba31cbe2a6d347f45c0abb14574e4c1f6bbf913bd4b1352755f0182`.
Distinct checkpoint descriptor hashes are not a numerical mismatch. The proof's
full-state comparison is true; exactness has not been replaced by a tolerance.

Runtime: PyTorch2.9.1+rocm7.2.1.gitff65f5bc, HIP7.2.53211-e1a6bc5663,
host driver/kernel observation6.18.50, AMD Radeon Graphics. Execution policy:
`sha256:7e8b6150696841d883936a6599c05d80b7055cb6d8c518846c29bc0897f5d71d`.
Immutable qualification image:
`sha256:86138e95db1d4b892fd943b347884abc1603f6cb052d37af59d93ba0131878db`.
Immutable GPU image:
`sha256:f07fd131792156826b3da99393e515ebf43c07051f9217e8014abdeff9ff30fb`.
Original qualification identity:
`sha256:96f88ec92aabff7a59ff9d39d0f7c868007e9aa8e4e78a74bd80f7f150ef5962`.

## Gate reconciliation

T064: immutable images, original producer/source/build/runtime/security observations
are present. T065: successful canonical preflight and frozen serial stage budgets
are recorded. A separate dry-run console transcript is not in the archive; the
normal invocation itself records and validates the same plan. T066: the full
serial repaired-source session passed. T067: original canonical verify-bundle
completion plus independent local archive reconciliation pass. The embedded
payloads/workflow.json is the earlier PARTIAL packaging-time snapshot;
workflow-final.json is the later PASS terminal record, verified without rewriting
the original snapshot.

Keep run-state/generalization/acceptance-016/report/qualified/ and the full local
private run intact. Sharing archives intentionally cannot reconstruct private
weights, sealed truth or prepared research inputs. The checked-in documentation
records this evidence; it does not replace original qualification artifacts.

## Decision and limits

Keep the current image and measured no-Lt policy for the first scientific run.
A runtime upgrade can be tested separately on saved operands before promotion;
it is not a prerequisite to proceed. This is software acceptance only, not
measurement of generalization improvement, seed variation or sustained throughput.
The finite short resume proof is not a universal reproducibility guarantee.
No new HIP command, image build, host driver change or CI wait ran during review.
