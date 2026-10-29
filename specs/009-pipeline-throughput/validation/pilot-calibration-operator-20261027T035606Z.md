# CPU repair passes; UID operations timeout leaves calibration unstarted

The [machine review](pilot-calibration-operator-20261027T035606Z.json) verifies all
152ZIP members (151inventoried payloads), no omissions and no hash/size mismatch.
Exact source is32e77120…; elapsed1485.418s. All819CPU cases now pass, together
with24actual HIP,45policy,123retirement and1115workflow cases/9skips. Actual
private prepared-data/three-snapshot import passes again. The read-only CPU
fixture defect is closed at this producer; no previous failure is erased.

The only failed prerequisite is `pilot.operations.podman`: its arbitrary-UID
export fixture exceeds the original60s external deadline. `process.json` retains
a TimeoutError, -9client termination and60.258s elapsed; both stdout and stderr
are empty and no device/ready/export attestation is present. Cleanup is recorded
within its allowance (0.799s); total through cleanup is61.085s. A subsequent
`podman inspect` exits125 and overrides the primary timeout in the stage reason.
The archive does not contain that inspect stderr. It cannot establish whether
container creation, process startup, pre-attestation imports or another engine
operation stalled. The root cause is **UNKNOWN**, not a proven model, UID mapping,
GPU or native fault. The same UID fixture passed earlier213151Z/030746Z collections.
The stage disk observation has no breach and minimum212307779584free bytes;
there is no disk-full evidence in this failure.

Calibration is BLOCKED and none of the three evaluations starts. Cost stays
UNKNOWN and the24-hour learner/48-hour total pilot remains unadmitted. No
learning result or evaluation timing can be inferred. Preserve all output trees;
there is no calibration work, data generation or training to repeat.

## Scoped diagnostic repair

`scripts/profile_bigint_pipeline.py` now retains the original launch exception
when inspect fails, exports bounded inspect return code/stderr/stdout/error and
an INCOMPLETE isolation audit, and retains the shared-deadline reclamation
acknowledgement instead of replacing it with elapsed time alone. A successful
launch without inspection still fails; an unacknowledged cleanup still raises a
fatal ResourceBreach; an original fatal error is never downgraded. No automatic
retry, time extension, isolation exception or silent prerequisite bypass is added.
All model/data/numerical/scientific budgets, UID60s and shared cleanup5s remain.

This improves failure diagnosis; it does **not** claim to fix the unknown Podman
stall.31scoped supervision/soak cases pass, and all242stdlib host tests pass.
Tests explicitly cover inspect exit125 after an original timeout/fatal error,
successful launch with missing inspection, and failed reclamation remaining fatal.
Only the host launcher and its owning tests change in the761-entry source closure;
new source is3b084306… . Current affected target checks remain required.

Use the unchanged [single calibration command](pilot-calibration-implementation.md#run-once-from-the-existing-checkout)
after pulling, with a fresh output and the complete213151Z producer as `--inputs`.
This is one bounded invocation, retaining all failures and the successful older
source evidence; no manual operations/profiling loop is added. If the UID stall
recurs, the primary timeout plus available inspect diagnostics will now survive.
K006 remains partial and T110 remains pending for the changed source; original
87/9/15 and all learning/default/carryover dispositions remain unchanged. No CI wait.
