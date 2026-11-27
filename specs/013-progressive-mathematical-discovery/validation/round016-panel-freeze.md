# Round016 preparation repair and panel freeze

Evidence: `prepare-research(20261116-031626).zip`, qualified source
`sha256:e3592c86ab55c8aad83763cbaae686eaca895bfe58b560e1b64b422231b17ce5`,
commit `2faa8e43fb772b7ad5d4232fe99ff93b3d96b81d`.

The original-reservation repair completed canonical preparation in 2,970 seconds.
Its resource estimate is READY with no deficits. Freeze terminated after 192
seconds with `ValueError: cross-panel detectable related group leakage`;
acceptance and scientific execution remain blocked. Completed qualification,
trajectory measurements and original calibration are retained. The error occurs
while augmenting panel membership with value-derived detector groups, before
campaign publication. The ZIP intentionally excludes private cohorts and truth,
so it cannot identify the actual conflicting rows.

An independently reproduced allocation defect is a plausible cause: two unseen
ASTs may share the same normalized proper-subtree/operator composition identity.
The previous allocator grouped composition-stratum matches but missed pairs
consisting only of unseen structures. Canonical freeze detects those pairs.
This is not evidence that the guard should be weakened, nor proof that it is the
only conflict in the retained workstation data. The new allocator preserves those
components, and new materials publication runs the canonical freeze check early.

The diagnostic module `progressive_panel_audit.py` can be fetched into run-state
and executed with system Python `-S` against the unchanged qualified checkout.
It checks original prepared references and private prompt/truth hashes, uses the
installed detector, reports direct overlaps by category and panel pair, and
continues independent rows after malformed inputs. Example identities are hashed;
raw truth/programs and assessment panels are not exported or scored. This is a
bounded-scope diagnostic, not a complete history scan or freeze replacement.
Row-pair counts are explicitly non-additive across shared keys.

Run from the existing pinned checkout (do not pull/switch source):

```bash
git fetch origin 013-progressive-mathematical-discovery &&
mkdir -p run-state/discovery-tools &&
git show origin/013-progressive-mathematical-discovery:src/oeis_learn/pipeline_validation/progressive_panel_audit.py > run-state/discovery-tools/audit-round016-panels.py &&
python3 -S run-state/discovery-tools/audit-round016-panels.py \
  --session run-state/discovery/013-research-round-016
```

Share `run-state/discovery/013-research-round-016/share/panel-audit.zip`.
Exit 3 means overlaps or incomplete diagnostic inputs were recorded, not that
the ZIP was lost. Failure packaging uses the original session infrastructure.
The original prepare/freeze journal entries and input files remain unchanged;
an additional audit step records this inspection. Reinvocation reuses a completed
audit. No images, tests, calibration or training are launched.

Validation: 54 material/census/revision checks passed, with two actual-native
checks requiring the pinned image and skipped locally. Another 73 calibration,
freeze and cohort checks, four packaging checks and 19 CI inventory checks passed
(150 passed, two native skips in total). Four audit tests also
passed against the actual pinned package. A Python 3.13 `-S` end-to-end invocation
on that checkout emitted the expected composition conflict ZIP, reused it on
reinvocation, and left the source digest exactly unchanged. No HIP or scientific
success is inferred. The actual retained-panel diagnosis remains externally
blocked on the workstation audit; no redraw or weakened exclusion is authorized.

## Audit v2 publication lookup repair

The uploaded `panel-audit.zip` contains no panel inspection: v1 failed in 0.045
seconds because it expected `prepared.json` directly in journal outputs. The real
wrapper journals `result.json`, whose `metrics.prepared` reference binds the
prepared document relative to that result's directory. The initial integration
fixture modeled this incorrectly. The v2 resolver follows and verifies the real
chain, including the completed operation, output binding and both file hashes.
It successfully replayed the original uploaded 56,556,304-byte document.

The corrected test obtains its journal from production `Workflow.run_command`,
doubling only external process execution. The CLI uses `panel-audit/inspect-v2`
so the failed v1 step and binding remain intact. The new step cannot retry any
qualification/preparation/scientific operation. Four audit checks passed on each
of current and pinned packages; an actual pinned Python 3.13 `-S` invocation
confirmed the v2 failure-to-diagnostic path, ZIP, repeat reuse and unchanged
qualified source. Fetch the updated module with the command above and run the
audit again; no checkout change or expensive preparation repetition is needed.
