# Collect existing evaluation timing diagnostics

The preparation ZIP currently omits `candidates.json` and `scores.json`. When
timing fails its completeness check, its PASS producer/timing files are never
published. This read-only observer recovers the diagnostic information from the
original training-only generation/scoring files without repeating any execution.

From the repository root:

```bash
python3 -S tools/research_timing_diagnostics/collect.py \
  --session run-state/discovery/013-research-round-002
```

Share the new ZIP whose full path is printed. It is written under the session's
`share/` directory with a unique name. No Podman, packages or GPU are needed.
Each of the fourteen cases is collected independently; missing, malformed,
oversized or symlinked records are reported in the ZIP and cause exit status 1.
Share partial ZIPs too. A missing/invalid timing request stops collection because
the collector cannot establish that these are training-only diagnostic cases.

Exports are **projections**, not original authority records: each includes the
original file's local path, byte count and SHA-256, plus a separate hash of the
exported projection. They retain original completion reasons, attempt counts,
token counts, generation/native timings, portfolio clocks and scoring statuses.
Programs, token sequences, assignments, native admission payloads, sequence
values/indices and checkpoints are not exported. Original files remain unchanged.
The companion original preparation ZIP supplies request, runtime, container,
qualification, failure logs and other provenance.

This directory is outside the existing production qualification source snapshot,
like `tools/oeis_diagnostics`. It is never imported or executed by production
roles. It cannot grant qualification, alter budgets or resume research. No source
exclusion rule was changed. Adding this observer preserves the original source
identity `sha256:e7211c5a74c35072a49f07707dbe21595d40d9873bccb8b2cabef2e579629118`.
Do not move production fixes here to evade qualification.

Tests (stdlib only):

```bash
python3 -S -m unittest discover -s tools/research_timing_diagnostics -p 'test_*.py' -v
```
