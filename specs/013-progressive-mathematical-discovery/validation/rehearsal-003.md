# Workstation rehearsal 003

The operator's `rehearse(2).zip` has SHA-256
`591f7d42013199b90be5dc1e30a6f43820dbe8a584b3bd2f0bed8b4e7d5b8a89`.
All 140 inventoried files match their original sizes and hashes, with no
collection omissions. The session is `run-state/discovery/013-hip-003`, source
`sha256:ecb3c3807e8b9a82d8638982e0c2eb138f32772e48242574e1de85bfff8e63f4`.

Both images built. All 2,310 CPU/native tests passed (ten deselected, 82 subtests
passed), as did all 27 actual-HIP preflight tests and eight assessment contract
tests. The original base qualification is PASS, identity
`sha256:88afcf2c5216f8c9426692715cee897b42a04ba5b3638f255cd288f9bff61b03`.
Its device facts identify gfx1151, PyTorch `2.9.1+rocm7.2.1.gitff65f5bc`, HIP
`7.2.53211-e1a6bc5663` and 120,259,084,288 device-visible bytes. This base result
is scoped to its original source; it does not complete selected-model or
campaign qualification.

The first two selected matrix cases (input lengths 4 and 20, microbatch 7)
completed their update/restore/generation paths but failed the final allocated
memory check. No selected-case PASS producers were published. Recording the
second failure then raised `fresh publication required`, preventing the two
remaining independent cases from running. Preparation, freeze and campaign
acceptance stayed blocked. T082 and T088–T111 remain pending their original
requirements; no sustained or scientific result is inferred.

## Repairs

- A real CPU reproduction with weak references showed the 20-input encoder
  memory retained by the cleanup check's own `locals()` snapshot on Python 3.12.
  Initialize the optional memory explicitly and delete it without a snapshot.
- Release hipBLAS workspaces before the baseline and after owned tensor cleanup,
  following [PyTorch's versioned HIP documentation](https://docs.pytorch.org/docs/2.9/notes/hip.html#hipblas-workspaces).
  This addresses a documented allocation source not released by `empty_cache`;
  the uploaded logs alone do not quantify how much of the 4-input residue it
  explains. Actual-device confirmation is still required. The strict
  allocation-at-most-baseline gate remains unchanged.
- Retain `measurements.json` before evaluating cleanup success, including
  allocation counts before workspace release, final allocated/reserved bytes
  and allocator statistics. Failed cleanup cannot publish a PASS producer.
- Qualification and calibration now publish separate immutable failure records
  as they occur, and write the aggregate once. Repeated failures continue to
  independent cases after a bounded successful device-health probe; a failed
  health probe is retained and stops further device work. Failure records and
  allocator measurements are included in the support ZIP even if aggregation
  never completes.
- Refresh the current-dispositions implementation digest without changing its
  historical outcomes or completion claims.

## Local verification

All 29 cleanup, workstation and qualification-contract tests passed with native
requirements enabled. The four new cleanup cases exercise real CPU updates,
complete checkpoint continuation, generation and native scoring with explicit
HIP-boundary/allocator doubles. Both input lengths release their memory and
workspace; both reject deliberately retained allocations and preserve failure
measurements without a producer. These doubles provide no actual-HIP evidence.
Repeated qualification/calibration failures, eventual successes, health failure
and support collection are covered. The collection test was rerun after adding
checks for per-failure snapshots and failed cleanup measurements and passed.

All 58 adjacent acceptance, calibration and pipeline-contract checks passed;
the expensive installed all-stage suite was explicitly deselected. No local
image build or HIP execution occurred. `git diff --check` passed. CI was not
awaited. The next unchanged-source workstation rehearsal must verify these
repairs and reach selected-model qualification and end-to-end acceptance.

The [retained inputs manifest](research-input-resolution.json) remains committed;
[research registration and preparation](research-inputs.md) are still pending.
