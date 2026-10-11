# Phase 13 — convergence fixes and CPU prerequisites

T109–T111 fix the confirmed provenance, G8 status and G0 identity defects. New
owning-path regressions were run before implementation and failed as expected.
Current software/native/wheel acceptance is re-run in this phase's PR; historical
phase-11 and phase-12 reports retain their original bindings.

## Changes

- Generic archive admission, packet reload and frozen measurement loading enforce
  the rejection returned by generator replay. Mutating origin, sampler config,
  seed, counter, statistics or tokens cannot publish objects or change the index,
  even after an outer packet is rehashed. G3/G8 suites include the new regressions.
- G8 reports use `g8/v2`. Requested hardware execution retains process output,
  exit/launch failure, validation reason and any available JUnit. Executed failures
  are `FAIL`; only unrequested work is `PENDING`. Report replay rejects failures
  even in diagnostic mode; the CLI and release readiness remain fail-closed.
  Transport/archive collection includes both hardware process and JUnit files.
- Preflight reports use `preflight-report/v2`. `config_digest` hashes the full
  effective probe settings, including constructor defaults; `base_image_digest`
  identifies the resolved parent image. `final_image_id`, lock and Dockerfile
  byte hashes have separate fields. G0 loads a separately selected preflight
  config and cross-checks its pins against checkpoint provenance, executed
  containment and the observed Python/PyTorch/HIP/device identity. Every measured
  step must use that exact device. Rehashed foreign evidence fails these bindings.
- Source distributions include the current inventory validator, dependency graph,
  current disposition record and documentation needed by extracted-package tests.
  Current dispositions are refreshed for the final inventory; historical records
  are preserved. A checklist or packaged snapshot does not create a gate pass.

## CPU evidence and external inputs

`python scripts/verify_phase13.py --output DIR` freezes the first 256 registered
raw generic counters, records rejected draws, replays provenance, traces distinct
valid sources at indices 0–99 and runs checked/unchecked comparisons only for
independently in-range programs. The existing 4096-bit/50M-step trace cap,
external deadlines, ten warmups and thirty alternating repeats remain enforced.
The driver records actual legacy configuration/wrapper identities and a census
of the available stripped sample, without inferring offsets or metadata.

The checkout contains no trained checkpoint bytes and no authoritative indexed
release manifest. T113 stays unchecked: stripped/names data cannot satisfy
source indices, offsets, finite/table metadata or release-archive acceptance.
T114 stays partial: the available CPU wrapper regressions and genuine bootstrap
traces run, but trained-parent parity, model/verified strata and real sealed
source scoring still require their inputs. These are external-data/artifact
blockers, separate from phase-14 HIP/container and telemetry acceptance.

## Validation boundaries

The focused owning tests pass locally. The broad local foundation suite also
executes, but this sandbox cannot read `/proc/sys/kernel/random/boot_id`; protected
bootstrap, evaluation, accounting and resume tests therefore fail/setup-error.
Those protections remain intact. Fresh Ubuntu CI runs the complete CPU suite,
actual quickstart, Rust/native G7–G9, Docker roles and installed-wheel cases.
Phase 13 is not fully accepted while T113 or T114 lacks mandatory evidence;
phase 14 and full spec 007 release remain pending.

## Executed local checks

The current implementation passes 91 focused owning-path/wrapper/overflow checks,
47 source-policy/admission/integration checks, and 37 extracted-distribution
inventory/handoff/current-binding checks, each with zero failures or skips. The
original and source-distribution-rebuilt wheels have identical bytes for all 181
packaged source/resource files. Lint and the structural validator pass: all 121
tasks and all 52 requirements remain mapped, with 68 original review inputs.

The actual diagnostic driver issued counters 0–255 and rejected 18 invalid draws
under the existing prior, leaving 238 distinct valid raw programs. Provenance was
replayed for every selected program; 108 independently in-range programs received
CPU checked/unchecked timing. Bounded reference execution records 113 runtime
failures and 14 unknown results explicitly. Two diagnostic candidate rescues do
not establish development matching or admission eligibility. Ten historical
configuration files and eleven wrapper files were hashed; no parent weights were
available. The 24-record stripped sample contains 20 full 20-term prefixes, two
full 50-term prefixes and no full 100/120-term prefixes, with all offsets and
keywords missing. None of these diagnostics qualifies T113 or completes T114.

Local CPU input evidence was generated against implementation identity
`sha256:1a89244c8e1f14f668469d47287b13a13d1d6d74d003e9570e919c1415719088`.
Its `cpu-inputs.json` report digest is
`sha256:35c071cece6e93d259a1f6a6be96b9a2a8a240d1f02832cd5c983a1e3a79f02c`;
its overflow report digest is
`sha256:683d94cb07dd9dd3e31df2b0708e6a25da6efc13b9c211bc79c69fb8b1170dee`.
These identities describe this execution; subsequent inventory changes require
fresh evidence, rather than rebinding these historical results.

## CI software requalification

The implementation at `bcc92cd8ebf1c795a642a3b72d9b0ae0779a546b` was tested in
[review run 38113973427](https://github.com/grugnog/oeis-learn/actions/runs/38113973427)
and [wheel run 38113973428](https://github.com/grugnog/oeis-learn/actions/runs/38113973428).
The complete CPU suite passed 788 tests with 56 explicit capability skips; the
actual CPU quickstart and immutable archive replay passed. Native G7 and G9
reports passed, and G8 software passed with hardware correctly pending. Every
native JUnit suite had zero failures, errors or skips. Downloaded native artifact
`11693201321` matches ZIP digest
`sha256:e3e638c04bed9df2c8c369680af232f8a35d93e55173bfb1b86063f95fe63fb3`;
its G7/G8/G9 report digests and implementation bindings were independently checked.
The clean installed-wheel/native acceptance and extracted-source rebuild step
also passed. These executed checks complete T112's outstanding software and
packaging acceptance.

Completing the task marker changes the packaged inventory identity. Current
dispositions are bound to that final inventory; the PR's final workflows rebuild
and requalify those bytes. The preceding execution records keep their original
identity. Docker-role and handoff checks are additionally exercised in the wheel
workflow. Passing software checks does not complete real-source, trained-parent,
HIP, throughput or live-experiment acceptance.
