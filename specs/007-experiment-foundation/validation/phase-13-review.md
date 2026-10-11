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
