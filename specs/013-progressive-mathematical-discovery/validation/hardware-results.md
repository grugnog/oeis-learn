# Workstation hardware results

## Rehearsal 006: tiny qualification passed, device acceptance setup failed

Original support ZIP SHA-256:
`30e842c4ab77fa218f661794aa8fd1b847ab514f2d971c9d62f0987ccfcf6277`.
All 262 inventoried members matched their recorded sizes and hashes; there were
no collection omissions. Session: `run-state/discovery/013-hip-006`. Source:
`sha256:419039e9d99d9468b001e5bbfcdb94869471a52916b16710abb7157cf044bfb9`.

| Stage | Original result |
| --- | --- |
| CPU/native preflight | 2,319 passed; ten deselected; 82 subtests passed |
| HIP preflight | 27 passed |
| Tiny selected-model qualification | Four of four cases PASS; no omissions |
| Preparation and freeze | COMPLETE, diagnostic purpose only |
| Installed CPU/native feature acceptance | All 26 registered cases PASS |
| Seven-scenario HIP feature acceptance | All seven failed during fixture setup |
| Research/sustained execution | Not performed |

The four original case producers cover input lengths 4/20 and microbatches 7/16.
Their seals and exact source/image/engine/hardware/case bindings were checked.
All seven requested measurements passed, including full restore, deterministic
continuation, generation, native scoring and zero allocated bytes after cleanup.
Their producer identities (4/7, 20/7, 20/16, 4/16) are:

- `sha256:4fdc661fdbc5b1ea1cc0c5b80c8c8efcdea6959ca6a6003272e5ef74aa55860e`
- `sha256:765faa055a31c4854583961ea2eb37a1d8336c4162a107969a6a76b4aa312303`
- `sha256:b5c65b4b4e1d250a1cdbdd06eda6e2d32078f091e0acd38df7e51e45fc882826`
- `sha256:d3827499dd6c13891dbd62abe51e593a0a9c382dd675bdd0a93483b3b4146465`

Original qualification identity:
`sha256:e29284d474e610d6514a07c38a10a0460162a586f432baae7d4980a9183a564d`.
The reconciled device delta is PASS with four executed cases and no omissions:
`sha256:a9a03d60c3a397e5e909cb451850510cfe09f3a19297777d29e263f496b8fa98`.
The frozen tiny campaign is
`sha256:4fa8395f50b853f8216e6ac59abe089cb05eabd44d2d696b137b849e3c782251`.
The installed software acceptance receipt is COMPLETE, DIAGNOSTIC_ONLY,
qualified=false:
`sha256:ae29e0d3cea9c02b74ff37b9383fc751c1f73c3a0988075f2b3e2e42ba702109`.

Every device acceptance case failed before its scenario ran. Pytest creates each
requested basetemp but does not create its parent; the batch launcher had not
created `/work/output/integrated/private`. The original per-case XML and logs all
show that same FileNotFoundError. All independent attempts and diagnostics were
retained. This is not a failed learning/restore assertion, and it provides no
seven-scenario device acceptance evidence.

The launcher repair explicitly creates the private parent with mode 0700 and
keeps separate per-case scratch directories. It also disables pytest's cache
plugin, since the installed checkout is read-only. A real two-process pytest
regression exercises both tmp_path and tmp_path_factory, verifies private
permissions, confirms a later case does not erase the first case's files, and
checks that no checkout cache is created. It failed before the repair.

The original tiny qualification is now available, but it cannot qualify a changed
source or the retained research model by extrapolation. T082/T088–T091 retain
pending scope requirements: repaired-source qualification, actual campaign device
acceptance, research input preparation and sustained evidence as applicable.
The [retained research inputs](research-inputs.md) remain identified but their
registration/preparation is separate. No historical receipt is rewritten.

Repair validation passed all **103** focused workstation, acceptance/reuse,
handoff, packaging/inventory, CI and implementation-digest checks. The expensive
installed all-stage test was deselected from that focused command. Separately,
all **seven** registered acceptance scenarios passed through the repaired batch
launcher with real CPU/native operations, separate pytest processes and installed
role/checkpoint paths. That batch covers lifecycle recovery, OEIS, components,
short sources, refreshed learning, refutation/quarantine and selected-trial
restore/continuation. It is CPU software evidence only: no device flag or HIP
request was supplied, and no hardware authority was produced. Local originals
are retained under `run-state/rehearsal-007-repair/cpu-batch`.

The current package implementation digest was refreshed without changing
historical dispositions. `git diff --check` passed. No container build or HIP
execution occurred locally, and CI was not awaited. Rehearsal 007 must qualify
the repaired source and perform the actual seven-scenario device acceptance.
