# Workstation reliability and scheduling review

Reviewed 2026-11-08 from the completed `pac-reliability` diagnostic. Archive
SHA-256: `a1bc443497e1de48c13ea48dae045c1d851d36e4c1c9644fa25d3603b02e145f`.
The 198 inventory entries, their bytes/hashes and seal, producer bindings,
benchmark clocks/metrics and telemetry were checked. The source identity matches
the implementation delivered by #63. Only this summary is published.

## Qualification and training-only gate

Current-source qualification passes: 1,578 CPU/native tests and 27 actual-HIP
tests, with zero failures, errors or skips in either recorded suite. This is new
qualification evidence, rather than reuse of the previous implementation's PASS.

The complete 256-task training-only probe panel passes its predeclared technical
gate, with all seven controls and both selectors. Primary equal-budget results:

| Arm | Completed / 256 | Original successes | Coverage-selector successes |
| --- | ---: | ---: | ---: |
| B | 255 | 182 | 182 |
| D | 244 | 83 | 84 |

Every task's completion, success and chosen candidate identity matches the prior
successful probe run, across both arms, all controls and both selectors. There
are no grammar dead ends in the new generation panel. D's 12 incomplete primary
cases now all report `token_cap_without_eos`; B has one fitting timeout. Removing
a grammar failure class therefore did not improve these probe scores. The
1,024-token, four-second service and 60-second portfolio limits remain unchanged.

This is a technical screen, not model selection or fresh scientific evidence.
No training or assessment was executed. The wrapper's `DIAGNOSTIC_ONLY` /
`freeze_eligible: false` is intentional; the enclosed probe gate passes. The
previous fresh structural result remains **INCOMPLETE**.

## Scheduling counterfactual

Each arm ran drain/overlap/overlap/drain on the same 16 training probes, four
controls and eight attempts: 512 fitting outcomes per run. Wall times below are
means of measured portfolio spans, excluding model loading and role overhead.
These are not whole-campaign speedups or scored accuracy comparisons.

| Arm | Drain seconds | Overlap seconds | Less wall time | Resource-exhausted fits: drain -> overlap |
| --- | ---: | ---: | ---: | ---: |
| B | 92.41 | 64.29 | 30.4% | 2 -> 29 in each repeat |
| D | 148.44 | 125.34 | 15.6% | 3 -> 94 / 95 |

Generated tokens, proposal-source hashes, seeds and generation completion are
identical across all four runs within each arm. Both drain repeats have identical
attempt-behavior identities. Overlap repeats differ downstream: B changes 27
fitting outcomes per repeat; D changes 91/92. Primary selected programs change
on 1-3 of 16 B tasks and four of 16 D tasks relative to the first drain run.
The benchmark does not score full-horizon correctness, so these changes cannot
be called accuracy improvements or regressions.

In each D overlap run, 28 fits finished within their service deadline but were
observed by the owner only after the deadline. Observation lag after worker
completion is 7.05-9.70 seconds, with a median about 9.6 seconds. The scheduler
calls GPU sampling synchronously and drains native completions only after that
call returns. Another 55 fits per D overlap run expire without dispatch. Long
decode calls block timely observation and dispatch. Deadline enforcement
correctly excludes late observations. Raising allowances or accepting them retroactively would change
the experiment.

Long generations remain costly: in the fixed D drain subset, six of 96 decode
batches consume about 46.5% of decode time. Both overlap repeats retain six
capped batches. Larger permitted batches plus overlap did not remove this tail.

## Resource interpretation and disposition

The role envelope was four CPUs and 16 GiB RAM. Peak cgroup memory was about
1.58 GiB; GPU peak allocated memory was about 0.58 GiB with drain and 1.05 GiB
with overlap. Average CPU consumption was about 1.2-1.8 cores. CPU throttling was
small, collector overhead stayed below 0.3%, and no samples or events were lost.
These measurements do not indicate a role-memory or sustained CPU-quota limit.
APU memory counters must not be added together as independent physical usage.

GPU busy, memory-busy and power sensors were unavailable because selected
render-node sysfs mapping was unavailable. Allocator peaks cannot establish GPU
utilization, hardware saturation or a memory-bandwidth bottleneck.

Retain the qualified drain schedule. Any further overlap optimization should
service native dispatch/completion at bounded decode intervals and control queue
admission while retaining strict existing deadlines. Benchmark batching and
overlap separately, requalify changed source, and compare outputs and completion
as well as wall time before adopting a replacement.

The next scientific step is a read-only fresh-cohort availability check excluding
**every** already observed structural cohort, including fresh-002. Existing
freeze commands do not establish that exclusion. Only after feasibility is
known should a separate B/D/template comparison be frozen with three paired
training seeds, checkpoints, selectors and budgets. Do not relax sparse-family
quotas or reseed after seeing scores. Keep assessment untouched, preserve the
historical INCOMPLETE result, and defer broad HPO. A learned/template portfolio
still requires its own visible-only selection rule and prospective total budget.
