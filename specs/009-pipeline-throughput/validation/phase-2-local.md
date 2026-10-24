# Phase 2 inference implementation

Status: implementation/local fixture evidence; target qualification pending.

Hoisted cache metadata is an explicit hoisted/v1 option; baseline remains default. Greedy unused log-probability transfer was removed. CPU/HIP fixtures cover batch 1/8, eager/SDPA, positions 1/32/128/513/1024, mixed/finished/reordered rows, capacity and stale-weight invalidation. These fixtures compile and are registered in the pinned image.

Torch and engines are unavailable locally. CPU/HIP cache/logit/token fixtures have NOT run here. No numerical equivalence, isolated speedup or production adoption is claimed.

Evidence: [actual CLI record](local-evidence.json), [retained host receipt](local-host-receipt.json), [80-test output](local-host-tests.txt), [task evidence](task-evidence.json). The common host suite is one 80-test run, not 80 tests per phase. Python compilation and git diff whitespace checks passed. No CI result is claimed.

## Current-source operator fixture follow-up

Current-source target evidence supersedes the original unrun-fixture status: all 26 CPU decoder cases and 15 actual-HIP suite cases pass. The timed inference comparison is unstarted on incomplete baseline calibration; no speedup/adoption follows. T035/T036/T040 fixture/run tasks close; T041 comparison collection remains pending. See the [fourth operator review](operator-20261024T141450Z-review.md) for exact source/archive/JUnit evidence.
