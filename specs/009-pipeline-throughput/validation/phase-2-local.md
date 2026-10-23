# Phase 2 inference implementation

Status: implementation/local fixture evidence; target qualification pending.

Hoisted cache metadata is an explicit hoisted/v1 option; baseline remains default. Greedy unused log-probability transfer was removed. CPU/HIP fixtures cover batch 1/8, eager/SDPA, positions 1/32/128/513/1024, mixed/finished/reordered rows, capacity and stale-weight invalidation. These fixtures compile and are registered in the pinned image.

Torch and engines are unavailable locally. CPU/HIP cache/logit/token fixtures have NOT run here. No numerical equivalence, isolated speedup or production adoption is claimed.

Evidence: [actual CLI record](local-evidence.json), [retained host receipt](local-host-receipt.json), [80-test output](local-host-tests.txt), [task evidence](task-evidence.json). The common host suite is one 80-test run, not80 tests per phase. Python compilation and git diff whitespace checks passed. No CI result is claimed.
