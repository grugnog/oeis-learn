# US1 acceptance matrix

| Acceptance boundary | Current evidence | Disposition |
| --- | --- | --- |
| One public command, stdlib host, static phase graph | Actual python3 -S all-ready invocation;80 passing host tests | Local PASS |
| Failed prerequisite blocks descendants; safe independent stage continues | Host DAG regression and missing-engine real invocation | Local PASS |
| One bounded verified archive and failure recovery | Actual 54-member archive; integrity/failure/export fixtures | Local PASS |
| Immutable reuse, image inspection, no inherited whole-run PASS | Bound CPU reuse fixture; current image adapter always executes | Local PASS; real image reuse pending |
| Engineering cannot train, resume T096 or expose finals | Early authority/manifest tests; phase 6 excluded from all-ready | Local PASS |
| Heartbeat, startup interruption, shared cleanup allowance | Subprocess deadline/log-tail and fake process/container fixtures | Local PASS; engine fault injection pending |
| Arbitrary UID/rootless permissions, installed package/native inventory | Registered existing pinned-image and operations checks | PENDING; no engine here |
| HIP and Docker+Podman coverage | Registered primary HIP and both-engine stages | PENDING; no target evidence |

This matrix does not close real-operator tasks025/026. No separate manual build/test/profile/archive command is needed for the next workstation collection; the [quickstart](../quickstart.md) selects the combined scope. Missing capabilities remain visible and cannot become PASS. See [local-evidence.json](local-evidence.json).
