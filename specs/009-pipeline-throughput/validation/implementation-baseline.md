# Implementation baseline

Branch: `009-pipeline-throughput`; starting commit `fa49a5ebc0a018544f1d0c6ccfc687fc74b1dbee`.
PR #59 is open; its head `50fc1629dddb0005d90f2d27686452d804e399bc` is an ancestor of this branch. This implementation remains stacked on that dependency. Existing PR #60 is reused.

Public entry point: `scripts/foundation_run.py`, routed through `foundation_local.py`. Existing qualification owner: `container_validation.py`; bounded operations: `qualify_bigint_phase5.py`; diagnostic preparation, isolated GPU/scorer roles and profiling: `profile_bigint_pipeline.py`. Keep those owners and legacy commands.

The requirements checklist passes 16/16. No extensions.yml hooks are registered. Feature resolution uses `SPECIFY_FEATURE_DIRECTORY=specs/009-pipeline-throughput`. Host execution must remain standard-library-only. All real-engine, HIP and study outcomes are pending; 008:T093 remains deferred and 008:T096 retains its negative 7,730/14,400-second ledger.
