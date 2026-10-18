# Retained compatibility baseline (T009)

On the Phase 1 worktree based on `b96b92ff680a1509d3cede89e9abf0ddb65d2665`, **165 retained CPU tests passed**. A separate preparation/CI-image suite passed **47 tests**, with the two actual installed-image/native smoke tests explicitly skipped because this agent environment has no container engine/native binding. The new frozen signed-256 cases also pass through the retained Python Wasmtime runtime. Results and SHA-256 identities are in [compatibility-baseline.json](compatibility-baseline.json). Workspace JUnit logs are `tmp-validation/phase1/{compatibility,contracts}.xml`; they are not committed bulk artifacts.

The retained tests cover signed limbs, existing arithmetic/constant fitting counterexamples, logical codec round trips/rejections, original hybrid and tri-stream finite gradients/padding, existing decoder input dispatch, runtime thread settings and cached/full-prefix parity. CPU test interpreter is Python 3.12.14, torch 2.9.1+cpu and Python Wasmtime 20.0.0. This is available CPU regression evidence, not an assertion of pinned-container Python 3.12.12, native Wasmtime 20.0.2 or actual HIP execution. The final container builds separately run their committed locks and the retained native tests.

Commands (local disposable Python environment):

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=2 .venv-phase1/bin/python -m pytest -q \
  tests/unit/test_foundation_codec.py tests/unit/test_multi_limb_arithmetic.py \
  tests/unit/test_wide_result_profile.py tests/unit/test_foundation_grounding.py \
  tests/unit/test_smt_constant_solver.py tests/unit/test_dixon_diophantine_solver.py \
  tests/unit/test_foundation_cache.py tests/unit/test_bootstrap_models.py \
  tests/unit/test_bootstrap_decoder_inputs.py tests/unit/test_tri_stream_encoder.py \
  tests/unit/test_foundation_runtime_threads.py \
  --junitxml=tmp-validation/phase1/compatibility.xml
PYTHONPATH=src:. .venv-phase1/bin/python -m pytest -q \
  tests/contract/test_bigint_build.py tests/contract/test_bigint_contracts.py \
  tests/unit/test_foundation_ci_images.py \
  --junitxml=tmp-validation/phase1/contracts.xml
```

The JSON explicitly hashes numerical profiles, logical codec, inference cache, wide helpers, historical hybrid encoder and model construction; all those bytes match the baseline commit. Frozen old names remain `wat_i256_checked_v1`, `wat_body_decimal_v1`, `ryzen_foundation_v1`, `prefix20_total100_v1` and the historical `logical-wat-decimal/v1` codec implementation. No historical weights were supplied: these are historical model **regressions**, not an evaluation of a saved checkpoint. Checkpoint rescue and actual HIP checks remain their later-phase tasks.

New source and compiled binary identities necessarily change when adding the core dependency and replacing the five-file native inventory with the supported source closure. Do not pretend those build identities are stable merely because the old numerical/codec semantics are unchanged. The packaged report records actual native/runtime/lock identities; a changed image cannot silently resume an old run.

Pure-Python wheel and source-distribution builds succeeded. Archive inspection confirms the wheel contains the new schema resource and the source distribution contains all three committed Cargo locks and the frozen fixture manifest. This does not qualify a native binary. The current (rolling) 007 disposition file is rebound to the new package bytes with `full_007_complete: false`; the historical template and qualification records remain unchanged. Its owning test plus task-inventory tests pass (13 tests).
