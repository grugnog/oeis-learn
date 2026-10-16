# Bootstrap learnability implementation

The fresh workflow selected an experimental byte encoder and larger model without
an encoder comparison. Restore an explicit historical tri-stream control, add an
exact-byte supplementary branch, and retain the old byte profile for existing
inputs. No encoder is labelled a measured winner.

Implementation sequence:

1. Explicit hashed tri-stream, hybrid and byte-control profiles with identical
   decoder dimensions and frozen historical feature settings.
2. Bounded production-path learnability diagnostics: train/development execution,
   shuffled conditioning, cached/uncached logits, completion and retained tokens.
3. A separate versioned generic generator with connected dataflow, parameter
   variants and increasing structural complexity. Never alter generic_sampler_v1.
4. Frequent complete synthetic validation and competency-driven curriculum,
   explicit optimization/stop settings, durable state and checkpoint selection.
5. Prefix-verified bounded search and opt-in verified training-only replay; no
   development/final targets enter optimization. Preserve incomplete evidence.
6. Portable CLI commands, bounded analysis bundles and CPU/HIP regressions.

Acceptance: tiny generic tasks must be learned through the same model and decoder
used in production. Compare encoders on identical inputs, seeds and budgets before
claiming superiority. Existing final cohorts remain consumed. New experiments use
new run/input identities; historical runs and ledgers are never rewritten.

Hardware acceptance must run on an actual ROCm host. CPU and mocked-container
tests cannot establish HIP readiness or sustained GPU utilization.

Implemented validation (CPU-only development environment):

- The final unit suite plus fresh lifecycle and bootstrap admission/GPU test
  collection produced 960 passes, five skips, and four failures. All four
  failures reproduce on unchanged base `37981d8`: the legacy elite-buffer empty
  conditioning canary, progressive-readiness unsupported CGI configuration, and
  two allocator fixtures that submit zero elapsed cost. No guards were relaxed.
- Reduced-size production training passes the eight-example generic constant
  task for tri-stream, hybrid and byte encoders: 8/8 exact 100-term executions,
  0/8 with shuffled conditioning, complete EOS and cached-logit parity.
- Tests cover independently verified archive/replay admission, exact resumed
  weights and curriculum state, full validation batching, explicit final
  evaluation, and separate checkpoint diagnosis without changing the old run.
- Changed Python files pass the repository's `ruff --select F,E9` checks.
  The implementation digest was refreshed without changing release dispositions.
- Docker/Podman, the Rust extension build and actual HIP hardware were unavailable
  here. The new full-profile HIP tests and portable validation command must run
  on the target host before treating the new workflow as GPU-validated.

Copy-paste startup, comparison, diagnosis, scaling and packaging commands are in
[bootstrap-training.md](bootstrap-training.md). Start with learnability gates;
CPU memorization evidence does not establish OEIS generalization or an encoder
winner.
