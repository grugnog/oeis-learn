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
