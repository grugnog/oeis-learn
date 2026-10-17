# Hybrid repeated-digit checkpoint investigation

The selected update-900 checkpoint from `bootstrap-step1-hybrid-control-001`
generates correct programs for 30 of 32 memorization examples. The failures are
`n - 11` and `n + 11`, decoded as `n - 1` and `n + 1`. These are small-integer
decoding failures, unrelated to the checked-integer range limit.

The source archive, selected checkpoint, saved learning-state hashes, and relevant
model/training source identities were verified before these CPU probes. The source
run used HIP; the new experiments below used PyTorch 2.9.1+cpu, not a GPU. Original
weights and run records were not modified.

## Evidence and controlled continuation

The selected checkpoint reproduces both errors on CPU. The expected second digit
is grammar-legal. Cached/full decoding agrees. After the first and second `1`,
decoder hidden-state cosine similarity exceeds 0.99994, despite requiring opposite
decisions: continue the literal, then stop. The scaled token-embedding norm is
about 258, versus 11.3 for the positional vector and 2.67 between adjacent positions.
This suggests weak positional discrimination; scale alone does not prove causality.

Four disposable continuations reload the same selected weights and AdamW state,
use the same new sample order, and train for 200 updates on all 32 examples.
Learning rate is 1e-4, batch size 8, with the production mean-per-program SFT
objective and gradient clipping at 1.0. Dropout changes apply to encoder and
decoder dropout modules and attention dropout together.

| Dropout | Decoder sinusoidal gain | Final greedy execution matches |
| --- | --- | --- |
| 0.1 | 1 (control) | 30/32 |
| 0 | 1 | 30/32 |
| 0.1 | 16 | 31/32 |
| 0 | 16 | 32/32 |

Greedy programs were compared with their targets using the Python reference
interpreter at every index 0 through 99. The combined intervention also scored
32/32 on teacher-forced checks at updates 100, 125, 150, 175, and 200. Raising
position gain only at inference did not fix the original checkpoint.

This is one seed, one previously trained checkpoint, and a tiny offset-only set.
It is not from-scratch validation, native/WASM admission, GPU qualification,
held-out transfer, or evidence that a 16x gain is optimal. Production defaults
remain unchanged. Direct buffer scaling is an experimental intervention, not a
supported checkpoint/configuration migration.

## Encoder feature probe

Zeroing individual branch outputs in the selected model gives the following
teacher-forced all-token argmax counts (not free-generation execution scores):

| Zeroed stream | Correct examples |
| --- | --- |
| None | 30/32 |
| Magnitude | 23/32 |
| Modular | 3/32 |
| Differences/p-adic | 28/32 |
| Exact bytes | 30/32 |

These interventions change the trained activation distribution and include
feature interactions. They support retaining mathematical features but cannot
rank optimal encoder inputs for full OEIS. In particular, small offsets provide
little evidence about exact bytes or large/complex difference patterns.

## Next implementation and GPU gates

1. Add explicit, versioned decoder positional/token scaling and dropout controls,
   preserving the existing default and checkpoint identities. Test checkpoint
   round trips, positional-buffer extension, and cached/uncached equivalence.
2. Compare the control and candidate from scratch on the same hybrid dataset and
   seeds. Require stable synthesis accuracy, not a single threshold crossing.
3. Add a focused literal diagnostic with repeated digits, zeros, sign changes,
   and multiple lengths (for example 11, 111, 22, 101, and 1000). Require correct
   literal termination as well as execution; retain incomplete output tokens.
4. Confirm on actual ROCm, then expand the generic curriculum. Only after those
   gates should longer training and encoder input comparisons guide full-scale
   choices. Arbitrary-precision execution remains separate from this decoder fix.

## Reproduce and inspect

The JSON records are in `docs/evidence/hybrid-digit-probe/`. Start with
`results_summary.json`; `provenance.json` records exact archive/checkpoint hashes
and the tested source revision. `source_history.json` contains the original GPU
run, with evaluation coverage retained. Zero evaluated examples do not constitute
a completed evaluation. The other files contain the individual CPU probes.

The scripts in `scripts/diagnostics/hybrid_digits/` reproduce this specific
checkpoint investigation. They are research probes, not additions to the host
wrapper or qualified training commands. They require the project's existing
Python dependencies inside a prepared environment/container; the portable host
wrapper still requires only the standard library. No weights are committed.

From the repository root, inside that environment, place the supplied archive
under `upload/` and run:

```sh
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export OEIS_PROBE_ARCHIVE="$PWD/upload/bootstrap-step1-hybrid-control-001-weights.zip"
export OEIS_PROBE_OUTPUT="$PWD/tmp-hybrid-probe"

python scripts/diagnostics/hybrid_digits/inspect_weights.py
python scripts/diagnostics/hybrid_digits/feature_probe.py
python scripts/diagnostics/hybrid_digits/position_conflict.py
python scripts/diagnostics/hybrid_digits/finetune_probe.py
python scripts/diagnostics/hybrid_digits/validate_generation.py
python scripts/diagnostics/hybrid_digits/extract_history.py
```

Use a fresh output directory. The loader writes a verified selected-weight copy
there, and continuation writes four disposable experimental checkpoints. Allow
approximately 250 MB of output and several minutes on CPU. The reference
environment was Python 3.12.14 with PyTorch 2.9.1+cpu. Small differences near tied
logits may arise across kernels/thread settings. The probes verify the supplied
archive's inventory and saved learning-state hashes before model loading.

The CPU experiments intentionally use a new seeded sample order shared across
arms, not the source run's saved RNG continuation. Changing dropout changes both
encoder and decoder dropout; these results do not isolate decoder dropout alone.
The positional-buffer intervention stays within the saved positional extent and
must not be used as a general deployment patch. Buffer extension, model identity,
and supported scaling controls were requirements for the implementation below.

The supported controls and matched from-scratch ROCm commands are now documented
in [decoder input comparison](decoder-input-comparison.md). The research scripts
above preserve the original intervention for reproduction; use the portable
learnability command for new experiments.
