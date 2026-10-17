# Bootstrap training and encoder comparison

The new `fresh-workflow/v2` settings make the encoder explicit. The default in
`configs/foundation/bootstrap.yaml` is the historical tri-stream control:
256-wide, four encoder and four decoder layers, four heads, FFN 1024, dropout
0.1, FiLM, no summary tokens, the historical 16 odd primes and valuation cap 32.
The hybrid adds a gated exact-byte feature branch before the same Transformer.
The byte control uses the same decoder and dimensions. Decoder initialization is
seeded independently of encoder parameter count. These are comparison arms;
none has been established as the best encoder.

This restores numeric input features, not the old vocabulary, family labels,
handwritten sequence teachers, reward shaping or historical checkpoints. The
current body codec, strict execution, provenance and held-out scoring remain.
Existing v1 prepared inputs still select their original 384-wide byte model
unless a new run explicitly supplies `--encoder`. That override creates a matched
bootstrap model and explicit non-curriculum learning settings without rewriting
the sealed data or interpreting old weights as the new architecture.
Do not edit sealed inputs or resume a run after changing its code/image.

## Start with the smallest useful test

Run from the checkout on a Docker or Podman host with a supported ROCm GPU.
Substitute `docker` for `podman` consistently if needed. The host needs only
Python's standard library and engine access; there is no sudo or lab service.

```bash
python3 scripts/foundation_run.py --engine podman image ensure
python3 scripts/foundation_run.py --engine podman validate \
  --suite gpu --output tmp-bootstrap-gpu-validation

# Reuse a complete, verified bulk snapshot when available. This does not fetch
# individual sequences. An existing cache retains its frozen acquisition plan.
if [ ! -f tmp-oeis-bulk/fetch-result.json ]; then
  python3 scripts/foundation_run.py --engine podman data fetch \
    --config configs/foundation/bootstrap.yaml
fi
python3 scripts/foundation_run.py --engine podman prepare \
  --config configs/foundation/bootstrap.yaml --output tmp-bootstrap-inputs

python3 scripts/foundation_run.py --engine podman learnability \
  --inputs tmp-bootstrap-inputs --encoder tri_stream --run runs/bootstrap-tiny-tri-001
python3 scripts/foundation_run.py --engine podman learnability \
  --inputs tmp-bootstrap-inputs --encoder hybrid --run runs/bootstrap-tiny-hybrid-001
python3 scripts/foundation_run.py --engine podman learnability \
  --inputs tmp-bootstrap-inputs --encoder integer_bytes --run runs/bootstrap-tiny-bytes-001
```

Each tiny run selects the same 32 shortest admitted training examples from the
same sealed inputs, allows at most 3,000 updates, checks every 100 updates and
has a one-hour lifetime cap. It uses the production model, optimizer, grammar,
EOS and cached decoding. It does not evaluate or consume the OEIS final cohort.
`learnability.json` passes only with at least 31/32 exact 100-term executions,
a decrease of at least four successes when conditioning is shuffled, and cached
logit parity. Completion of the command is not a passing learnability result.
A failed gate is evidence to inspect, not a reason to launch a longer run.
The tiny test measures memorization and conditioning, not generalization.

For the matched decoder-position/dropout experiment, use the versioned controls
and seeded ROCm commands in [decoder input comparison](decoder-input-comparison.md).

The portable GPU suite includes full-sized update, checkpoint restoration and
cached-logit tests for all three encoder arms. CPU tests alone do not qualify HIP.

## Inspect an already completed checkpoint

This separate, one-hour-capped diagnostic reads the selected checkpoint and its exact
original learner examples from retained `run-state`. It verifies the original
contract/profile/runtime and loads weights only under the new diagnostic code.
It records both identities, leaves the original optimizer/run/budget untouched,
and never loads OEIS truth into the learner.

```bash
python3 scripts/foundation_run.py --engine podman diagnose \
  --run runs/fresh-portable-12h-001 --output tmp-analysis/original-model-diagnostic-001
```

Share `tmp-analysis/original-model-diagnostic-001.zip`. It contains sampled
training and **all** synthetic-development execution (with explicit timeout and
coverage counts), results by length and complexity, shuffled conditioning, partial
and complete token traces, raw/grammar EOS probabilities, output diversity,
prefix/continuation fit, actual cached/uncached representative generation, full synthetic
validation loss and logs. This is explicit diagnosis under changed code, not a
claim that the original runtime was reproduced bit for bit. Choose a new output
path for another diagnostic. This command requires a retained qualified learning
checkpoint; it does not import pre-007 checkpoints.

## Attribute encoder and data changes separately

First screen the matched architectures against the existing sealed data. Use
new run names; no input editing or historical weight initialization is involved:

```bash
for encoder in tri_stream hybrid integer_bytes; do
  python3 scripts/foundation_run.py --engine podman train \
    --profile fresh-pilot-1h --inputs tmp-fresh-inputs-extended-002 \
    --encoder "$encoder" --run "runs/original-data-$encoder-001"
done
```

These pilots use the same explicit learning defaults and disable curriculum for
unlabelled old data. They cap training segments at 1,800 seconds within each
one-hour lifetime. Compare parameter counts, throughput and synthesis diagnostics,
then hold the encoder fixed for the original-versus-connected data comparison.
See [the recommendation audit](bootstrap-recommendation-audit.md) for the complete
coverage table and decision gates. The tiny connected-input tests below are the
scale-up gate, not a claim that the original long-target dataset is easy to memorize.

## Pilot and longer runs

After the tiny gate passes, run a one-hour capped pilot on the full prepared set:

```bash
python3 scripts/foundation_run.py --engine podman train \
  --profile fresh-pilot-1h --inputs tmp-bootstrap-inputs \
  --encoder tri_stream --run runs/bootstrap-pilot-tri-001
```

New v2 training runs score development only. They leave final evaluation pending
for an explicit command after you choose one run.

For encoder comparisons, use identical inputs, seeds, update budgets and hardware
limits. The `--encoder` override is explicitly frozen into each new run; it does
not alter input bytes. Use synthetic validation and OEIS **development** for
comparison. Final scores, when explicitly requested, are finite diagnostic reports, not a
release gate or an encoder-selection signal. Do not claim repeated final evaluations are fresh
holdouts. Before reporting a new final result, compare the old and new prepared
`truth.json` memberships and ensure they are disjoint (a changed seed alone does
not prove this). For example, using the earlier input directory:

```bash
python3 - <<'PY'
import json
from pathlib import Path
old = json.loads(Path('tmp-fresh-inputs-extended-002/truth.json').read_text())
new = json.loads(Path('tmp-bootstrap-inputs/truth.json').read_text())
seen = {m for split in ('development', 'final') for g in old[split] for m in g['members']}
final = {m for g in new['final'] for m in g['members']}
assert seen.isdisjoint(final), 'Choose a new cohort seed and prepare new inputs before final evaluation'
print('New final membership is disjoint from previously examined groups')
PY
```

The bootstrap config exposes learning rate (default 1e-4), validation cadence
(250 updates), update ceiling (5,000), patience (eight checks), minimum loss
improvement, diagnostic sample/time/token caps, curriculum promotion accuracy,
prefix verification and opt-in replay. A run stops on its update ceiling,
plateau, or reserved evaluation budget, so a 12-hour cap is not a promise to
occupy the GPU for 12 hours. Full synthetic loss covers every exported development
example; execution diagnostics have explicit smaller fixed denominators.

The connected sampler draws short expression trees, deeper compositions,
bounded one- or two-register state updates, then conditionals. Index-valued
scalar operands permit nonlinear polynomials and iteration-dependent updates;
coupled registers permit second-order recurrences without named family templates. It draws 16 parameter variants per
structural seed; the existing constant-insensitive structural split, independent
execution, overflow rejection and conditioning deduplication still apply.
The exported training set has approximately equal stage quotas, at most 5%
constant prefixes and at most one identity prefix; the preparation report records
these visible-prefix counts. All admitted archive evidence remains intact.
New non-tiny bootstrap runs reserve up to 128 examples (at most one eighth of the
original training view) to test new parameters of familiar structures. Each held
structure retains at least two training behaviors. Reports separate this parameter
split from structural development; zero eligible structures means unavailable,
not successful generalization.
Curriculum starts at stage zero, mixes already unlocked stages, and promotes only
when both training and structural-development sampled execution reach 80%.
An absent development stratum does not count as success. The stage, permutation,
cursor, RNG and admitted replay targets are durable checkpoint state.

Checkpoint selection prioritizes unlocked competence stage, synthetic execution
accuracy, full validation loss, then the earliest update. For new v2 runs, OEIS development
scoring occurs on the selected checkpoint rather than after every short training
segment. Final evaluation remains pending. Optional replay (`run.learning.replay: true` in a **new** config) admits
at most eight proposals per check and retains at most 32 shorter targets. Only
training prompts are used; every candidate passes structural/parameter/prefix
holdouts and independent 100-term native/reference execution before use.

Measure the chosen model's batch size before scaling:

```bash
python3 scripts/foundation_run.py --engine podman benchmark \
  --inputs tmp-bootstrap-inputs --encoder tri_stream --run runs/bootstrap-benchmark-001
python3 scripts/foundation_run.py extended-settings \
  --benchmark runs/bootstrap-benchmark-001 --output tmp-analysis/bootstrap-12h.json
python3 scripts/foundation_run.py --engine podman prepare \
  --config tmp-analysis/bootstrap-12h.json --output tmp-bootstrap-inputs-12h
python3 scripts/foundation_run.py --engine podman train \
  --profile fresh-extended-12h --inputs tmp-bootstrap-inputs-12h \
  --run runs/bootstrap-12h-001
```

The extended config retains the measured encoder and batch settings, requests
2,048 training prompts and 128 development structures, and raises the update
ceiling to 100,000 with checks every 1,000 updates. Confirm cohort usage before
interpreting its final report. Repeat the benchmark when model or hardware
changes; measured throughput and memory headroom determine batch size.
`--memory-gib` and `--cpus` are global options before the command. Keep resource
options consistent on resume. The wrapper does not stop unrelated GPU workloads.

## Evaluate, recover and share evidence

```bash
python3 scripts/foundation_run.py status --run runs/bootstrap-pilot-tri-001
python3 scripts/foundation_run.py --engine podman resume --run runs/bootstrap-pilot-tri-001
python3 scripts/foundation_run.py --engine podman evaluate \
  --run runs/bootstrap-pilot-tri-001 --split development
# Only after choosing one run and checking that its final cohort is unconsumed:
python3 scripts/foundation_run.py --engine podman evaluate \
  --run runs/bootstrap-pilot-tri-001 --split final
python3 scripts/foundation_run.py pack --run runs/bootstrap-pilot-tri-001 \
  --output tmp-analysis/bootstrap-pilot-tri-001.zip
```

Run commands also package automatically. Bundles include learning settings,
model/data/image identities, validation denominators, exact/shuffled execution,
partial tokens, cache checks, curriculum stages, replay summaries, stop reasons,
throughput/resource metrics and role logs. Sampled or omitted artifacts are
listed explicitly. Weights are excluded by default; add `--checkpoint` to
`pack` only when weight inspection is needed. Downloads, admitted data, runtime
state and all reports remain under the workspace.

Prefix verification executes only the visible 20 terms before selecting a
candidate. Attempts share a time/token/expansion budget. The last bounded slots
may repair literals using visible values or neighbouring constants/operators;
at eight or more attempts one slot also tries width-two beam search. Every
repair is re-executed. This is bounded literal grounding, not a general symbolic
solver, and a prefix fit does not establish the hidden continuation.
