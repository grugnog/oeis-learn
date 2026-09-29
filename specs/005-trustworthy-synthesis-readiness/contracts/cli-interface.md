# CLI Contract: Trustworthy Synthesis Readiness

**Feature**: [../spec.md](../spec.md)  
**Contract version**: `1.0`

## Common Rules

- Commands write machine-readable JSON before rendering optional Markdown or console summaries.
- Qualified commands require contract-valid checkpoint metadata, a frozen benchmark manifest, and the native evaluator.
- Exact integer terms supplied in JSON use decimal strings.
- Relative paths are resolved from the invocation working directory and recorded as resolved artifact identities.
- Exit code `0` means the requested operation completed with its command-specific successful outcome.
- Exit code `1` means the operation completed but did not meet its target outcome, such as no extrapolating candidate or failed readiness gates.
- Exit code `2` means request, checkpoint, manifest, protocol, or dependency validation failed before a valid evaluation completed.
- Exit code `3` means an execution or artifact-write failure interrupted the operation.

## `convert-checkpoint`

Converts one explicitly described legacy checkpoint into checkpoint format v2.

```text
oeis-learn convert-checkpoint \
  --input-checkpoint PATH \
  --config PATH \
  --output-checkpoint PATH
```

Rules:

- The command never infers missing architecture or vocabulary metadata.
- The output records the source checkpoint digest.
- Conversion validates strict state loading before writing the output.
- Exit `0` only when the v2 checkpoint can be loaded for inference.

## `synthesize`

Runs the shared checkpoint-to-verdict pipeline for one target.

```text
oeis-learn synthesize \
  --checkpoint PATH \
  --benchmark-manifest PATH \
  --oeis-id A000000 \
  --candidate-budget {1,8,16} \
  --seed INTEGER \
  [--temperature NUMBER] \
  [--top-p NUMBER] \
  [--max-tokens INTEGER] \
  [--constant-resolution | --no-constant-resolution] \
  [--solver-timeout-ms INTEGER] \
  [--fuel-per-invocation INTEGER] \
  [--memory-limit-mib INTEGER] \
  [--mdl-max NUMBER] \
  [--device cpu|cuda] \
  --output-json PATH \
  [--output-markdown PATH]
```

Rules:

- The target must exist in the frozen manifest and contain exactly 20 observed plus 100 unseen terms.
- `--fuel-per-invocation` cannot exceed `10000`; `--memory-limit-mib` cannot exceed `16`; `--mdl-max` cannot exceed `1.20` for qualification.
- `--candidate-budget` defaults to `8`.
- `--seed` is mandatory for qualified evaluation.
- `--extrapolate` remains a deprecated alias for `--unseen-horizon`; any value other than `100` is unqualified.
- `--terms` may be retained for diagnostic evaluation only. It requires exactly 120 values and an explicit `--diagnostic` flag; its output is marked unqualified because it lacks frozen source provenance.
- Exit `0` when at least one candidate is `EXTRAPOLATING_SUCCESS`, `1` when evaluation completes without one, and `2` or `3` according to the common rules.
- JSON output conforms to [synthesis-evaluation.schema.json](synthesis-evaluation.schema.json).

## `test-progressive`

Collects preflight evidence and evaluates a versioned readiness policy.

```text
oeis-learn test-progressive \
  --max-tier {0,1,2,3} \
  --policy PATH \
  --output-report PATH \
  [--diagnostic-override \
   --override-operator TEXT \
   --override-reason TEXT \
   --override-intent TEXT]
```

Rules:

- Without an override, any mandatory failed gate exits `1` and blocks production authorization.
- All three override metadata fields are required together.
- An override never changes a failed gate to passed; it changes only the run qualification state to `OVERRIDDEN_UNQUALIFIED`.
- Exit `0` only for `AUTHORIZED`. An overridden diagnostic run exits `1` after writing its report so automation cannot mistake it for qualification.
- JSON output conforms to [readiness-report.schema.json](readiness-report.schema.json).

## `run-ablations`

Executes a predeclared paired experiment.

```text
oeis-learn run-ablations \
  --manifest PATH \
  --output-directory PATH \
  [--resume]
```

Rules:

- The manifest is validated and persisted before the first trial begins.
- `--resume` may continue missing seed/variant pairs but cannot alter the frozen manifest.
- Inference variants share the same ordered candidate cache per target and seed.
- Training variants start from the same checkpoint, optimizer snapshot, replay snapshot, and total active/replay budgets.
- Exit `0` only for a `COMPLETE` experiment. `PARTIAL` and `FAILED` exit `1` after retaining artifacts.
- The manifest and outcomes conform to [experiment-manifest.schema.json](experiment-manifest.schema.json).

## `discover`

Runs the shared checkpoint-to-claim discovery pipeline.

```text
oeis-learn discover \
  --checkpoint PATH \
  --benchmark-manifest PATH \
  --protocol PATH \
  --definitions PATH \
  --seed INTEGER \
  --output-json PATH \
  [--output-markdown PATH]
```

Rules:

- The checkpoint, benchmark, discovery protocol, and definition-registry digests are recorded.
- Latent candidate generation, canonicalization, exact numerical validation, and symbolic verification have separate evidence records.
- Missing symbolic definitions do not fail the command and do not promote a numerical conjecture.
- Exit `0` when the complete pipeline and artifact writes succeed, even when there are zero claims. Exit `2` for invalid inputs and `3` for interrupted execution or artifact failure.
- JSON output conforms to [discovery-report.schema.json](discovery-report.schema.json); definition input conforms to [symbolic-definitions.schema.json](symbolic-definitions.schema.json).

## Production Benchmark Adapter

The production benchmark accepts matching evaluation options:

```text
scripts/run_long_e2e_benchmark.py \
  --eval-checkpoint PATH \
  --benchmark-manifest PATH \
  --readiness-policy PATH \
  --eval-candidate-budget {1,8,16} \
  --eval-seed INTEGER \
  [--diagnostic-override metadata options]
```

After training, the adapter reloads the selected checkpoint from disk and invokes the same synthesis and discovery pipelines as the CLI. It stores immutable reports under unique artifact IDs rather than overwriting a single result file.

## Artifact Naming

```text
runs/<run>/reports/
├── synthesis/<evaluation_id>.json
├── readiness/<report_id>.json
├── experiments/<experiment_id>.json
└── discovery/<report_id>.json
```

Optional Markdown files share the JSON basename. JSON remains authoritative.
