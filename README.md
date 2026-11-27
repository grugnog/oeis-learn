# OEIS Learn

OEIS Learn asks whether a model can infer an executable rule for an integer sequence and use it to predict terms it has never seen. In the main research profile, the model gets **20 indexed terms**. It proposes programs, a native service can fit unknown integer constants using those visible terms, and a selector commits to one program. An independent scorer runs it on the **next 80 terms**. A single wrong value loses the exact solve. A match is evidence about those indices. A theorem about the whole sequence needs a separate argument or checked certificate.

The long-term goal is mathematical discovery through reusable program structures, transfer to new families and, where the evidence permits, checkable explanations. The project uses synthetic problems with known programs to study *how* a learner succeeds, and real sequences from the Online Encyclopedia of Integer Sequences (OEIS) to measure transfer. A useful experiment can establish that the current model should be kept, that a conventional algorithm is stronger, or that a comparison is still unresolved. [Project constitution](.specify/memory/constitution.md) · [Scientific scope](specs/013-progressive-mathematical-discovery/spec.md) · [Certificate and finite-evidence rules](docs/foundation.md#finite-evidence-and-certificates)

| If you want to… | Start here |
| --- | --- |
| Write a research spec in a new conversation | [How to design a study](#how-to-design-a-study), then the [spec and Git history](#how-the-project-got-here). The repository carries its own [Spec Kit skills](.agents/skills/). |
| Run the software | [First commands](#first-commands), [operator guide](USAGE.md) and [fresh training and recovery](docs/fresh-training.md). |
| Understand a model score or performance claim | [Reading experimental results](#reading-experimental-results) and the linked original validation reports. |
| Change the code | [Architecture](#from-an-indexed-sequence-to-a-scored-program), [development and testing](#development-and-testing) and [repository instructions](AGENTS.md). |
| Pick up a historical campaign | Use that feature’s quickstart and pinned operator instructions. The [013 workstation guide](specs/013-progressive-mathematical-discovery/quickstart.md) and [historical evidence index](specs/015-learned-program-search/research.md#finding-the-evidence-without-the-conversation) identify which records belong together. |

## From an indexed sequence to a scored program

1. **Acquire and separate the data.** The corpus reader preserves exact integer text, offsets, source revisions and hashes. The strict training track starts from random weights and mechanically generated, executable programs. An OEIS entry in the local cache is available for inventory; its assigned role determines whether a study may use it. Training, development, confirmation and final assessment have separate exposure records. Family links, duplicated behavior and shifted windows matter even when record IDs differ. [Corpus reader](src/oeis_learn/fresh/corpus.py) · [Corpus config](configs/foundation/oeis_corpus.json) · [Exposure and cohort rules](specs/013-progressive-mathematical-discovery/contracts/experiment.md)
2. **Represent the prompt and generate candidates.** The PyTorch encoder turns visible integers into model features. Profiles include magnitude, differences, modular residues, valuations and an exact-integer chunk representation. The decoder generates tokens under a declared grammar. The selected profile fixes encoding, numerical semantics and model settings. Exact source integers remain available to the verifier even when neural features lose information. [Encoders](src/oeis_learn/encoder/) · [Decoder and codec](src/oeis_learn/decoder/) · [Research model construction](src/oeis_learn/generalization/progressive/models.py)
3. **Ground and select.** A program may contain literal constants or placeholders. The Rust synthesis service uses exact arithmetic and a pinned Z3 solver to fit eligible templates from the visible input. The final grounded source must execute and pass admission. A registered search policy decides how to spend attempts and which admitted candidate to submit. Templates, polynomial fits and linear recurrences are useful conventional controls. The older checked signed-256/macro-WAT profiles and the bigint program language have distinct semantics. [Fitting entry point](src/oeis_learn/synthesis/fitting.py) · [Native crates](crates/) · [Portfolio and selection](src/oeis_learn/fresh/synthesis_policy.py) · [Baseline implementations](src/oeis_learn/generalization/progressive/baselines.py)
4. **Score the sealed choice.** The search role sees the prefix. A separate scorer receives the chosen source and private continuation. An independent Python-integer implementation supplies reference semantics for the bigint profile. Unfinished fitting, wrong output, unsupported execution and a verified solve have different evidentiary meanings. Discovery and certificate tools can subsequently examine a narrower mathematical claim. [Reference evaluator](src/oeis_learn/synthesis/reference.py) · [Evaluation roles](src/oeis_learn/pipeline_validation/progressive_evaluation.py) · [Discovery and certificate checkers](src/oeis_learn/discovery/)

During supervised training the system can run an admitted synthetic program *forwards* to create an indexed sequence, then teach the model the inverse task of predicting program tokens from the prefix. Some recipes train a structure with placeholders and leave the constants to the fitter. Two different programs can produce the same 100 values. Token loss measures agreement with a training target, while held-out execution measures the behavior the project cares about. The historical ingestion and discovery interfaces also cover OEIS metadata and symbolic relations. Imported implementations or named-family teachers belong to a separately declared assisted track. [Training targets](src/oeis_learn/generalization/targets.py) · [Learning contract](specs/007-experiment-foundation/contracts/learning.md)

The three native crates have different jobs. [oeis_wasm_evaluator](crates/oeis_wasm_evaluator/) executes the retained WebAssembly profiles with fuel metering. [oeis_bigint_runtime](crates/oeis_bigint_runtime/) supplies the bigint runtime, and [oeis_synthesis_core](crates/oeis_synthesis_core/) implements the native synthesis service. The independent bigint reference in Python lets tests compare exact outcomes across implementations. A failure in one numerical profile can have a different meaning in another.

The **control plane** lives outside the learner. The standard-library host CLI launches Docker or Podman roles with explicit mounts. The installed role registry dispatches named operations. Campaign code owns the frozen task graph, trial roster, budgets, journals, checkpoints and report verification. Model code can therefore change without having to invent a scheduler or a new definition of completion. Configuration and packaged schemas participate in source identity. Some entry points are reached by filename or registry lookup. [Host entry point](scripts/foundation_run.py) · [Installed role registry](src/oeis_learn/pipeline_validation/progressive_roles.py) · [Campaign and trial code](src/oeis_learn/generalization/progressive/) · [Stage configurations](configs/foundation/progressive/) · [Qualification source map](src/oeis_learn/experiments/source_snapshot.py)

## How the project got here

A **spec number** identifies a feature. A **round** identifies a campaign session, and a **trial** identifies a training recipe or run within its study. Trial 00 in two studies can have different data, weights, budgets and outcomes. The table describes what each phase introduced. For measured results, follow the applicable validation report; a spec header or checked task records a different kind of evidence.

| Specs | Contribution | Git landmark |
| --- | --- | --- |
| [001](specs/001-oeis-neurosymbolic-synthesis/spec.md), [002](specs/002-synthesis-bootstrapping-and-soundness/spec.md), [003](specs/003-algorithmic-generalization-and-credit-assignment/spec.md), [004](specs/004-decoupled-grounding-and-symple-engine/spec.md) | Original neuro-symbolic design, bootstrapping and executable soundness, shortcut learning and credit assignment, then separate structural proposals and numerical grounding. | The initial Git import `627301c2` already contains these designs. |
| [005](specs/005-trustworthy-synthesis-readiness/spec.md) and [006](specs/006-multilimb-curriculum-scaling/spec.md) | Linked synthesis and proof claims to real checkpoint evaluation. Multi-limb arithmetic addressed sequences that outgrow a single machine word. | 005 is in the initial import. The 006 spec appears at `62139dcc`. |
| [007](specs/007-experiment-foundation/spec.md) | Established the experiment foundation: numerical profiles, isolated roles, independent arithmetic, generic training inputs and durable acceptance evidence. | PR #1 merged at `f21fb714`. Portable runtime work followed at `3e8006ee`. |
| [008](specs/008-bigint-constant-synthesis/spec.md) and [009](specs/009-pipeline-throughput/spec.md) | Introduced the bigint language and native fitting service, then measured CPU/GPU scheduling, storage, throughput and production readiness. For old fitters, codecs and run scripts use the [migration inventory](specs/008-bigint-constant-synthesis/retirement.md). | 008 design merged at `b96b92ff`. The 009 spec was added at `8657b683`. |
| [010](specs/010-pilot-generalization/spec.md) and [011](specs/011-controlled-generalization/spec.md) | Repaired template and checkpoint comparisons. Added controlled studies of structural coverage, refreshed data, parameterized targets and verified training-only replay. | PRs #61 and #62 merged at `fba656c4` and `efce05d6`. |
| [012](specs/012-parameterized-adaptive-curriculum/spec.md) | Compared fixed and adaptive allocation of training examples, with and without scheduled review. Its trained arm **D** became a retained reference for later studies. | PR #63 merged at `9ea2d12d`. |
| [013](specs/013-progressive-mathematical-discovery/spec.md) | Put capability checks, diversity comparisons, tuning, scaling and discovery behind one registered graph. Preparation, freeze, acceptance, research, reporting and assessment have distinct owners and evidence. | PR #72 merged at `7b8914e4`, including the two stacked features below. |
| [014](specs/014-repository-cleanup/spec.md) and [015](specs/015-learned-program-search/spec.md) | 014 tested compatibility and rollback when retiring repository content. 015 used the campaign infrastructure to study program selection and generation at matched search cost. | PRs #73 and #74 merged into 013 at `c6f7a092` and `95380f62`. |

To trace a decision, compare the design, the implementing change and the record produced by the run. These commands work from a repository checkout.

```bash
git log --first-parent --oneline --merges origin/main
git log --oneline -- specs/012-parameterized-adaptive-curriculum
git show --stat 95380f62
```

## Reading experimental results

**A trial has several budgets.** Training tokens and optimizer updates measure exposure. Search attempts and time determine how many proposals can be tried on one problem. Fitting, scoring, checkpoint I/O and preparation also consume resources. Keep training-only time, end-to-end elapsed time, GPU lease time and sampled device activity labeled separately. The 013 tuning configuration illustrates a 16-trial design with 12 learning-setting trials and four model-size trials. Those are historical design counts. Executed looks and actual charges come from the study journal and receipts. [Tuning policy](configs/foundation/progressive/tuning.json) · [Trial journal](src/oeis_learn/generalization/progressive/trials.py) · [Timing tools](tools/research_timing_diagnostics/)

**Ask what transferred.** Parameter tests change constants within a familiar structure. Composition tests combine familiar operations in held-out arrangements. Unseen-structure tests reserve entire structures. OEIS cases test transfer to observed real sequences. Retention measures previously supported tasks. Report the count of verified solves, measured misses and unresolved cases across the full roster. Some comparisons weight families equally so a large family cannot dominate. Missing-outcome bounds and variation across independently trained seeds answer different questions. For attribution, compare the model with templates and, where registered, shuffled or absent model input while the fitter continues to see the real prefix. [Experiment contract](specs/013-progressive-mathematical-discovery/contracts/experiment.md) · [Metrics implementation](src/oeis_learn/generalization/progressive/metrics.py)

| Historical observation | Why it matters when designing or interpreting a trial |
| --- | --- |
| In spec 010’s matched policy comparison, all 28,387 attempts decoded successfully in both runs had identical proposal hashes. Changing how the worker queue drained still affected fitting completion. | Record candidates, fits and selected outputs separately. An end-to-end gain can come from execution even with unchanged neural proposals. [Matched pilot review](specs/010-pilot-generalization/validation/010-followup-004-review.md) |
| In spec 012’s R1 study, arms B, C and D each trained for about 25M charged tokens. B used fixed allocation, C alternated adaptive allocation and review, and D used adaptive blocks throughout. D solved 42 of 68 certified cases, with 58 complete. Forty of its 42 solves also succeeded when the model input was absent. | A fitter can make a model-assisted system effective even when the evidence for learned conditioning is weak. Historical A was an unmatched literal-weight reference. [Training arms](specs/012-parameterized-adaptive-curriculum/validation/research-plan.md) · [R1 findings](specs/012-parameterized-adaptive-curriculum/validation/research-results.md) |
| On a separate structural cohort, D with its original selector solved 50/68 and completed 64/68. Generic templates solved 55/68 and completed all 68. D’s decode calls consumed 954.7 seconds of a 1,082.8-second measured four-control portfolio span. | Check both learning and conventional baselines. Measure slow decode tails, completed attempts and native fitting under the real policy before projecting cost. These timings belong to that cohort and machine. [Fresh structural comparison](specs/012-parameterized-adaptive-curriculum/validation/fresh-002-review.md) |

## First commands

From a fresh checkout of [grugnog/oeis-learn](https://github.com/grugnog/oeis-learn), run these from the repository root. The host launcher uses Python 3.11+ and the standard library. Help works before installing the model stack or starting a container.

```bash
python3 -S scripts/foundation_run.py --help
python3 -S scripts/foundation_run.py --engine podman generalization --help
python3 -S scripts/foundation_run.py --engine podman discovery --help
```

For execution, use a Linux host with a local Docker or Podman connection, filesystem paths visible to that engine and working resource controls. GPU work also needs a compatible AMD device and ROCm driver. The research workstation recorded for the progressive studies was a Ryzen AI Max+ 395 with 128 GB shared RAM, NixOS, Podman and ext4. Its measurements give a reference point for reproducing a study. Container dependencies come from the root Dockerfile and committed locks. [Operator prerequisites](docs/fresh-training.md) · [Workstation profile](specs/015-learned-program-search/plan.md)

| Task | Entry point and inputs |
| --- | --- |
| Prepare a fresh training run | The [fresh training guide](docs/fresh-training.md) covers bulk fetch, preparation, validation of already-sealed inputs, smoke, benchmark, train, resume and package. The [foundation quickstart](specs/007-experiment-foundation/quickstart.md) includes a fixture route. |
| Run a controlled generalization comparison | The [011 quickstart](specs/011-controlled-generalization/quickstart.md) binds study and runtime, qualification and role-specific inputs. Use `scripts/generalization_inputs.py --help` with Python for the input assembler. |
| Operate a progressive campaign | Follow the source-pinned [013 workstation instructions](specs/013-progressive-mathematical-discovery/workstation-handoff.md) for a matching campaign. Inspect progress with the Git-backed [research progress reader](specs/013-progressive-mathematical-discovery/validation/research-progress.py). Use the specific study’s launcher and verifier when it has its own archive format. |
| Verify an existing 013 report | Supply the **report** ZIP to `discovery verify-bundle`, along with a new output directory. A support ZIP named `run-research.zip` can instead contain execution diagnostics. Use the bundle type and its source-bound reader to identify the evidence. [Reporting contract](specs/013-progressive-mathematical-discovery/contracts/cli.md) |

For example, if you have an original 013 report, place a copy under the checkout and substitute its workspace-relative path below. The reader writes a verification record to the specified directory and can check the report independently of model training.

```bash
python3 -S scripts/foundation_run.py --engine podman discovery verify-bundle \
  --archive run-state/review/report.zip --output tmp-validation/report-review-001
```

In a progressive campaign, preparation assembles inputs and exposure records, freeze binds the roster and resource reservations, and acceptance checks the registered execution path. Research uses that frozen campaign and writes task receipts. Reporting and verification can proceed from saved evidence after execution ends, including when the scientific comparison is incomplete. Final assessment has its own access decision. [Campaign CLI contract](specs/013-progressive-mathematical-discovery/contracts/cli.md)

## Development and testing

The Python package uses a `src/` layout and `uv.lock`. The installed console, `oeis-learn`, also exposes ingestion, legacy experiment commands and the portable pipeline. The development group includes pytest and the formatting/type tools. For a focused first check of scientific accounting in a development environment:

```bash
uv sync --locked
uv run oeis-learn --help
uv run pytest -q tests/unit/test_progressive_metrics.py
```

| Layer | What to exercise |
| --- | --- |
| Unit and contract | Pure Python reducers, roster separation, parser and schema boundaries, budgets and incomplete outcomes. Start with [unit tests](tests/unit/) and [contract tests](tests/contract/). |
| Integration and production entry points | Generate, score, interrupt, restore and export using the installed roles. Verify a saved report independently. [Reporting integration](tests/integration/test_progressive_reporting.py) · [Portable CLI tests](tests/integration/test_portable_cli.py) |
| CPU/native and packaging | The `qualification` image builds the native components. The default `gpu` target supplies ROCm. CI exercises native/reference parity, engine isolation and installed resources outside the checkout. The manual release audit tests the built wheel. [Dockerfile](Dockerfile) · [CI jobs](.github/workflows/foundation.yml) · [Manual release jobs](.github/workflows/foundation-release.yml) |
| Physical hardware and research | A real HIP update, checkpoint restore and device-bound receipt establish hardware behavior for that candidate. Separately, freeze and accept the study before research dispatch. A scientific gain requires the study’s own comparison and verification. [Hardware validation](specs/013-progressive-mathematical-discovery/hardware-validation.md) |

A CPU/native qualification image can be tested independently on a configured host. Use a fresh output path for a new validation. To resume an existing validation, use its original command and output so completed stages can be authenticated.

```bash
podman build --target qualification -t localhost/oeis-learn:cpu .
python3 scripts/foundation_run.py --engine podman --image localhost/oeis-learn:cpu \
  validate --suite cpu --output tmp-validation/cpu-001
```

Two targeted developer checks are also useful. `python3 scripts/inventory_synthesis_retirement.py --check` checks the 008 retirement inventory. On Linux with the development dependencies installed, `OMP_NUM_THREADS=2 uv run python scripts/verify_phase11.py --output runs/007-cpu-review` exercises the older 007 fixture workflow, including pause/resume and evidence packaging. Choose a new output directory for a new fixture run. [Original foundation handoff](specs/007-experiment-foundation/validation/handoff.md)

For physical GPU or combined pipeline validation, use the [GPU procedure](docs/fresh-training.md#build-and-validate) or [pipeline guide](specs/009-pipeline-throughput/quickstart.md). The latter distinguishes `all-ready` validation from a separately admitted research study. When modifying an installed file, check the wheel outside the checkout and exercise its normal caller. Configuration, schemas, contracts, script-path calls and dynamic role lookup can be dependencies even when a static import search finds no callers. [Cleanup and compatibility evidence](specs/014-repository-cleanup/implementation.md) · [CI image and cache behavior](docs/ci-performance.md)

## Preserving a workstation run

A **qualified donor** is the original checkout, images and saved outputs that a later study may reference. Keep its source pin and run state together. From a pinned checkout, Git can fetch and read a newer standalone operator helper without changing HEAD. Use the helper’s documented full commit. Some helpers also require the matching package source, which their operator instructions extract together. Changed runtime or scientific inputs require their own compatibility evidence. [Recovery history and pins](specs/015-learned-program-search/research.md#recovery-history-that-should-shape-the-implementation)

Where a newer helper is needed, a typical read-only Git sequence is shown below. In this example the file is documentation. Use a recovery script only with its own pinned instructions.

```bash
git fetch origin main
git show 7b8914e4:specs/013-progressive-mathematical-discovery/quickstart.md
```

A session directory holds stage journals, logs, freeze and acceptance records and shareable outputs. Sibling directories under `run-state/`, including `discovery-control/`, `discovery-learning/` and `discovery-accounting/`, hold controller transitions, checkpoints and charges. A full learner resume needs optimizer, RNG and data cursor as well as weights. A failed or interrupted attempt retains its accounting. The documented resume path can credit a validated output once and continue at its saved boundary. A status reader or report watcher observes the run. Starting or resuming it is a launcher operation. [State and recovery contracts](specs/013-progressive-mathematical-discovery/contracts/execution.md) · [Recovery and reporting examples](specs/013-progressive-mathematical-discovery/quickstart.md#recovery-and-independent-review)

Before expensive work, check available **bytes and inodes on the actual mount**, the projected peak while inputs and outputs coexist, and what shares the storage. A directory’s apparent size, a cached image layer or a historical free-space reading alone cannot tell you what is reclaimable. Keep checkpoints, verified calibration, repaired panels, evidence dependencies and failure receipts until a scoped retirement proves they are disposable. Record worker time, supervisor time and actual GPU activity separately. A deadline is an allowance; use measured comparable runs for an estimate. [Storage investigation](specs/013-progressive-mathematical-discovery/validation/round019-storage-continuation.md) · [Qualification retirement](specs/013-progressive-mathematical-discovery/validation/round019-qualification-cleanup.md)

## How to design a study

Start a new spec with the decision the experiment should enable. A useful reading order is the nearest prior feature’s `spec.md`, `plan.md`, `contracts/`, then its dated `validation/` reports. The [constitution](.specify/memory/constitution.md) defines exactness, independence, exposure and scoped claims. The [Spec Kit templates](.specify/templates/) and [repository workflow](.agents/skills/) take a question through specification, planning, tasks and implementation. The [013 experiment contract](specs/013-progressive-mathematical-discovery/contracts/experiment.md) and [015 experiment contract](specs/015-learned-program-search/contracts/experiment.md) are worked examples at different scales.

- **Define the claim and comparator.** Write down what the model sees, what the fitter sees, what is held out, which single output earns credit, the unit of comparison and which observation would change a spending decision. Separate exploratory measurements from evidence needed to adopt a model.
- **Account for data exposure.** Identify the source, offset, family, split, previous use and training eligibility. Reserve independent families before scoring. Difficulty calibration and training examples need their own provenance. Repeated looks at development results make those cases exposed.
- **Match the work.** State how training tokens, seeds, search and fitting allowances, retention and unresolved cases enter the comparison. Base a critical new limit on production measurements with margin. If those measurements are unavailable, use a conservative provisional value and name the validation that will replace it.
- **Build on existing evidence.** Inventory immutable inputs, calibration and prior outputs. Record exactly what the original source and images qualified. Plan for interrupted updates, stage publication, and independent reading of the exported result. Rehearse the largest expected inputs through the final reader.
- **Leave a usable handoff.** Put exact Git and invocation commands in the feature quickstart. Record the commit, input and output identities, denominators, costs, negatives and open questions in the validation report. Keep private sequences and model weights in their authorized workspace. A future reader should be able to tell which claims they can check from Git, which require a returned archive, and which require the workstation.
