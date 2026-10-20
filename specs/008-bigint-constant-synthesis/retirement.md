# Retirement disposition — spec 008 (T001)

Inventory baseline: `b96b92ff680a1509d3cede89e9abf0ddb65d2665`. Phase 4 code dispositions are closed below; clean installed-image validation remains pending T088. Check current reachability with `python3 scripts/inventory_synthesis_retirement.py --check`; this does not overwrite historical inventory decisions. The path-by-path direct consumer inventory is [retirement-inventory.json](retirement-inventory.json). Phase 4 must close each row and rerun a whole-repository import/caller audit, including dynamic imports and installed distributions.

| Old surface | Consumers | Disposition / replacement | Rationale / regression |
| --- | --- | --- | --- |
| `decoder/constant_solver.py` public parsing/solving/splicing API | CLI `solve-constants` (handle_solve_constants); evaluation experiments/synthesis; RL trainer; run008/009/010; tests in inventory | Migrate supported callers to `synthesis` transport/admission; delete implementation at T085 | One native fitter; retain nonlinear square (16 vs 4), affine, inconsistent and recurrence fixtures |
| `decoder/grounding.py` parser, affine/SMT/ground/splice | constant_solver, experiments/config, repair_gate, grounding_reference/evidence and tests | Delete after native fitting and independent parser are qualified | Keep intermediate overflow, control rejection, visible-only and final-source checks |
| `decoder/dixon_solver.py`, `qfnia_solver.py`, `mersenne61_filter.py` | grounding, dixon, algebra/filter tests | Retire; native exact affine/QF_LIA/QF_NIA service | Preserve singular modular pivot, rank-deficient integer feasibility and nonintegral solution cases |
| `decoder/recurrence.py` fitting/recognition/reference | grounding/evidence and recurrence tests | Retire fitting/recognition; independently derive required reference fixtures | No hidden recurrence fallback; Fibonacci/order2 fixtures remain |
| `decoder/grounding_reference.py` | grounding, evidence, elite_buffer | Replace with independently parsed final-source reference, then delete | Current import of production instructions prevents independence |
| `decoder/grounding_evidence.py` | qualified_models, grounding, symbolic pipeline tests | Native scoped replay plus read-only historical reader; delete old executable module | Preserve `historical_unreplayed`; replay is scoped evidence, not theorem proof |
| `fresh/search.py::grounded_variants` | fresh/worker; test_bootstrap_workflow | Remove duplicate literal/operator edits; route fitted portfolio to one native service | Retain raw/fitted identity and independent admission tests |
| Python `z3-solver` (`pyproject.toml`, `uv.lock`) | grounding/qfnia/evidence **and discovery/loop_certificate.py** | Remove only after certificate consumer migrated to native or independently separated at T083/T087 | Do not delete certificate validation to satisfy dependency audit |
| SymPy | discovery polynomial/recurrence certificates, formula_syntax, numerical_validator, pslq_solver; evaluation/engineering; real-data tests | Retain while these independent consumers remain supported | Not solver-only; preserve certificate/discovery tests |
| `scripts/run_008_learning_run.py`, `run_009_recurrence_sprint.py`, `run_010_overnight_curriculum.py` | Historic launch paths | Retire scripts with exclusive `configs/train_run008.yaml`, `train_run009.yaml`, `train_run010.yaml` after confirming config references | Reproduction uses baseline git history; saved results/specs unchanged |
| `configs/train_run011.yaml::enable_z3_qfnia` and old method settings | Experiment config/CLI dispatch | Audit and remove unsupported choices with migration errors | No silent selection of old algorithm |
| CLI main, evaluation experiments/synthesis/repair_gate, experiments config/qualified_models | Current supported commands/qualification | Migrate at T082, preserve supported old-profile controls | Replacement regression tests before deletion |
| RL trainer/elite_buffer | Constant solving and public evidence acceptance | Migrate at T083; provisional results cannot enter replay | Keep evidence rejection and training-only admission |
| `data/models.py::ASTSkeleton,ConstantSolverResult` public transport records | Old constant-solver API and owning unit/contract tests | Migrate supported fit transport; retain only deliberate read-only historical serialization | Do not retain callable solver algorithms behind old records |
| `configs/train_tier1.yaml::solver` | Earlier training recipe | Migrate supported recipe or retire with explicit option error | One native fitter; no legacy method fallback |
| Tests importing old interfaces (inventory) | Unit/contract/integration suites | Port assertions/fixtures at T084; remove obsolete API assertions only with replacement coverage | No retention of executable compatibility solvers |
| Signed-256 evaluator, codec, checkpoint readers, strict validator | Supported compatibility/control | Retain unchanged numerical/codec semantics; record changed build identities | Existing arithmetic, cache, hybrid and codec tests |
| Historical specs, contracts, provenance/results | Readers/reproduction | Retain original semantics and status; no rewritten histories | Baseline SHA above identifies old executables |

No examples directory exists at the baseline. Other subsystems remain supported unless explicitly dispositioned after consumer review. No removed implementation may be copied/renamed or installed as an executable fallback. New files/consumers introduced after this inventory must be audited before T088 closes.

## Phase 4 closure

Every baseline consumer row has a code disposition in
[retirement-inventory.json](retirement-inventory.json). T081–T087 remove eight
implementation modules, their four obsolete public result/skeleton/filter
records, the three inventoried historical launchers and three exclusive configs.
Retained run011/tier1 diagnostic configs have explicit native settings; they are
not new training recipes. No historical result/specification is deleted.

Supported CLI/evaluation/RL/activation callers use `synthesis.fitting`; the fresh
worker already used the Phase 3 native client. New qualified native fit records
replay production execution plus independent admission or rederive scoped UNSAT
through the native worker. Old qualified grounding envelopes remain valid JSON
shapes but can only return `historical_unreplayed`. Legacy teacher ingestion is
explicitly unsupported; verified training-archive admission remains.

The non-fitting loop-certificate checker preserves its existing independent
obligations while serializing SMT-LIB to `certificate-solver`, built against the
same pinned Rust/Z3 native closure. It exports no assignment/model search API.
Python `z3-solver` is removed from package metadata/lock and uninstalled from the
image if inherited from its base. SymPy consumers listed above remain supported.
The source audit detects retired imports (including literal dynamic imports),
old callable names and unchanged nontrivial function bodies copied under another
name. Clean installed audits use `python -I` outside the checkout and check both
importability and distribution dependencies.

Replacement regressions retain polynomial, square16-versus4, shared/large
coefficients, singular prime pivots, integer feasibility/nonintegrality,
inconsistent systems, carried-state recurrence arithmetic, checked intermediate
overflow, visible-only fitting, provisional/tampered replay rejection and explicit
unsupported old control/bitwise forms. Old modular-filter implementation-only
microsecond/rank assertions are retired; exact integer rejection is the owning
acceptance test. No global optimum is asserted for bounded discovered solutions.

The budgeted `evaluation/search.py::typed_repairs` proposal generator is retained:
it does not read visible target values or fit parameters, and each generated
source is charged/checked under the existing search ledger. It is not a renamed
fitter or a replacement for native admission.

The code inventory is not clean-install proof. T088 closes only when the new
installed image, native regressions and preserved compatibility/certificate
checks pass. See [phase-4.md](validation/phase-4.md). Later Phase 5 qualification
and all research runs remain separate.
