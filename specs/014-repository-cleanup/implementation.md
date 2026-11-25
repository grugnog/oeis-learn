# Cleanup implementation review

## Comparison baseline

Draft PR #73 targets PR #72. The planning inventory uses `a2618bbd`. The implementation baseline is `a3f9808e1a6b93f45e52279e0878435fedf5bb02`, which incorporates `5489aed2`. Owen’s qualified worktree and the accepted-run replay remain pinned to `77667e67176ae32fa16457d52565bae1dbe41afc`. Each report group has its own immediate parent, detached trial and disposable reversal. All generated reports, logs and journals go outside the source trees.

| Area | Tracked files | Area | Tracked files |
|---|---:|---|---:|
| src | 612 | specs | 593 |
| tests | 514 | runs | 68 |
| configs | 54 | scripts | 50 |
| crates | 44 | .specify | 21 |
| docs | 20 | reports | 18 |
| root files | 11 | .agents | 10 |
| .pi | 10 | tools | 8 |
| .github | 7 | data | 5 |
| docker | 4 | **Total** | **2,049** |

The AST scan parsed 1,179 Python files and found 18 attribute calls to `run_path` and one to `import_module`. Grimp 3.20 found 558 modules, 3,291 import edges and zero direct importers of the installed worker. CI invokes the progressive foundation qualifier and the phase 5 helper. The workstation preparation chain reaches `progressive_research_launch` through `progressive_research_workstation`. The handoff loads `scripts/qualify_bigint_phase5.py` by filename. Role invocation looks up the frozen operation in `REGISTRY` and passes a literal `runpy.run_module` command to the worker process. Vulture 2.16 reported 15 leads at 100% confidence. The synthesis retirement check passed. [Reproduce the scans](quickstart.md#2-cross-check-code-and-data-consumers)

The later screening audit reads the frozen campaign, study and controller, follows result receipts to generator and scorer artifacts, and reports an exploratory sensitivity analysis for six 1M checkpoints. Retain both `audit-round027-screening.py` (Git blob `5aea4aba40e125989632eecbe95b945fd40278b6`) and `test_round027_screening_audit.py` (`5ed7d6d192238f68ecf03a9583294f35e8b15d6d`). The audit suite passed two tests and seven subtests on the implementation baseline. [Audit and tests at the source revision](https://github.com/grugnog/oeis-learn/tree/5489aed2/specs/013-progressive-mathematical-discovery/validation)

The qualification map has 1,332 entries and digest `sha256:e1c6bfb7838cc0999d2477978bdfa890fb4f504b18b47211771b79bdde37d073`. It includes the HPO launcher and excludes the three reports, both HPO YAML paths and the two audit files. The reports are excluded from Docker’s build context and have no sdist inclusion rule. Git revision, wheel resources, native inputs and image ID have separate identities. The historical direct wheel had 1,332 keys and the sdist-built wheel had 1,331. Round009 donor reuse passes at the qualified checkout and already rejects `budgets.py` at the reporter baseline. The report groups must reproduce the entire source map. Each later source-changing group requires its own delta on the passing donor control. [Packaging and donor comparisons](research.md#source-identity-and-packaging-controls)

## Report candidates

| Candidate | Former purpose | Consumer tracing and affected check |
|---|---|---|
| `reports/hpo_sweep_results.json` and `.md` | Sample encoder ablation and RL sweep export | The launcher writes these default paths. A fixed single encoder and RL row recreated both files in an empty directory. The JSON contained both trial lists, the winning constants, RL parameters and score. Markdown contained both ranking tables and the winning configuration. Preserve the launcher, `configs/hpo_search_space.yaml` and the `--apply-best` target `configs/train_tier1.yaml`. |
| `reports/real_data_e2e_report.md` | Historical real-data proof summary | The real-data integration writes a new file in its own temporary directory. Its Fibonacci/Lucas relation remained a numerical conjecture and the triangular/square relation carried checked formula evidence. The two symbolic tests also rejected altered evidence and rendered a legacy `PROVEN` claim as `UNVERIFIED_LEGACY_LABEL`. All three tests passed on the baseline. |

Searches covered each basename, relative path, SHA-256, Git blob, report directory readers and globs, CI, packaging and guide links. The cleanup documents inventory these files. The legacy run archiver reads three named long-run reports. The receipt validator reads named, hash-bound producers. Neither opens these candidates. Reversal restores the exact parent blobs with `git revert <group-commit>`; rerun the affected writer and tests in a fresh temporary directory. [HPO writer](../../scripts/run_hpo_sweep.py) · [Real-data writer](../../tests/integration/test_real_data_e2e.py) · [Proof checks](../../tests/unit/test_foundation_symbolic.py)

## Workstation originals

| Acceptance case | Originals to preserve |
|---|---|
| T023, private recovery | Sealed replay and its 47 completed outputs, including 12 in round027 and 35 whose producers reside in rounds 022–025. Follow the stored paths into those rounds. Keep the four full diversity checkpoints, original controller and accounting prefix, and the original `diversity-comparison/publish/97` reservation and terminal receipts. |
| T029, historical report | Keep the immutable ZIP, its independent saved verification receipt and accounting evidence. The operator verified SHA-256 `d24c9ad72551e927cb63b2b77b18f31ea585733b767990d14c4eddf764626c9e`, 352 accounting rows, `INCOMPLETE` and the single missing `primary_comparison`. |
| T031, diversity scale | Keep the diagnostics ZIP, sealed replay, eight authenticated original role-result receipts and matching generation and scoring artifacts for both arms and both seeds. The operator verified all eight receipts. |
| T037, trial 00 screening | Keep all six 1M screening checkpoints with their full-state components, journal cuts, proposals and matching development, retention-baseline and retention-candidate generation and scoring outputs. The frozen study and control records bind the choice of trial, model and evidence. Check trial 00’s 1M→2M inheritance on an isolated copy before any follow-up training. |

The inventory describes tracked bytes. Workstation reclamation requires inspection of the actual saved data and its cross-round references.
