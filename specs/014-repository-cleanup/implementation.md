# Cleanup implementation review

## Comparison baseline

PR #73 is stacked on PR #72. Planning used `a2618bbd`. Implementation began at `a3f9808e1a6b93f45e52279e0878435fedf5bb02`, which includes the screening audit from `5489aed2`. Owen’s qualified checkout stays at `77667e67176ae32fa16457d52565bae1dbe41afc`. Detached parent, candidate and reversal worktrees used separate output directories.

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

The AST scan found 18 `run_path` calls and one `import_module` call in 1,179 Python files. Grimp 3.20 reported 558 modules, 3,291 import edges and zero direct worker importers. Manual tracing found both dynamic paths. CI and the workstation handoff reach `qualify_bigint_phase5.py` by script name. The workstation calls research preparation through `progressive_research_workstation` and `progressive_research_launch`. The supervisor resolves frozen operations through `REGISTRY` and starts the installed worker with `runpy.run_module`. Vulture 2.16 reported 15 leads at 100% confidence. The synthesis retirement inventory passed. [Reproduce the probes](quickstart.md#2-cross-check-code-and-data-consumers)

Retain `audit-round027-screening.py` and `test_round027_screening_audit.py`. The audit follows the frozen study, controller and producer receipts for six screening trials. Their Git blobs matched `5489aed2` on the combined candidate, and both tests and seven subtests passed on each revision. [Source and tests](https://github.com/grugnog/oeis-learn/tree/5489aed2/specs/013-progressive-mathematical-discovery/validation)

The complete qualification maps matched byte-for-byte, with 1,332 keys and digest `sha256:e1c6bfb7838cc0999d2477978bdfa890fb4f504b18b47211771b79bdde37d073`. The HPO launcher is included. The three reports, both HPO YAMLs and the audit files are outside the map. Docker excludes `reports/` and the sdist has no rule for it. The historical direct and sdist-built wheels had 1,332 and 1,331 keys. Round009 donor reuse passes at `77667e67` and already rejects `budgets.py` at `a2618bbd`. The guide covers the separate Git, installed-resource, native and image identities and the passing-control gate for source changes. [Packaging and donor results](research.md#source-identity-and-packaging-controls)

## Accepted report removals

| Group | Immediate parent | Removal commit | Restored Git blobs (mode `100755`) | Tracked bytes removed |
|---|---|---|---|---:|
| T007 · `reports/hpo_sweep_results.{json,md}` | `dbd565ed` | [`4fbb124d`](https://github.com/grugnog/oeis-learn/commit/4fbb124d9f2069cf01ccba12bfc16de4772b7c02) | `d1afaad7050560995ce789964ca6a050a62bf40d` (26,647), `7cdbcbae7f26f30f24a5a005877a11bffdd2c660` (3,570) | 30,217 |
| T008 · `reports/real_data_e2e_report.md` | `4fbb124d` | [`b3dfb484`](https://github.com/grugnog/oeis-learn/commit/b3dfb484c097749c095529ebe957a39c580bffa5) | `968e466ecadd5e30f10285c14c30c6644f8807d2` | 1,005 |

Searches covered basenames, paths, hashes, globs, guide links, CI and packaging in each worktree. The legacy archiver selects three named long-run reports. Research receipts bind individual producer hashes. References to these three archived outputs came from writers and the cleanup inventory.

- **HPO.** A single encoder and RL row regenerated both default outputs, with the winning constants, hyperparameters, scores and both ranking tables. The launcher still uses `configs/hpo_search_space.yaml` and applies winners to `configs/train_tier1.yaml`. Normalizing the UTC timestamps yielded JSON SHA-256 `0f49ab6847fa99b2a135667b0bd4e7de55da144a0b567892193f476fb0b34bb9` and Markdown `f0fed384d32d761002f48d525821c960834c277147f66a3de6e8dd6db1d2d329`. [Exporter](../../scripts/run_hpo_sweep.py)
- **Real-data proofs.** The integration wrote its own temporary report. Fibonacci/Lucas remained a numerical conjecture. The triangular/square identity retained its certificate and external assumption. Altered evidence failed verification and legacy `PROVEN` rendered as `UNVERIFIED_LEGACY_LABEL`. All three tests passed. Normalizing only the certificate’s `verified_at` yielded identical Markdown, SHA-256 `7e975f375c3f3acbbd6ae1f5ede10849f63f5378060dca334a99ca33bdf943f3`. [Writer](../../tests/integration/test_real_data_e2e.py) · [Proof tests](../../tests/unit/test_foundation_symbolic.py)

Each `git revert <removal-commit>` restored its entire parent tree on a disposable checkout. Independently reverting either commit from the combined tip restored only its own blobs and modes. Both writers and the three proof checks passed with the other deletion still applied. The four wider regression selections also passed on the merged baseline and combined candidate (23 + 40 + 41 + 3).

Run the HPO comparison from either worktree with the project test dependencies.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src:$PWD" python - <<'PYTHON'
import hashlib, json, os, re, runpy, tempfile
from pathlib import Path

write = runpy.run_path("scripts/run_hpo_sweep.py")["export_hpo_reports"]
encoder = dict(primes=[2, 3], primes_name="tiny", num_primes=2, max_valuation=8, moduli_count=5,
               compound_score=.75, latency_seconds=.5,
               metrics=dict(final_oracle_ppl=1.25, tier2_max_pass_rate=.5, rank_dispersion_ratio=.25))
rl = dict(compound_score=.625, metrics=dict(tier2_max_pass_rate=.25, tier3_acr=.125),
          hyperparameters=dict(learning_rate=.0003, beta_sft=.2, beta_kl=.05, alpha_ent=.01, gamma_pbrs=.99))
with tempfile.TemporaryDirectory() as out:
    os.chdir(out)
    write([encoder], [rl])
    data = json.loads(Path("reports/hpo_sweep_results.json").read_bytes())
    del data["timestamp"]
    md = Path("reports/hpo_sweep_results.md").read_text()
    md, dates = re.subn(r"(?m)^\*\*Date\*\*: .* UTC  $", "**Date**: <UTC>", md)
    assert dates == 1
    for content in (json.dumps(data, sort_keys=True), md):
        print(hashlib.sha256(content.encode()).hexdigest())
PYTHON
```

## Original evidence and trial 00

| Task | Preserve on the workstation | Evidence |
|---|---|---|
| T023 | Sealed replay, 12 round027 outputs and 35 producers in rounds 022–025, four diversity checkpoints, controller, original ledger and `diversity-comparison/publish/97` reservation and receipts | Private recovery rehearsal open. |
| T029 | Immutable report ZIP and its independent saved receipt | Operator verified SHA-256 `d24c9ad72551e927cb63b2b77b18f31ea585733b767990d14c4eddf764626c9e`, 352 accounting rows, `INCOMPLETE` and only `primary_comparison` missing. |
| T031 | Diagnostics ZIP, sealed replay, eight diversity result receipts and their generation and scoring outputs for both arms and both seeds | Operator authenticated all eight. |
| T037 | Six full screening checkpoints, study proposals, controller, learning and accounting journals, and each trial’s development and retention generation and scoring outputs | Bounded CPU restore passed. The original trial 00 checkpoint and its bindings were unavailable in this workspace. |

The first-dispatch, confirmation and screening fixtures passed all eight cases on the merged baseline and combined candidate. T037 compiled the 1M→2M screening link and bound the saved ordinal-zero proposal. The CPU fixture scaled the targets to 480→960 tokens. Before the child’s first update, it recovered weights, AdamW, RNG, controller cursor 32, two completed updates and all 480 tokens. The offline check preserved the entire cost journal, including the parent’s elapsed charge. The installed worker also paused with zero new training tokens. T024 separately exercised the 12.5M→25M confirmation link. The private restore remains a prerequisite for further training. [Original and fixture procedures](quickstart.md#check-trial-00-before-further-training)

## Repository sweep and third removal

The retired 008, 009 and 010 launchers wrote the following synthetic SFT warm-up inputs into their own run directories. The old scripts and generators can be retrieved at `b96b92ff`. Searches of the tracked parent found no incoming basename, path, Git blob or SHA-256 reference to the datasets. The current benchmark generates its own `ctx.run_dir / "sft_demonstrations.json"`. The public generator and warm-up commands use `data/sft_demonstrations.json`. The run010 checkpoint consumed by warm-start is a separate artifact. [Retirement inventory](../008-bigint-constant-synthesis/retirement.md) · [Current benchmark](../../scripts/run_long_e2e_benchmark.py)

| Removed input under `runs/` | Examples | Bytes | Restorable Git blob |
|---|---:|---:|---|
| `008_targeted_stage1_stage2/sft_booster_demonstrations.json` | 1,500 | 1,203,056 | `b32865d6bf8afa328d8725a04b9348877b367b46` |
| `009_stage2_recurrence_sprint/sft_recurrence_demonstrations.json` | 2,000 | 1,811,603 | `26d5a1eb075daae19af40bb537f203bd7b352f2e` |
| `010_overnight_production_curriculum/sft_overnight_demonstrations.json` | 2,000 | 1,577,788 | `6369e0ab35b0977a8e7a15d7096a1a82b9802eca` |

The group removed 4,592,447 tracked bytes from parent `a318868c`. On its parent, candidate and independent reversal, `generate-sft --num-samples 6 --seed 73` produced six loadable examples with SHA-256 `9f331261df93d8fbc53a7fbcfe429a868d16c37a9174613010b54ffc1d7b6111`. All 12 run-list entries agreed after normalizing the checkout prefix. The generator, sweeps, dataset-contract and run-manager selections passed six cases on each tree. All 1,332 qualification-map entries matched. Docker excludes `runs/` and the package manifest omits the run directories. Reverting the group restored the complete parent tree, including all three executable modes. [Removal and rollback target](https://github.com/grugnog/oeis-learn/commit/4cbc77797098b4d1554784e96d4d28dd68c6c714)

| Area | Disposition and reason |
|---|---|
| Python, scripts and tools | Hold `_populate_canonical_defaults` and the HPO launcher/default YAML after the qualified donor failures below. The current run011 caller seeds multilimb canaries. The retirement inventory validates current reachability and the timing collector reads generation/scoring timings for workstation recovery. Preserve both commands and their tests. The calibration stamp arguments distinguish cached calls. |
| Configs | Retain `train_tier2.yaml` for the specified d=768 multi-GPU stage after Tier 1 graduation. The training CLI accepts an explicit config. The operator diagnostics template is an input to preparation. The installed diagnostics policy is a package resource with an identity check. The packaging test compares them. |
| Runs and reports | Retain run001 metadata for the run index and its 741-byte summary for the six per-sequence parse failures underlying the spec 002 diagnosis. Metadata alone preserves the aggregate duration and rate. The run008/009 synthesis records retain per-sequence evidence for their summaries. Their discovery reports are identified by the surviving theorem reports. |
| Manuals and tooling | README routes operators to USAGE and the foundation guide. The Pi task prompt serves the integration selected in `.specify/integration.json`. The agent skill exposes a second task entry point. Pinned recovery scripts remain retrievable by revision. |
| Tests, CI, native build and data | The four foundation partitions feed one checked inventory. The native build fingerprints the Dockerfile, rustfmt config and build scripts. Trustworthy-synthesis benchmarks feed current readers. Numerical counterexamples, certificate/legacy-label rejection and independent report tampering checks exercise shared behavior. Retain those fixtures, installed schemas, policy copies and the inputs to both image targets. |

[Canary caller](../../src/oeis_learn/cli/train_run011.py) · [Tier 2 requirement](../001-oeis-neurosymbolic-synthesis/spec.md) · [Run001 evidence](../../runs/001_baseline_cold_start/reports/summary.md) · [Run001 diagnosis](../002-synthesis-bootstrapping-and-soundness/research.md) · [Package contract](../../tests/contract/test_progressive_packaging.py) · [Native identity](../../crates/oeis_wasm_evaluator/build.rs)

Both source-changing trials used the round009 donor at `7332f0be`. The clean qualified control at `77667e67` accepted 89 changes with map digest `2561f82fb31364557e42af5ed7ff7558c3e8c27d307fb805ac301cd1cb68b750`. In separate Git roots, the changed key, modification/deletion type and bytes agreed with the independently exported group. The previous reporter and the integrated dataset removal each rejected `generalization/progressive/budgets.py` with the previously recorded reporter map.

| Disposable group | Reporter trial map / first rejection | Qualified patched map / first rejection |
|---|---|---|
| Delete the old elite-buffer method | `6276d1522be9f19ff8b740eece4e114c976bda0cd05d9b9d9e259b75175865da` · `budgets.py` | `f9d16289bba2beaf4fd14f06b9742050cebb987567da9d0ce15702f7c5ca0d48` · `src/oeis_learn/rl/elite_buffer.py` |
| Remove HPO launcher and its default config | `ce2edbaf21b941a0e9bdbf8ac97a52bd3852286ad0b56d42f7365018c904d420` · `run_hpo_sweep.py` | `2c886a01119d9edad7efe71e32b57d600bd404430766ef0b26a2b656c785e64f` · `scripts/run_hpo_sweep.py` |
