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
