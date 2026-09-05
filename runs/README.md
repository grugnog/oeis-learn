# OEIS Learn Experiment Run Tracking

This directory contains tracked experiment runs, benchmark logs, hyperparameter configurations, checkpoints, and generated reports.

---

## 📁 Run Directory Structure

Each experiment or long-running benchmark is allocated an isolated directory: `runs/<RUN_ID>_<NAME>/` (e.g., `runs/001_baseline_cold_start/`, `runs/002_phase2_bootstrapping/`).

```text
runs/
├── 001_baseline_cold_start/
│   ├── config.yaml                   # Hyperparameter configuration snapshot
│   ├── metadata.json                 # Run metadata (timestamps, status, host specs, summary metrics)
│   ├── checkpoints/                  # Saved model weights (.pt)
│   │   ├── model_epoch_010.pt
│   │   ├── ...
│   │   └── model_epoch_100.pt
│   ├── logs/                         # Execution logs & telemetry
│   │   ├── run.log                   # Full timestamped console log
│   │   └── telemetry.json            # Step-by-step telemetry (entropy, ACR, trap rate)
│   └── reports/                      # Evaluation reports & proofs
│       ├── preflight_report.json     # Pre-flight Tiers 0-3 validation report
│       ├── summary.md                # Markdown execution summary & synthesis metrics
│       ├── discovered_theorems.md    # Formal SymPy proofs & PSLQ integer relations
│       └── synthesis_results.json    # Extrapolation & MDL evaluation data
└── 002_phase2_bootstrapping/
    └── ...
```

---

## 📑 Tracked Runs Index

| Run ID | Name | Date | Status | Duration | Stage 1 Pass Rate | Key Innovations / Notes |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| `001` | `baseline_cold_start` | 2026-08-30 | `COMPLETED` | 18.65h | 0.0% | Initial Phase 1 baseline. Random weight initialization with binary rewards, revealing cold-start zero-advantage collapse. |
| `002` | `phase2_bootstrapping` | 2026-08-31 | `READY` | ~11.0h | Target >80% | Phase 2 engine: Pre-flight progressive checks (Tiers 0–3), Synthetic SFT warmup (2,500 pairs, 5 epochs), S-GRPO + CGI trajectory injection, cosine-annealed composite reward shaping. |
| `007` | `phase4_production_symple` | 2026-09-04 | `COMPLETED` | 53.30h | 11.8% | Phase 4 Decoupled Diophantine/SMT grounding, SYMPLE multi-task engine, parsimony DCE, and 2 proven theorems. |

---

## 🔒 Qualification States & Governance

Production promotion enforces strict lifecycle transitions:

- `INITIALIZED`: Run workspace allocated with immutable manifest snapshots.
- `PREFLIGHT`: Executing 4-tier progressive readiness gates.
- `AUTHORIZED`: All mandatory gates passed. Eligible for `COMPLETED_QUALIFIED`.
- `BLOCKED`: At least one mandatory gate failed. Cannot launch production training.
- `OVERRIDDEN_UNQUALIFIED`: Authorized for diagnostic investigation with immutable operator, reason, and intent records. Permanently excluded from graduation or best-run eligibility.

---

## 🛠️ Python & CLI Usage

### Programmatic Usage

```python
from oeis_learn.tracking import RunManager

# Initialize RunManager (auto-detects runs/ in workspace)
manager = RunManager()

# Create a new run (auto-allocates sequential ID like '002')
ctx = manager.create_run(name="phase2_bootstrapping", config={"lr": 3e-4, "epochs": 60})

print("Run Directory:", ctx.run_dir)
print("Checkpoints Dir:", ctx.checkpoints_dir)
print("Log File:", ctx.log_file)

# Update run status & metrics
ctx.set_status("RUNNING")
ctx.record_summary_metrics({"final_pass_rate": 0.85, "competence": 0.90})
ctx.set_status("COMPLETED")
```

### Listing All Tracked Runs

```bash
python -m oeis_learn.cli.main list-runs
```
