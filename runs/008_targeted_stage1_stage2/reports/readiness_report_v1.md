# Readiness Qualification Report: tier1_readiness_v1

- **Report ID**: `rep_008_1788643956`
- **Run ID**: `008`
- **Qualification State**: `BLOCKED`
- **Overall Passed**: `NO`
- **Evaluated At**: `2026-09-05T21:32:36.721459+00:00`
- **Policy ID**: `sha256:cfbb7afd87874fa04a56208a8d5433739a165db4ee8a7eb2fa0aa2c8ad9fb4a8`

## Mandatory Gate Results

| Gate ID | Metric | Comparator | Threshold | Measured | Status | Non-Relaxable |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `single_prompt_exact_success` | `single_prompt_exact_success_count` | `GE` | 1.0 | 1.0000 | ✓ PASS | YES |
| `assembly_validity_rate` | `assembly_validity_rate` | `EQ` | 1.0 | 1.0000 | ✓ PASS | YES |
| `runtime_trap_rate` | `runtime_trap_rate` | `LE` | 0.15 | 0.0000 | ✓ PASS | YES |
| `stage1_rolling_competence` | `stage1_rolling_competence` | `GE` | 0.85 | 0.8700 | ✓ PASS | YES |
| `stage1_minimum_coverage` | `stage1_minimum_coverage` | `GE` | 0.5 | 0.0000 | ✗ FAIL | YES |
| `stage1_competence_variance` | `stage1_competence_variance` | `LE` | 0.01 | 0.0004 | ✓ PASS | YES |
| `stage1_synthesis_pass_rate` | `stage1_synthesis_pass_rate` | `GE` | 0.8 | 0.3969 | ✗ FAIL | YES |
| `verified_task_retention_rate` | `verified_task_retention_rate` | `GE` | 0.95 | 1.0000 | ✓ PASS | YES |
| `extrapolation_pass_rate` | `extrapolation_pass_rate` | `EQ` | 1.0 | 1.0000 | ✓ PASS | YES |
| `mdl_ratio_max` | `mdl_ratio_max` | `LE` | 1.2 | 1.2000 | ✓ PASS | YES |
| `advantage_collapse_rate` | `advantage_collapse_rate` | `LE` | 0.05 | 0.8500 | ✗ FAIL | YES |
