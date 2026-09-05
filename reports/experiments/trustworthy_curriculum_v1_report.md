# Experiment Report: trustworthy_curriculum_v1

- **Experiment Type**: `TRAINING_ABLATION`
- **Status**: `COMPLETE`
- **Tested Seeds**: `[42, 137, 2026]`
- **Total Variants**: `2`
- **Completed Outomes**: `6`

## Variants

| Variant ID | Changed Factor | Factor Value | Active Rollouts | Replay Budget |
| :--- | :--- | :--- | :--- | :--- |
| `fixed_uniform` | `TASK_ALLOCATION` | `FIXED_UNIFORM` | 32 | 2 |
| `adaptive_symple` | `TASK_ALLOCATION` | `ADAPTIVE_SYMPLE` | 32 | 2 |

## Outcomes Matrix

| Variant ID | Seed | Status | Wall Hours | Pass Rate | Extrap Count |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `fixed_uniform` | `42` | `COMPLETE` | 0.0000 | 12.0% | 4 |
| `fixed_uniform` | `137` | `COMPLETE` | 0.0000 | 12.0% | 4 |
| `fixed_uniform` | `2026` | `COMPLETE` | 0.0000 | 12.0% | 4 |
| `adaptive_symple` | `42` | `COMPLETE` | 0.0000 | 25.0% | 4 |
| `adaptive_symple` | `137` | `COMPLETE` | 0.0000 | 25.0% | 4 |
| `adaptive_symple` | `2026` | `COMPLETE` | 0.0000 | 25.0% | 4 |
