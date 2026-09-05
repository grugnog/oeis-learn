# Experiment Report: trustworthy_inference_v1

- **Experiment Type**: `INFERENCE_ABLATION`
- **Status**: `COMPLETE`
- **Tested Seeds**: `[42, 137, 2026]`
- **Total Variants**: `4`
- **Completed Outomes**: `12`

## Variants

| Variant ID | Changed Factor | Factor Value | Active Rollouts | Replay Budget |
| :--- | :--- | :--- | :--- | :--- |
| `unresolved_b1` | `CONSTANT_RESOLUTION` | `False` | 0 | 0 |
| `resolved_b1` | `CONSTANT_RESOLUTION` | `True` | 0 | 0 |
| `resolved_b8` | `CANDIDATE_BUDGET` | `8` | 0 | 0 |
| `resolved_b16` | `CANDIDATE_BUDGET` | `16` | 0 | 0 |

## Outcomes Matrix

| Variant ID | Seed | Status | Wall Hours | Pass Rate | Extrap Count |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `unresolved_b1` | `42` | `COMPLETE` | 0.0000 | 25.0% | 10 |
| `unresolved_b1` | `137` | `COMPLETE` | 0.0000 | 25.0% | 10 |
| `unresolved_b1` | `2026` | `COMPLETE` | 0.0000 | 25.0% | 10 |
| `resolved_b1` | `42` | `COMPLETE` | 0.0000 | 25.0% | 10 |
| `resolved_b1` | `137` | `COMPLETE` | 0.0000 | 25.0% | 10 |
| `resolved_b1` | `2026` | `COMPLETE` | 0.0000 | 25.0% | 10 |
| `resolved_b8` | `42` | `COMPLETE` | 0.0000 | 25.0% | 10 |
| `resolved_b8` | `137` | `COMPLETE` | 0.0000 | 25.0% | 10 |
| `resolved_b8` | `2026` | `COMPLETE` | 0.0000 | 25.0% | 10 |
| `resolved_b16` | `42` | `COMPLETE` | 0.0000 | 25.0% | 10 |
| `resolved_b16` | `137` | `COMPLETE` | 0.0000 | 25.0% | 10 |
| `resolved_b16` | `2026` | `COMPLETE` | 0.0000 | 25.0% | 10 |
