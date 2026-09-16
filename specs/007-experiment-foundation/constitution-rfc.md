# RFC: Constitution 2.0.0 for trustworthy experiments

**Date**: 2026-09-16. **Status**: Proposed with this feature branch, not ratified on main. **Scope**: Separate enduring scientific/correctness obligations from architecture choices being investigated.

The companion proposed revision is [.specify/memory/constitution.md](../../.specify/memory/constitution.md). Its baseline is version 1.0.0 at `aa2c8c4579dd74fdd8114ec9456960a98fcf500e`. User instructions authorize the new machine, fresh weights, generic bootstrap and native LODA evaluation; they do not establish hardware feasibility or maintainer consensus on every default.

## Rationale

Version 1.0.0 makes one encoder, WAT, named curricula, EGCA-GRPO, VICReg, Rust/Rayon, a 10,000-fuel ceiling and the old laptop mandatory. Those constraints prevent the requested comparison and can make correct programs fail fixed resource gates. Exact authoritative integers do not imply an injective continuous encoder; existing FP32 streams are not injective. Finite extrapolation and MDL cannot eliminate every memorizing program or prove an identity for every index.

The proposal keeps exact data, constrained languages, bounded execution, provenance, isolated evaluation and honest evidence mandatory. Each experiment freezes and validates its architecture, precision, numerical and resource choices. 007 initially retains FP32 and the existing small encoder.

## Migration and conflicts

| Old obligation | Proposed rule and rationale | Effect |
| --- | --- | --- |
| Exact tri-stream FiLM; FP32 forever | Preserve exact source integers; declare/test neural features and precision | Current encoder reused; no injectivity claim. Specs 001/003 remain historical. |
| WAT only; particular grammar engines/latency | Freeze supported language/grammar; reject invalid candidates | WAT in 007; native LODA in 008; no corpus calls. |
| Rust/Rayon only; 10k fuel; 16 MiB | Enforce declared finite profiles and worker recovery; measure throughput | Isolated Python/Wasmtime permitted; Rust optional after parity. |
| Laptop graduation before workstation | Validate actual declared hardware before budgeted runs | Ryzen requires real GPU updates, not inference from laptop results. |
| Named curricula and 20+100 | Strict generic bootstrap and explicit frozen horizons/groups | 007 uses 20+80; 005/006 evidence is not automatically migrated. |
| EGCA-GRPO and VICReg mandatory | Objectives/representations are experimental choices | Verified SFT first; RL/discovery work deferred. |
| Finite/PSLQ/SymPy acceptance | Scope empirical evidence; require sound replayable certificates for proof | Fail closed now; full proof repair later. |

Specs 002/004 teachers, placeholder solvers, optimizer and replay remain legacy interfaces, unreachable from the strict profile. Reuse their code only through the new admission contract. Specs 005/006 offer useful data/checkpoint vocabulary; specification existence does not establish implementation correctness. Mark historical artifacts `legacy_unverified`, never manufacture missing evidence, and never turn weights-only files into resumable checkpoints by adding a label.

## Adoption and feasibility

The proposed revision is **2.0.0** because it redefines mandatory principles. Planning analysis evaluates this explicit proposal and reports adoption separately. Merge/adoption must accept this RFC and the compatibility break; no approval or benchmark is claimed here.

The old amendment procedure requires feasibility evidence. A bounded standalone Docker/device preflight is the first implementation task and can run before adoption. Its report must be attached before adopting this revision for production experiments. Remaining implementation tasks are gated on adoption; all feature acceptance gates precede the 008 multi-day comparison. GPU failure produces evidence and a specific host-change request, never an automatic NixOS modification.

## Compatibility and rollback

Use new profile/schema IDs and isolated run directories. Legacy commands remain diagnostic and cannot emit strict evidence. Old datasets/checkpoints remain readable for inspection, with no automatic strict import. Rollback returns to the old commit and its run identities; it never opens new-profile state under old schemas. This proposal does not rewrite completed runs or promise compatible weights.
