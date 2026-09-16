# OEIS Learn Constitution

**Amendment status**: Proposed version 2.0.0 on the 007 feature branch. Version 1.0.0 remains the ratified main-branch baseline until maintainers adopt [RFC 007](../../specs/007-experiment-foundation/constitution-rfc.md) with its required feasibility evidence. This file records the proposed rules explicitly so planning can be checked against them; it does not assert approval or completed benchmarks.

## Core Principles

### I. Exact authoritative data and declared numerical semantics

Sequence source values, program constants and verification results MUST preserve exact integers and source indices. Every executable language profile MUST specify signedness, width, exceptional operations, intermediate limits and return representation. Logical overflow MUST be classified according to that profile, independently of intentional limb carry arithmetic. Training features may be lossy neural representations, but MUST NOT replace authoritative exact values or be described as mathematically injective without proof. Precision and architecture MUST be frozen in each run contract and tested for finite forward/backward behavior.

### II. Supported program languages and independent acceptance

An experiment MUST declare its grammar, available primitives, entry point and complete token/operand encoding. Constrained generation MUST target that declared subset; invalid or unsupported programs MUST be rejected before acceptance. Any transformation or constant grounding MUST be followed by final-source validation and exact execution checks. Independent conformance expectations MUST NOT be computed by the production arithmetic/lowering being tested. A language, grammar library, runtime or helper primitive is an experimental choice and a disclosed prior, not a universal requirement.

### III. Bounded execution and recoverable operation

Generated programs MUST execute under finite memory, per-call and aggregate work limits plus an external wall-time deadline. Host process isolation and worker recovery MUST contain hangs and runtime crashes. Queue, cache, checkpoint, log and disk budgets MUST be explicit and enforced. The chosen runtime may use isolated processes or qualified native workers; backend changes MUST preserve the declared semantics or create a new profile. Unavailable metrics MUST be marked unavailable, never invented.

### IV. Workstation feasibility and reproducible experiments

The run contract MUST identify hardware, software, effective configuration, seeds and budget accounting. Accelerator readiness MUST demonstrate real forward/backward parameter updates on the requested device; silent CPU fallback is prohibited. On the user's NixOS workstation, native/toolchain dependencies beyond a simple Python virtual environment MUST run in Docker. Host driver changes require a separate explicit action, not an automatic setup step. Complete checkpoints MUST restore active learning, RNG, data-order and budget state at update boundaries; weights-only artifacts are not resumable runs.

### V. Strict provenance and isolated generalization evaluation

The primary strict track MUST start from random weights and use mechanically generated generic-program demonstrations. Imported OEIS/LODA programs, named sequence-family teachers, metadata-guided scaffolds and unaudited legacy replay are excluded. An assisted track requires an explicit separate decision and distinct provenance. Exact source terms may support frozen evaluation, but hidden evaluation values MUST be inaccessible to learning, candidate generation and selection. Splits MUST keep detectable duplicate/prefix/shift groups together and report limitations. The visible prefix, verification horizon, attempt/time budgets and scoring denominator MUST be frozen before the run. Finite continuation is evidence of finite agreement, not proof of a unique generating rule.

### VI. Honest discovery and scoped proof

Reports MUST distinguish finite matches, conjectures, bounded verification and proved statements. Proof promotion MUST require a sound, replayable certificate with explicit assumptions and domain; unqualified legacy proof labels cannot satisfy this rule. Novelty requires separate comparison with existing knowledge and MUST NOT follow merely from model generation. Program archives MUST retain provenance, numerical/resource profiles and verification scope so later analysis can assess the evidence independently.

## Hardware Constraints & Operational Division of Labor

The initial new experimental target is AMD Ryzen AI Max+ 395, 128 GB shared RAM and a 2 TB SSD running NixOS. CPU workers perform bounded execution/data checks and a single learner performs neural updates. Actual worker counts, memory partitions, precision and token budgets belong to versioned experiment profiles and MUST be validated on this hardware. The original four-core/4 GB GPU laptop remains a historical and optional diagnostic target, not a mandatory graduation gate for this machine. Docker does not provide missing host kernel/GPU support.

## Development Workflow, MVP Acceptance Gates & Quality Standards

1. Specifications MUST state the research decision, permitted information, numeric semantics, measured outcomes and explicit non-goals. Experiments MUST freeze hypotheses, baselines, manifests, seeds, budgets, selection/stopping rules and interpretation of inconclusive results before execution.
2. Arithmetic, admission, leakage, resume and containment changes MUST have meaningful regression/contract tests before implementation is considered complete. Known counterexamples and independent differential checks are mandatory for acceptance paths.
3. A qualifying model evaluation MUST load and identify the actual checkpoint. Canonical programs are separately labeled runtime diagnostics.
4. Long runs MUST wait for the exact acceptance, provenance, checkpoint, containment and real-device gates of their foundation feature. Documentation checks cannot satisfy runtime gates.
5. Optional unsafe or unqualified subsystems MUST remain disabled until their own validation succeeds. No architecture or algorithm sweep is required without a concrete unresolved research decision.

## Governance

The ratified constitution governs implementation. On a proposal branch, planning MUST explicitly identify the proposed version and outstanding adoption/feasibility gates; successful document validation MUST NOT be described as ratification or hardware readiness. User instructions govern the requested work; new technical defaults MUST be distinguished from user-confirmed decisions.

Amendments require a written RFC with rationale, evidence appropriate to the claims, explicit maintainer adoption, and migration/rollback rules. A bounded preflight may gather feasibility evidence before adoption; it does not authorize a multi-day experiment or assert success. New claims about hardware performance require actual measurements on that hardware. Historical artifacts MUST retain their original status.

Versions use Semantic Versioning: major for removed/redefined mandatory principles, minor for compatible additions, patch for clarifications. Contributors MUST read this document and the active Spec Kit templates during specification, design, tasks and review. Templates and upstream skill instructions MUST NOT be altered merely to hide a conflict or make a check pass.

**Version**: 2.0.0 (proposed) | **Original Ratified**: 2026-08-30 | **Amendment Proposed**: 2026-09-16
