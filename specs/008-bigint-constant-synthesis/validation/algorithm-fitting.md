# Native fitting implementation and pending gate (T026–T036)

All fitting/admission/service-policy implementation is present. Counterexamples
were written first, initially failing Python collection on missing modules.
Native Rust fixtures cover exact elimination, integer feasibility, semantic
control taint and preserved nonlinear intermediate guards. The common binding/core
driver now routes runtime execution, fitting and scoped replay through the same
core APIs; no Python solver or trial callback participates.

Implemented: conservative whole-program dependence, distinct eligible literal
occurrences, bounded guarded symbolic DAGs, deterministic fraction-free exact
elimination/back substitution, worker-local QF_LIA/QF_NIA, coefficient scopes
16/64/256/1024/4096 under one native deadline, at most8 discovered assignments,
deterministic discovered-set ranking, typed final substitution, production
visible checks and independent source-parsed admission. UNKNOWN, timeout,
production exhaustion and incomplete uniqueness are not global UNSAT. Scope
replay derives constraints from source/profile/prompt rather than executing
attached solver text. Native and Python caches use bounded reservations and
identity-aware reuse; provisional results never become verified cache entries.

Retained fitting supports the explicit checked signed-256 const/zero/add/sub
subset. Its independent reference checks every intermediate, including cancelled
overflow. Unsupported legacy operations fail explicitly; they do not acquire
bigint semantics. Runtime identity binds both the native executable and guest
bytes, including the retained backend. Reference identity binds the independent
parser, interpreter, resource contract and admission code.

See [local evidence](algorithm-learning-local.json) for the final CPU test counts,
source hashes and skips. Earlier implementation milestone:22 passed,4 native
skips. The current native-only tests include large negative affine fitting,
rank-deficient 2p+3q=1, square=4, re-derived/tampered UNSAT, control taint,
cancelled overflow and checked retained fitting. CPU admission injects a wrong
16-for-4 result, missing coverage, changed identities, false public status and
reference exhaustion; none can be admitted.

**T036 stays unchecked** until the new locked Rust tests and packaged driver
execute successfully. Docker/Podman and Rust are unavailable locally, and new PR
CI is deliberately not monitored. The qualification utility rejects all skips
inside the image. A passing run establishes only the recorded finite/resource
scope; it is never proof of an integer sequence or mathematical novelty.
