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

## CI compilation repair after the completion audit

The completed [container run 39184890109](https://github.com/grugnog/oeis-learn/actions/runs/39184890109)
on `0687b37b1492205977246e3091a63e0bd654497b` failed while compiling the core
library. Rust reported E0282 at `symbolic.rs`: the collection type in the affine
operator match was ambiguous when indexed inside the subsequent limit-check
closure. The local binding is now explicitly `Option<Vec<IBig>>`, matching the
existing `Big.affine` field and `Builder::node` argument. Arithmetic, guards,
resource limits and solver policy are unchanged.

Native formatting and the pinned dependency-cache build passed in that run;
the downstream acceptance jobs were skipped because the image build failed.
The companion foundation-review run39184890078 passed. This repair has been
checked against the compiler diagnostic and with `git diff --check`; compilation
cannot be repeated locally because the workspace has neither the pinned Rust
container toolchain nor Docker/Podman. Existing Rust fitting fixtures exercise
the affected operator paths when CI rebuilds. T025/T036/T058 remain open, and the
replacement CI run is deliberately not monitored or awaited. Earlier local
source hashes remain historical snapshots, not qualification of this repair.

## Cross-UID artifact upload repair

The completed [container run39185860360](https://github.com/grugnog/oeis-learn/actions/runs/39185860360)
on `9532d7eaf4d2719289e95bafc3b35f0ae2f0a39e` built successfully. The contracts
partition ran152 tests with zero failures/errors/skips and reported PASS; only
artifact upload failed with EACCES while scanning `/evidence/pytest`. Pytest had
created that directory mode0700 as container UID12345, while upload ran under the
host runner UID. The root evidence mount's mode777 did not grant access through
the private child directory. The native partition, both portable-engine jobs,
Podman build, native qualification and foundation-review checks passed.
The image was
`ghcr.io/grugnog/oeis-learn@sha256:2c905ac43f6a63e40388ccdaa89b653661e216604ccfca4b63306e6b4f56541d`.

The helper now creates pytest scratch in a temporary directory outside the public
reports and removes it after both successful and unsuccessful test runs. XML,
summary and deliberate GPU evidence still go to the requested output directory.
The arbitrary-UID CI command also makes generated report descendants readable
on EXIT, including failure exits; it skips the runner-owned mount root so UID12345
does not attempt a forbidden chmod on a directory it does not own.

Two regression cases simulate success and failure with private0700 test scratch,
verify its removal, and require the public output to contain only summary/XML.
Together with the cache/image regressions,26 tests passed locally. Python F/E9,
workflow YAML, bigint step shell syntax and whitespace checks passed. This is an
artifact publication fix; native/HIP semantics and task gate markers are unchanged.
The new CI run is left unmonitored, per the user's instructions.
