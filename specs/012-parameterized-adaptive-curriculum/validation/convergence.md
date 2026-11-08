# T075–T078 conformance remediation

Recorded 2026-11-05. These are implementation repairs before the optional research
campaign. T073 remains deferred. No new research, hardware run, model superiority
or current-source qualification is claimed.

## Behavior and evidence boundaries

T075 makes a completed PAC charged allowance successful even after unsaved crash
work was discarded. `RECOVERED_WITH_DISCARDED_EXPOSURE` remains the reason; useful,
discarded, reserved and unspent exposure remain separate. Genuine partial/paused
runs and the historical non-PAC disposition stay unchanged. The independent CPU
regression crashes after a real optimizer call, recovers old state, completes the
charged budget and exercises canonical result transport through all wrapper stages
and a reuse-only resume.

T076 versions installed observers as v2. A separate `pac-retention-history/v1`
contains at most 64 bins: initial singleton, compacted oldest interior bins and
recent panels. Each bin preserves first/last panel coordinates, exact panel count
and per-family loss sums/token counts, completed/censored losses and semantic
attempt/completion/success counts. Loss sums use exact rational accumulation of
recorded decimal values. Compacted interiors are not an interpolated trajectory.
The initial and final measurements and whole-run aggregates remain reconstructible.
The separate recent-event tail still has at most 64 entries. Legacy v1 snapshots
restore with an explicit count of missing historical panels; no old measurement
is synthesized. Wall time is measured across resumes but excluded from exact
replay equality, alongside the existing elapsed-observation exclusion. Model,
optimizer, RNG, policy and all deterministic observer measurements remain compared.

T077 adds `pac-analysis/v2` and sealed `pac-research-inputs/v1`. Public inputs
contain the frozen experiment/control/profiles, admitted denominator, all seven
control outcome maps, original producer identities, measured per-arm exposure,
family/channel allocation, compact retention history and stage/learner costs.
When fresh A is selected, retained A-reference outcomes remain separately labeled
and cannot be overwritten by the new A learner. Optional D keeps its distinct
allocation and review-substitution comparisons.
The installed stdlib reconciler independently reconstructs every metric,
comparison, count adapter, exposure/cost total and retention aggregate. Review and
semantic timing components unavailable at the producer remain null; observation
wall time is available as a combined measurement. Components overlap and are not
summed into a compute claim. No tensors, optimizer/RNG state, protected teacher
programs/terms, raw logs or original host paths enter these inputs. Existing
explicitly generated candidate cases remain a separate bounded projection.

T078 adds a public `pac-research-authority/v1` projection taken from the original
accepted study/runtime references, with native admission, source/image/plan,
producer/binding and input identities. Offline verification checks the actual
analysis bytes against recomputation and cross-binds scope, control, arms, seed,
roster, stage costs and analysis producer. Acceptance-only evidence always has
scientific status UNMEASURED. Missing research evidence is INCOMPLETE. Exploratory
interpretation requires the reconstructed >=65/68 certified completion gate in
each compared arm; a labeled A reference does not become a compared control.
Software disposition and scientific eligibility are checked separately. Honest
negative/incomplete exports remain supported.

Historical acceptance evidence and source-bound receipts are unchanged. Legacy
analysis v1 remains descriptive data; it does not acquire a reconstructed frozen
authority or scientific eligibility by being repacked. The original acceptance
records retain their UNMEASURED meaning. Newly installed code requires a fresh
image/source scope and affected qualification before a new live campaign; prior
HIP evidence is not relabeled as proof of these changes.

## Local validation

Independent counterexamples were run before their fixes. The long-run observer
fixture covers 32,768 updates, 129 panels, censoring, tail eviction, compaction and
midpoint restore. Producer-shaped campaign fixtures exercise distinct raw/fitted
outcomes, unequal observation/stage costs, missing measurements and actual export.
Scientific counterexamples cover acceptance-only claims, missing payloads,
resealed roster/control/source/seed/scope/stage changes, altered aggregates and
boolean/integer substitutions. Both 64 and 65 completions round-trip final archives
under a dependency-free `python -S` verifier.

Reproduce the affected local selection from the repository root, using an existing
locked CPU development environment with torch (this is not HIP qualification):

```sh
PYTHONPATH=src python3 -m pytest -q \
  tests/unit/test_generalization*.py \
  tests/integration/test_generalization*.py \
  tests/integration/test_pac_structural_publication.py \
  specs/012-parameterized-adaptive-curriculum/validation \
  -m 'not gpu and not docker'
```

Final validation: the broad selection passed **598 tests**, with three actual-HIP
tests explicitly deselected. The initial run exposed an older publication fixture
missing its protocol version; the fixture now declares the historical v1 scope
without weakening production validation. A final reporting refinement separately
preserves retained A-reference data in a fresh-A/optional-D campaign; its failing
counterexample passed after the fix. The final affected reporting/export/policy
and spec-012 selection passed **93 tests**, including the dependency-free archive
round trips. Whitespace, compilation, prerequisites and the read-only 16/16 quality
checklist pass. No before/after implementation hooks are registered. CI was not
waited for and no live research or hardware qualification was launched.
