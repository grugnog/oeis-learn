# Phase 4 migration and retirement (T081–T088)

Entry: merged Phase 3 PR #56 at `66d13fdb93bc7d107c0f1ac8fe90cf030dea47e9`.
T080 passed the CPU/native integration gate; original evidence is preserved.

Work in progress. Initial source audit deliberately fails with 52 retired
module/import/API/script/config/dependency violations. Tests cover renamed and
dynamic imports, production source reachability and a clean installed image.
No retirement gate is claimed complete by this initial commit.

Migration will preserve independent arithmetic/certificates, supported signed-256
controls and read-only historical status. Python fitting implementations and
solver-only assets will be removed after callers and counterexamples migrate.
New CI is left unmonitored. Native image execution and HIP are not claimed here.

## Migration checkpoint

Supported fitting callers now route through `synthesis.fitting` to the single
native client and independent admission. New replay records have an explicit
native kind; historical grounding records return `historical_unreplayed`.
The Python loop-certificate verifier retains its restricted obligations but
serializes them to a fixed-budget pinned Rust/Z3 checker with no model interface.
Retired scalar/wrapping/bitwise/recurrence skeletons fail explicitly.

Eight implementation modules, three run008/009/010 launch scripts and their
three exclusive configs are removed. Obsolete method switches in retained
diagnostic recipes are replaced by explicit native profile settings. Python
z3-solver is removed without changing the other locked packages. SymPy remains
for independently supported certificate/discovery consumers.

Local checkpoint: source audit passes; 54 targeted tests pass, 23 packaged-native
or installed-image cases explicitly skip here. The affected caller/model routes
are still under regression review. Clean installed native execution remains
pending CI; it is not claimed from source checks.
