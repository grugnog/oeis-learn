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
