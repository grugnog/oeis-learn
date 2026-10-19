# Native fitting milestone (T026–T036)

Counterexamples were added before implementation; Python admission/capacity tests
failed collection on missing modules. Implemented conservative whole-program
control taint, distinct literal abstraction, bounded symbolic DAG with every
intermediate guard, fraction-free elimination/exact rational back substitution,
worker-local QF_LIA/QF_NIA, bounded model enumeration, typed substitution,
production checks, independent final admission, scoped re-derivation/replay and
aggregate reservation/LRU policies. No Python fitting or trial callbacks.

The common binding/core qualification driver is now built beside the guest and
runs execute/fit/replay diagnostics with external test deadline/RSS/output caps.
Production protocol stubs remain disabled. The new native build/tests have not
run locally: neither Rust nor Docker/Podman is available. New CI is not monitored.
T036 stays pending until the actual native tests pass. Retained signed-256
execution is supported; unsupported retained fitting translations are explicit.

Local milestone: 22 passed, 4 native-only skips across admission, capacity and
independent-reference tests; Python F/E9 and git whitespace checks passed.
The reference parser's optional parameter syntax is used only to independently
verify substitution; normal execution still rejects unresolved parameters.
