# First model-sizing collection: HIP encoder parity blocks timing

The unchanged archive `009-model-sizing-20261025T170755Z.zip` verifies all159 listed
members plus manifest, exact719-entry source closure and every receipt/file hash
against producer6135575/source `ddc4d496011e12175a49453b25af45e434dce4ba4a5887b85f1405d90b635042`.
SHA256 `a54c7dc75bacb548f66cafaf5502bfe9f65648839f51a8d68600842bff371378`,
4,719,424 bytes. [Exact review](model-sizing-operator-20261025T170755Z.json) preserves
all stage/reference/source/qualification identities. Invocation1049.87s endsFAIL;
no learning/research/long-run authority is granted.

165 host tests and687 CPU tests plus30 subtests (717 JUnit cases, one CPU-suite
GPU-only deselection) pass. All three CPU references and actual parameter counts
pass with qualified memory/cleanup:8,118,294;31,903,510;104,496,662. The CPU fixtures
include actual shapes, baseline byte compatibility, finite updates/cache/sampling
and the generated-program native clock/admission path. This establishes CPU-side
construction/work only, not HIP sizing throughput.

HIP has19 passes and one failure, zero skips. The new sizing fixture fails on the
first m8 encoder memory comparison, shape2×3×256:340/1536 elements mismatch,
maximum absolute difference0.0003997684 versus retained atol0.00002, maximum
relative0.0530876 versus rtol0.0002. The stack enters
`torch._transformer_encoder_layer_fwd`, with a warning from memory-efficient
attention. The declared eager decoder flag did not constrain this independent
encoder dispatch. Backend dispatch is an observed policy gap; whether it fully
explains the numerical error still requires corrected actual-HIP evidence.

All15 performance trials remain BLOCKED/never started. There are no timing medians,
hardware-eligible profiles, speedups or model-size choices to accept. This is not
a failure of104M memory capacity, and no tolerance is loosened or failed element
excluded. The existing profile/row/seed/update/attempt/deadline roster remains.

Inspecting already-available producer CI once found15 completed successful checks
and one in-progress Podman build across Foundation40214803098 and Container/native
40214803102. No failed CI check was reported; the build was not awaited. This does
not erase the actual target HIP failure or close unrelated operations gates.

The correction must enforce and audit the same explicit FP32 mathematical attention
path in CPU references, HIP trials and the runtime fixture, prevent fused encoder/
MHA bypass, and restore process flags after exit/failure. It must retain current
numerical tolerances and fail closed on wrong/missing execution-policy evidence.
CPU/HIP tests and all15 slots then need a fresh current-source collection under the
same finite screen. New code is completed locally before issuing that one command.
009:M004/T110 remain pending; original counts77/19/15 and the learning hold remain.
