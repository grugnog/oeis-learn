# Implemented bigint-body/v1 syntax

Phase 2 US1 implementation contract; native qualification is still pending at
T025. These details refine the typed-language contract, without enabling the
production service. No source repair, arbitrary WAT or Python fitting is added.

A source is one ASCII s-expression `(program DECLARATIONS STATEMENTS (return EXPR))`.
Whitespace has no meaning. Comments, quoted strings and trailing source are
rejected. Local declarations precede every statement. The final return must be
bigint; nested/early user returns are absent. The existing source, node and
nesting caps apply. Node accounting counts every list and atom, including
operator/declaration atoms, before type checking.

| Form | Type and behavior |
| --- | --- |
| `(local $b0 bigint)` | Names `$b0`…`$b15`; initialize to exact zero |
| `(local $c0 control)` | Names `$c0`…`$c7`; initialize to control zero |
| `(local $q0 bool)` | Names `$q0`…`$q7`; initialize to false |
| `$index` | Immutable actual signed input index, bigint |
| `(int DECIMAL)` | Canonical exact bigint literal within literal bound |
| `(param.get p0)` | Bigint parameter; up to eight distinct IDs, rejected in concrete execution |
| `(get LOCAL)` | Declared local's type |
| `(add A B)`, `(sub A B)`, `(mul A B)` | Two bigints → checked bigint, including full multiplication |
| `(eq A B)`, `(lt A B)`, `(le A B)` | Two bigints → bool |
| `(control N)` | Canonical nonnegative control literal within control bound |
| `(to_control A)` | Bigint → nonnegative bounded control; failure never truncates |
| `(to_bigint C)` | Control → exact bigint, subject to intermediate bound |
| `(increment C)` | Checked control increment, fails at control maximum |
| `(control.lt A B)` | Two controls → bool |
| `(true)`, `(false)` | Boolean literals |
| `(select CONDITION A B)` | Bool and two same-typed operands; evaluates condition, A, then B eagerly |
| `(set LOCAL EXPR)` | Same-type assignment; index and active loop counters are immutable |
| `(if CONDITION (then STATEMENTS) (else STATEMENTS))` | Lazy branch; both branches type-checked, empty branch allowed |
| `(loop COUNTER COUNT STATEMENTS)` | COUNT evaluated once as control; at least one statement; counter visits 0…COUNT−1 and finishes at COUNT |

A loop sets its counter to zero even for zero iterations. Its counter cannot be
assigned by the body or reused by a nested loop. Declared per-loop and aggregate
iteration caps fail with `loop_limit`; a control conversion outside its bound
fails with `control_range`. Body state and the complete arena are fresh per term.

The production parser records original source and parameter IDs, renames IDs by
first occurrence, and emits a canonical typed source and AST identity. Repeated
IDs remain shared; equal literals remain distinct occurrences. Codec targets use
that same first-occurrence naming rule. Runtime substitution may exceed the
1,024 generated-token cap but must still satisfy source/AST/literal caps. New
codec checkpoints have a distinct content identity and cannot load old weights.

The native signed-256 adapter accepts only bare `i256.const`, `i256.zero`,
`i256.add` and `i256.sub` bodies. It reuses the original checked preamble and strict
native evaluator. Unsupported scalar/wrapping/bitwise/control/streaming forms
return explicit unsupported results. This narrow subset does not claim support
for every old checkpoint candidate, and never executes those candidates as bigint.

Arena accounting charges each immutable slot 32 bytes plus at least one magnitude
byte. Linear-memory limits additionally cover allocator capacity and temporaries.
A multiplication is computed within that memory cap and checked at its exact
result size; an estimated maximum size cannot reject a valid borderline product.
The standalone guest has no native imports. Only the seven approved arithmetic
functions are linked into compiler-generated candidates; host loader, status,
serialization, memory and reset privileges are never linked. No reset/free API
exists. Controlled compiler exits use reserved negative return values for control
and loop limits; source programs cannot construct handles or those exits.

The chunk input profile uses `log1p` of true byte lengths and true final-chunk
length divided by eight. Padding byte embeddings are masked before projection;
padding chunks are masked before mean pooling. Zero uses one zero magnitude byte
and sign zero. The second-difference stream preserves the historical first-two
zero convention. Log approximations retain 53 leading bits with absolute error
below 1e-12 at supported widths; exact values, residues and differences never
undergo float conversion. Growing positional buffers are nonpersistent and have
no width-dependent trainable parameters.
