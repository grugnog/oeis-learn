# Contract: Multi-Limb Macro Instruction Set & Lowering Rules

**Profile**: `i256x4_v1`  
**Target Environment**: WebAssembly MVP with Multi-Value Return  
**Preamble Reference**: `contracts/multi-limb-preamble.wat`

---

## 1. Macro Instruction Specifications

| Macro Instruction | Operands (Stack / Immed) | Stack In | Stack Out | Lowering Transformation | Fuel Cost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `i256.add` | None | `(i64, i64, i64, i64, i64, i64, i64, i64)` | `(i64, i64, i64, i64)` | `call $i256_add` | 55 |
| `i256.sub` | None | `(i64, i64, i64, i64, i64, i64, i64, i64)` | `(i64, i64, i64, i64)` | `call $i256_sub` | 56 |
| `i256.mul_scalar` | None | `(i64, i64, i64, i64, i64)` | `(i64, i64, i64, i64)` | `call $i256_mul_scalar` | 271 |
| `i256.const <C>` | Immediate signed 64-bit int | `()` | `(i64, i64, i64, i64)` | Sign-extended quad-limb push: `i64.const <C> i64.const <sign> i64.const <sign> i64.const <sign>` where `sign = -1` if $C < 0$ else `0` | 4 |
| `i256.zero` | None | `()` | `(i64, i64, i64, i64)` | `i64.const 0 i64.const 0 i64.const 0 i64.const 0` | 4 |

---

## 2. Standardized Multi-Limb Register Conventions

To represent order-$k$ linear recurrences and holonomic sequences without memory allocations, local variables are standardized across four 64-bit limbs:

- **State Register A ($a(n-2)$)**:
  `(local $a0 i64) (local $a1 i64) (local $a2 i64) (local $a3 i64)`
- **State Register B ($a(n-1)$)**:
  `(local $b0 i64) (local $b1 i64) (local $b2 i64) (local $b3 i64)`
- **Transition Buffer T ($\text{temp}$)**:
  `(local $t0 i64) (local $t1 i64) (local $t2 i64) (local $t3 i64)`
- **Loop Counter I**:
  `(local $i i32)`
- **Evaluation Parameter**:
  `(param $n i32)`
- **Export Signature**:
  `(func (export "compute") (param $n i32) (result i64 i64 i64 i64))`

---

## 3. Macro Lowering Protocol

Before invoking `oeis_wasm_evaluator` or `wasmtime`, candidate programs undergo lowering:

1. **Preamble Inlining**: The verified static math preamble (`$mul64_wide`, `$i256_add`, `$i256_sub`, `$i256_mul_scalar`) is prepended to the module body.
2. **Opcode Substitution**:
   - Every `i256.add` is rewritten to `call $i256_add`.
   - Every `i256.sub` is rewritten to `call $i256_sub`.
   - Every `i256.mul_scalar` is rewritten to `call $i256_mul_scalar`.
   - Every `i256.const <C>` is expanded to the 4-limb literal push.
3. **Validation & Verification**: The lowered WAT must parse with `wat::parse_str` and pass WASM type checking with 0 linear memories.

---

## 4. Canonical Fibonacci Demonstration (29 Macro Tokens)

```wat
(module
  (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
    (local $a0 i64) (local $a1 i64) (local $a2 i64) (local $a3 i64)
    (local $b0 i64) (local $b1 i64) (local $b2 i64) (local $b3 i64)
    (local $t0 i64) (local $t1 i64) (local $t2 i64) (local $t3 i64)
    (local $i i32)
    i256.zero local.set $a3 local.set $a2 local.set $a1 local.set $a0
    i256.const 1 local.set $b3 local.set $b2 local.set $b1 local.set $b0
    (block $exit
      (loop $loop
        local.get $i local.get $n i32.ge_s br_if $exit
        local.get $b0 local.get $b1 local.get $b2 local.get $b3
        local.get $a0 local.get $a1 local.get $a2 local.get $a3
        i256.add
        local.set $t3 local.set $t2 local.set $t1 local.set $t0
        local.get $b0 local.set $a0 local.get $b1 local.set $a1
        local.get $b2 local.set $a2 local.get $b3 local.set $a3
        local.get $t0 local.set $b0 local.get $t1 local.set $b1
        local.get $t2 local.set $b2 local.get $t3 local.set $b3
        local.get $i i32.const 1 i32.add local.set $i
        br $loop
      )
    )
    local.get $a0 local.get $a1 local.get $a2 local.get $a3
  )
)
```
