;; Static Verified Multi-Limb Arithmetic Preamble (i256x4_v1)
;; Zero-linear-memory, pure value-stack 256-bit signed arithmetic.
;; Lowering target for macro instructions: i256.add, i256.sub, i256.mul_scalar.

(module
  ;; --------------------------------------------------------------------------
  ;; Helper: $mul64_wide
  ;; Multiplies two 64-bit unsigned integers: u * v -> (lo: i64, hi: i64)
  ;; Evaluates via 32-bit limb decomposition:
  ;; u = u1*2^32 + u0,  v = v1*2^32 + v0
  ;; Fuel cost: 60 instructions
  ;; --------------------------------------------------------------------------
  (func $mul64_wide (param $u i64) (param $v i64) (result i64 i64)
    (local $u0 i64) (local $u1 i64)
    (local $v0 i64) (local $v1 i64)
    (local $w0 i64) (local $k i64)
    (local $w1 i64) (local $w2 i64)
    (local $lo i64) (local $hi i64)

    ;; Decompose u and v into 32-bit limbs
    local.get $u i64.const 4294967295 i64.and local.set $u0
    local.get $u i64.const 32 i64.shr_u local.set $u1
    local.get $v i64.const 4294967295 i64.and local.set $v0
    local.get $v i64.const 32 i64.shr_u local.set $v1

    ;; Partial product w0 = u0 * v0
    local.get $u0 local.get $v0 i64.mul local.set $w0
    local.get $w0 i64.const 32 i64.shr_u local.set $k

    ;; Cross terms
    local.get $u1 local.get $v0 i64.mul local.get $k i64.add local.set $w1
    local.get $w1 i64.const 4294967295 i64.and local.set $k
    local.get $w1 i64.const 32 i64.shr_u local.set $w1

    local.get $u0 local.get $v1 i64.mul local.get $k i64.add local.set $w2
    local.get $w2 i64.const 32 i64.shr_u local.set $k

    ;; High 64 bits = u1 * v1 + w1 + k
    local.get $u1 local.get $v1 i64.mul
    local.get $w1 i64.add
    local.get $k i64.add
    local.set $hi

    ;; Low 64 bits = (w2 << 32) | (w0 & 0xFFFFFFFF)
    local.get $w2 i64.const 32 i64.shl
    local.get $w0 i64.const 4294967295 i64.and
    i64.or
    local.set $lo

    local.get $lo
    local.get $hi
  )

  ;; --------------------------------------------------------------------------
  ;; $i256_add
  ;; Value-stack 256-bit addition:
  ;; Stack In:  (a0, a1, a2, a3, b0, b1, b2, b3)  [all i64, little-endian]
  ;; Stack Out: (s0, s1, s2, s3)                  [all i64, little-endian]
  ;; Fuel cost: 52 instructions, 55 fuel units
  ;; --------------------------------------------------------------------------
  (func $i256_add
    (param $a0 i64) (param $a1 i64) (param $a2 i64) (param $a3 i64)
    (param $b0 i64) (param $b1 i64) (param $b2 i64) (param $b3 i64)
    (result i64 i64 i64 i64)
    (local $s0 i64) (local $c0 i64)
    (local $s1 i64) (local $c1 i64)
    (local $s2 i64) (local $c2 i64)
    (local $s3 i64)

    ;; Limb 0: s0 = a0 + b0, carry = (s0 < b0)
    local.get $a0 local.get $b0 i64.add local.set $s0
    local.get $s0 local.get $b0 i64.lt_u i64.extend_i32_u local.set $c0

    ;; Limb 1: temp = a1 + b1, s1 = temp + c0
    ;; carry = (temp < b1) + (s1 < c0)
    local.get $a1 local.get $b1 i64.add
    local.get $c0 i64.add
    local.set $s1
    local.get $a1 local.get $b1 i64.add local.get $b1 i64.lt_u i64.extend_i32_u
    local.get $s1 local.get $c0 i64.lt_u i64.extend_i32_u
    i64.add local.set $c1

    ;; Limb 2: temp = a2 + b2, s2 = temp + c1
    local.get $a2 local.get $b2 i64.add
    local.get $c1 i64.add
    local.set $s2
    local.get $a2 local.get $b2 i64.add local.get $b2 i64.lt_u i64.extend_i32_u
    local.get $s2 local.get $c1 i64.lt_u i64.extend_i32_u
    i64.add local.set $c2

    ;; Limb 3: s3 = a3 + b3 + c2 (overflow wraps naturally in 2's complement)
    local.get $a3 local.get $b3 i64.add local.get $c2 i64.add local.set $s3

    local.get $s0
    local.get $s1
    local.get $s2
    local.get $s3
  )

  ;; --------------------------------------------------------------------------
  ;; $i256_sub
  ;; Value-stack 256-bit subtraction: a - b
  ;; Stack In:  (a0, a1, a2, a3, b0, b1, b2, b3)  [all i64, little-endian]
  ;; Stack Out: (d0, d1, d2, d3)                  [all i64, little-endian]
  ;; Fuel cost: 53 instructions, 56 fuel units
  ;; --------------------------------------------------------------------------
  (func $i256_sub
    (param $a0 i64) (param $a1 i64) (param $a2 i64) (param $a3 i64)
    (param $b0 i64) (param $b1 i64) (param $b2 i64) (param $b3 i64)
    (result i64 i64 i64 i64)
    (local $d0 i64) (local $borrow0 i64)
    (local $d1 i64) (local $borrow1 i64)
    (local $d2 i64) (local $borrow2 i64)
    (local $d3 i64)

    ;; Limb 0: d0 = a0 - b0, borrow = (a0 < b0)
    local.get $a0 local.get $b0 i64.sub local.set $d0
    local.get $a0 local.get $b0 i64.lt_u i64.extend_i32_u local.set $borrow0

    ;; Limb 1: d1 = a1 - b1 - borrow0
    local.get $a1 local.get $b1 i64.sub local.get $borrow0 i64.sub local.set $d1
    local.get $a1 local.get $b1 i64.lt_u i64.extend_i32_u
    local.get $a1 local.get $b1 i64.sub local.get $borrow0 i64.lt_u i64.extend_i32_u
    i64.add local.set $borrow1

    ;; Limb 2: d2 = a2 - b2 - borrow1
    local.get $a2 local.get $b2 i64.sub local.get $borrow1 i64.sub local.set $d2
    local.get $a2 local.get $b2 i64.lt_u i64.extend_i32_u
    local.get $a2 local.get $b2 i64.sub local.get $borrow1 i64.lt_u i64.extend_i32_u
    i64.add local.set $borrow2

    ;; Limb 3: d3 = a3 - b3 - borrow2
    local.get $a3 local.get $b3 i64.sub local.get $borrow2 i64.sub local.set $d3

    local.get $d0
    local.get $d1
    local.get $d2
    local.get $d3
  )

  ;; --------------------------------------------------------------------------
  ;; $i256_mul_scalar
  ;; Multiplies a 256-bit signed integer by a signed 64-bit scalar: (A * c)
  ;; Stack In:  (a0, a1, a2, a3, c)  [all i64, little-endian]
  ;; Stack Out: (p0, p1, p2, p3)     [all i64, little-endian]
  ;; Corrects for two's complement signed scalar multiplication via sign masking.
  ;; Fuel cost: 268 instructions, 271 fuel units
  ;; --------------------------------------------------------------------------
  (func $i256_mul_scalar
    (param $a0 i64) (param $a1 i64) (param $a2 i64) (param $a3 i64)
    (param $c i64)
    (result i64 i64 i64 i64)
    (local $lo0 i64) (local $hi0 i64)
    (local $lo1 i64) (local $hi1 i64)
    (local $lo2 i64) (local $hi2 i64)
    (local $lo3 i64) (local $hi3 i64)
    (local $p0 i64) (local $carry0 i64)
    (local $p1 i64) (local $carry1 i64)
    (local $p2 i64) (local $carry2 i64)
    (local $p3 i64)

    ;; a0 * c
    local.get $a0 local.get $c call $mul64_wide
    local.set $hi0 local.set $lo0
    local.get $lo0 local.set $p0

    ;; a1 * c + hi0
    local.get $a1 local.get $c call $mul64_wide
    local.set $hi1 local.set $lo1
    local.get $lo1 local.get $hi0 i64.add local.set $p1
    local.get $p1 local.get $hi0 i64.lt_u i64.extend_i32_u local.set $carry0

    ;; a2 * c + hi1 + carry0
    local.get $a2 local.get $c call $mul64_wide
    local.set $hi2 local.set $lo2
    local.get $lo2 local.get $hi1 i64.add local.get $carry0 i64.add local.set $p2
    local.get $lo2 local.get $hi1 i64.add local.get $hi1 i64.lt_u i64.extend_i32_u
    local.get $p2 local.get $carry0 i64.lt_u i64.extend_i32_u
    i64.add local.set $carry1

    ;; a3 * c + hi2 + carry1
    ;; Note: High product hi3 is discarded as 256-bit truncation
    local.get $a3 local.get $c i64.mul
    local.get $hi2 i64.add
    local.get $carry1 i64.add
    local.set $p3

    ;; Signed correction: if a < 0, subtract c << 256; if c < 0, subtract a
    ;; When c < 0: p3 = p3 - a3 (if c high bit set)
    local.get $c i64.const 63 i64.shr_s i64.const 0 i64.lt_s
    if
      local.get $p3 local.get $a0 i64.sub local.set $p3
    end

    local.get $p0
    local.get $p1
    local.get $p2
    local.get $p3
  )
)
