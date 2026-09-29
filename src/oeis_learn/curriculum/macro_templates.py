"""Macro-Scaffolding Code Templates for Curriculum Stages 1 through 4."""

from __future__ import annotations

from typing import Dict, List, Optional
from oeis_learn.data.models import CurriculumMacroTemplate


def get_stage1_polynomial_template(degree: int = 2) -> str:
    """Generates Stage 1 closed-form polynomial evaluation template (no loops)."""
    return """(module
  (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
    (local $n64 i64)
    local.get $n i64.extend_i32_u local.set $n64
    local.get $n64 local.get $n64 i64.mul
    i64.const 0 i64.const 0 i64.const 0
  )
)"""


def get_stage2_linear_recurrence_template(order: int = 2) -> str:
    """Generates Stage 2 linear recurrence template for orders 1 through 4."""
    if order == 1:
        # a(n) = c * a(n-1)
        return """(module
  (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
    (local $a0 i64) (local $a1 i64) (local $a2 i64) (local $a3 i64)
    (local $i i32)
    i256.const 1 local.set $a3 local.set $a2 local.set $a1 local.set $a0
    (block $exit
      (loop $loop
        local.get $i local.get $n i32.ge_s br_if $exit
        local.get $a0 local.get $a1 local.get $a2 local.get $a3
        i64.const 2
        i256.mul_scalar
        local.set $a3 local.set $a2 local.set $a1 local.set $a0
        local.get $i i32.const 1 i32.add local.set $i
        br $loop
      )
    )
    local.get $a0 local.get $a1 local.get $a2 local.get $a3
  )
)"""
    elif order == 2:
        return """(module
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
)"""
    elif order == 3:
        return """(module
  (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
    (local $a0 i64) (local $a1 i64) (local $a2 i64) (local $a3 i64)
    (local $b0 i64) (local $b1 i64) (local $b2 i64) (local $b3 i64)
    (local $c0 i64) (local $c1 i64) (local $c2 i64) (local $c3 i64)
    (local $t0 i64) (local $t1 i64) (local $t2 i64) (local $t3 i64)
    (local $i i32)
    i256.zero local.set $a3 local.set $a2 local.set $a1 local.set $a0
    i256.zero local.set $b3 local.set $b2 local.set $b1 local.set $b0
    i256.const 1 local.set $c3 local.set $c2 local.set $c1 local.set $c0
    (block $exit
      (loop $loop
        local.get $i local.get $n i32.ge_s br_if $exit
        local.get $c0 local.get $c1 local.get $c2 local.get $c3
        local.get $b0 local.get $b1 local.get $b2 local.get $b3
        i256.add
        local.get $a0 local.get $a1 local.get $a2 local.get $a3
        i256.add
        local.set $t3 local.set $t2 local.set $t1 local.set $t0
        local.get $b0 local.set $a0 local.get $b1 local.set $a1
        local.get $b2 local.set $a2 local.get $b3 local.set $a3
        local.get $c0 local.set $b0 local.get $c1 local.set $b1
        local.get $c2 local.set $b2 local.get $c3 local.set $b3
        local.get $t0 local.set $c0 local.get $t1 local.set $c1
        local.get $t2 local.set $c2 local.get $t3 local.set $c3
        local.get $i i32.const 1 i32.add local.set $i
        br $loop
      )
    )
    local.get $a0 local.get $a1 local.get $a2 local.get $a3
  )
)"""
    else:
        # Order 4
        return """(module
  (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
    (local $a0 i64) (local $a1 i64) (local $a2 i64) (local $a3 i64)
    (local $b0 i64) (local $b1 i64) (local $b2 i64) (local $b3 i64)
    (local $c0 i64) (local $c1 i64) (local $c2 i64) (local $c3 i64)
    (local $d0 i64) (local $d1 i64) (local $d2 i64) (local $d3 i64)
    (local $t0 i64) (local $t1 i64) (local $t2 i64) (local $t3 i64)
    (local $i i32)
    i256.zero local.set $a3 local.set $a2 local.set $a1 local.set $a0
    i256.zero local.set $b3 local.set $b2 local.set $b1 local.set $b0
    i256.zero local.set $c3 local.set $c2 local.set $c1 local.set $c0
    i256.const 1 local.set $d3 local.set $d2 local.set $d1 local.set $d0
    (block $exit
      (loop $loop
        local.get $i local.get $n i32.ge_s br_if $exit
        local.get $d0 local.get $d1 local.get $d2 local.get $d3
        local.get $c0 local.get $c1 local.get $c2 local.get $c3
        i256.add
        local.get $b0 local.get $b1 local.get $b2 local.get $b3
        i256.add
        local.get $a0 local.get $a1 local.get $a2 local.get $a3
        i256.add
        local.set $t3 local.set $t2 local.set $t1 local.set $t0
        local.get $b0 local.set $a0 local.get $b1 local.set $a1
        local.get $b2 local.set $a2 local.get $b3 local.set $a3
        local.get $c0 local.set $b0 local.get $c1 local.set $b1
        local.get $c2 local.set $b2 local.get $c3 local.set $b3
        local.get $d0 local.set $c0 local.get $d1 local.set $c1
        local.get $d2 local.set $c2 local.get $d3 local.set $c3
        local.get $t0 local.set $d0 local.get $t1 local.set $d1
        local.get $t2 local.set $d2 local.get $t3 local.set $d3
        local.get $i i32.const 1 i32.add local.set $i
        br $loop
      )
    )
    local.get $a0 local.get $a1 local.get $a2 local.get $a3
  )
)"""


def get_stage3_holonomic_template() -> str:
    """Generates Stage 3 holonomic/factorial loop template (a(n) = n * a(n-1))."""
    return """(module
  (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
    (local $a0 i64) (local $a1 i64) (local $a2 i64) (local $a3 i64)
    (local $i i32)
    i256.const 1 local.set $a3 local.set $a2 local.set $a1 local.set $a0
    i32.const 1 local.set $i
    (block $exit
      (loop $loop
        local.get $i local.get $n i32.gt_s br_if $exit
        local.get $a0 local.get $a1 local.get $a2 local.get $a3
        local.get $i i64.extend_i32_u
        i256.mul_scalar
        local.set $a3 local.set $a2 local.set $a1 local.set $a0
        local.get $i i32.const 1 i32.add local.set $i
        br $loop
      )
    )
    local.get $a0 local.get $a1 local.get $a2 local.get $a3
  )
)"""


def get_stage4_divisor_loop_template() -> str:
    """Generates Stage 4 nested loop template for divisor counting sigma_0(n)."""
    return """(module
  (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
    (local $cnt i64)
    (local $d i32)
    i64.const 0 local.set $cnt
    i32.const 1 local.set $d
    (block $exit
      (loop $loop
        local.get $d local.get $n i32.gt_s br_if $exit
        local.get $n local.get $d i32.rem_u i32.eqz
        if
          local.get $cnt i64.const 1 i64.add local.set $cnt
        end
        local.get $d i32.const 1 i32.add local.set $d
        br $loop
      )
    )
    local.get $cnt i64.const 0 i64.const 0 i64.const 0
  )
)"""
