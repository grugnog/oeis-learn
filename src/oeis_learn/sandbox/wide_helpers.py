"""Trusted deterministic WAT helper generation for logical signed256 v1.

32-bit schoolbook products retain all 512 bits; unsigned long division retains
all four remainder limbs. Python arithmetic here constructs code, not outputs.
"""

from oeis_learn.sandbox.preamble import load_preamble_wat


def get(prefix):
    return " ".join(f"local.get ${prefix}{i}" for i in range(4))


def put(prefix):
    return " ".join(f"local.set ${prefix}{i}" for i in reversed(range(4)))


def decl(kind, prefix, n=4):
    return " ".join(f"({kind} ${prefix}{i} i64)" for i in range(n))


def numeric_trap():
    return "i32.const 1 global.set $__oeis_numeric_limit unreachable"


def helpers():
    base = load_preamble_wat()
    base = base[base.index("(module") + 7 : base.rfind(")")]
    chunks = [base]
    params = decl("param", "a") + " " + decl("param", "b")
    result = "(result i64 i64 i64 i64)"
    # Modular subtraction/negation are private unsigned-magnitude operations.
    steps = []
    for i in range(4):
        steps.append(
            f"local.get $a{i} local.get $b{i} i64.sub local.tee $t local.get $carry i64.sub local.set $r{i}"
        )
        steps.append(
            f"local.get $a{i} local.get $b{i} i64.lt_u local.get $t local.get $carry i64.lt_u i32.or i64.extend_i32_u local.set $carry"
        )
    chunks.append(
        f"(func $rawsub {params} {result} {decl('local', 'r')} (local $carry i64) (local $t i64) {' '.join(steps)} {get('r')})"
    )
    chunks.append(
        f"(func $neg {decl('param', 'a')} {result} i64.const 0 i64.const 0 i64.const 0 i64.const 0 {get('a')} call $rawsub)"
    )
    for signed, name in ((False, "ucmp"), (True, "scmp")):
        body = "i32.const 0"
        for i in range(4):
            suffix = "s" if signed and i == 3 else "u"
            body = f"local.get $a{i} local.get $b{i} i64.ne if (result i32) local.get $a{i} local.get $b{i} i64.lt_{suffix} if (result i32) i32.const -1 else i32.const 1 end else {body} end"
        chunks.append(f"(func ${name} {params} (result i32) {body})")
    for op in ("and", "or", "xor"):
        chunks.append(
            f"(func ${op} {params} {result} "
            + " ".join(f"local.get $a{i} local.get $b{i} i64.{op}" for i in range(4))
            + ")"
        )
    abs_a = f"local.get $a3 i64.const 0 i64.lt_s local.set $sa local.get $sa if {get('a')} call $neg {put('a')} end"
    abs_b = f"local.get $b3 i64.const 0 i64.lt_s local.set $sb local.get $sb if {get('b')} call $neg {put('b')} end"
    # A magnitude is <= 2^255 exactly when its top limb is <2^63 or equals
    # 2^63 and all lower limbs are zero. Positive results cannot equal 2^255.
    rangecheck = f"""local.get $r3 i64.const -9223372036854775808 i64.gt_u
      local.get $r3 i64.const -9223372036854775808 i64.eq
      local.get $r0 local.get $r1 i64.or local.get $r2 i64.or i64.const 0 i64.ne
      local.get $sa local.get $sb i32.xor i32.eqz i32.or i32.and i32.or
      if {numeric_trap()} end"""
    digits = []
    for name in ("a", "b"):
        for i in range(8):
            digits.append(
                f"local.get ${name}{i // 2} i64.const {32 * (i % 2)} i64.shr_u i64.const 4294967295 i64.and local.set ${name}d{i}"
            )
    products = []
    for i in range(8):
        products.append("i64.const 0 local.set $carry")
        for j in range(8):
            products.append(
                f"local.get $ad{i} local.get $bd{j} i64.mul local.get $rd{i + j} i64.add local.get $carry i64.add local.tee $t i64.const 4294967295 i64.and local.set $rd{i + j} local.get $t i64.const 32 i64.shr_u local.set $carry"
            )
        products.append(f"local.get $carry local.set $rd{i + 8}")
    assemble = " ".join(
        f"local.get $rd{i * 2} local.get $rd{i * 2 + 1} i64.const 32 i64.shl i64.or local.set $r{i}"
        for i in range(4)
    )
    high = " ".join(["local.get $rd8", *[f"local.get $rd{i} i64.or" for i in range(9, 16)]])
    chunks.append(f"""(func $mul {params} {result} {decl("local", "r")} {decl("local", "ad", 8)} {decl("local", "bd", 8)} {decl("local", "rd", 16)} (local $sa i32) (local $sb i32) (local $carry i64) (local $t i64)
    {abs_a} {abs_b} {" ".join(digits)} {" ".join(products)} {high} i64.const 0 i64.ne if {numeric_trap()} end {assemble} {rangecheck}
    local.get $sa local.get $sb i32.xor if {get("r")} call $neg {put("r")} end {get("r")})""")
    # Shift-one private routines; no native masked wide shift semantics.
    left = " ".join(
        f"local.get $a{i} i64.const 1 i64.shl "
        + (f"local.get $a{i - 1} i64.const 63 i64.shr_u i64.or" if i else "")
        for i in range(4)
    )
    chunks.append(f"(func $left {decl('param', 'a')} {result} {left})")
    for op, top in (("shr_s", "s"), ("shr_u", "u")):
        right = " ".join(
            f"local.get $a{i} i64.const 1 i64.shr_{top if i == 3 else 'u'} "
            + (f"local.get $a{i + 1} i64.const 63 i64.shl i64.or" if i < 3 else "")
            for i in range(4)
        )
        chunks.append(f"(func ${op}_one {decl('param', 'a')} {result} {right})")
    for op in ("shl", "shr_s", "shr_u"):
        step = (
            f"{get('a')} {get('a')} call $i256_add" if op == "shl" else f"{get('a')} call ${op}_one"
        )
        chunks.append(f"""(func ${op} {decl("param", "a")} (param $k i32) {result}
        local.get $k i32.const 256 i32.ge_u if unreachable end
        block $done loop $again local.get $k i32.eqz br_if $done
        {step} {put("a")} local.get $k i32.const 1 i32.sub local.set $k br $again end end {get("a")})""")
    bit = "local.get $i i32.const 192 i32.ge_u if (result i64) local.get $a3 else local.get $i i32.const 128 i32.ge_u if (result i64) local.get $a2 else local.get $i i32.const 64 i32.ge_u if (result i64) local.get $a1 else local.get $a0 end end end local.get $i i64.extend_i32_u i64.shr_u i64.const 1 i64.and"
    chunks.append(f"""(func $divmod {params} (result i64 i64 i64 i64 i64 i64 i64 i64)
      {decl("local", "q")} {decl("local", "r")} (local $i i32) (local $sa i32) (local $sb i32)
      {get("b")} i64.or i64.or i64.or i64.eqz if unreachable end
      {abs_a} {abs_b} i32.const 255 local.set $i
      block $done loop $bits
        {get("r")} call $left {put("r")} local.get $r0 {bit} i64.or local.set $r0
        {get("q")} call $left {put("q")}
        {get("r")} {get("b")} call $ucmp i32.const 0 i32.ge_s
        if {get("r")} {get("b")} call $rawsub {put("r")} local.get $q0 i64.const 1 i64.or local.set $q0 end
        local.get $i i32.eqz br_if $done local.get $i i32.const 1 i32.sub local.set $i br $bits
      end end
      {rangecheck.replace("$r", "$q")}
      local.get $sa local.get $sb i32.xor if {get("q")} call $neg {put("q")} end
      local.get $sa if {get("r")} call $neg {put("r")} end {get("q")} {get("r")})""")
    for op in ("div", "rem"):
        chunks.append(
            f"(func ${op} {params} {result} {decl('local', 'r')} {decl('local', 'q')} {get('a')} {get('b')} call $divmod {put('r')} {put('q')} {get('q' if op == 'div' else 'r')})"
        )
    chunks.append(f"""(func $bounds {decl("param", "a")} (result i32)
      local.get $a1 local.get $a2 i64.or local.get $a3 i64.or i64.const 0 i64.ne
      local.get $a0 i64.const 1024 i64.ge_u i32.or if unreachable end
      local.get $a0 i32.wrap_i64 i32.const 32 i32.mul)""")
    chunks.append(f"""(func $to_i32 {decl("param", "a")} (result i32) (local $v i64)
      local.get $a0 i32.wrap_i64 i64.extend_i32_s local.set $v
      local.get $v local.get $a0 i64.ne
      local.get $a1 local.get $v i64.const 63 i64.shr_s i64.ne i32.or
      local.get $a2 local.get $v i64.const 63 i64.shr_s i64.ne i32.or
      local.get $a3 local.get $v i64.const 63 i64.shr_s i64.ne i32.or
      if {numeric_trap()} end local.get $v i32.wrap_i64)""")
    return "\n".join(chunks)
