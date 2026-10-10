"""Narrow order-two recognizer. Every declaration/transfer/guard is part of the template."""

from functools import lru_cache
from oeis_learn.decoder.wat_grammar import tokenize_wat


def order2_template(first=0, second=1):
    def get(name):
        return " ".join(f"local.get ${name}{j}" for j in range(4))

    def put(name):
        return " ".join(f"local.set ${name}{j}" for j in reversed(range(4)))

    decl = " ".join(f"(local ${name}{j} i64)" for name in ("a", "b", "t") for j in range(4))
    return f"""(module (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
    {decl} (local $i i32)
    i256.const {first} {put("a")} i256.const {second} {put("b")}
    (block $exit (loop $loop local.get $i local.get $n i32.ge_s br_if $exit
    {get("b")} i64.const_? i256.mul_scalar {get("a")} i64.const_? i256.mul_scalar i256.add {put("t")}
    {get("b")} {put("a")} {get("t")} {put("b")}
    local.get $i i32.const 1 i32.add local.set $i br $loop)) {get("a")}))"""


def _structure(source, grounded=False):
    """Recognize the supported two-state transition after alpha/copy normalization.

    Roles come from initial-state writes and output reads. Local declarations
    can be reordered; parallel limb copies can be expressed as grouped stack
    transfers or interleaved get/set pairs. Every control guard, update, initial
    value and output must still match the proven transition.
    """
    if not isinstance(source, str) or len(source.encode()) > 65536 or ";" in source:
        return None
    words = tokenize_wat(source)
    try:
        # Recover roles from the two exact wide initial-state assignments.
        positions = [i for i, w in enumerate(words) if w == "i256.const"]
        if len(positions) != 2:
            return None
        values = tuple(int(words[i + 1]) for i in positions)
        if any(not -(1 << 255) <= v < (1 << 255) for v in values):
            return None
        mapping = {}
        groups = []
        for role, i in zip(("a", "b"), positions):
            names = []
            for j in range(4):
                if words[i + 2 + 2 * j] != "local.set":
                    return None
                name = words[i + 3 + 2 * j]
                names.append(name)
                mapping[name] = f"${role}{3 - j}"
            groups.append(names)
        # The temporary receives the wide update immediately after its add.
        adds = [i for i, w in enumerate(words) if w == "i256.add"]
        if len(adds) != 1:
            return None
        i = adds[0]
        for j in range(4):
            if words[i + 1 + 2 * j] != "local.set":
                return None
            mapping[words[i + 2 + 2 * j]] = f"$t{3 - j}"
        params = [i for i, w in enumerate(words) if w == "param"]
        counters = [
            words[i + 1] for i, w in enumerate(words) if w == "local" and words[i + 2] == "i32"
        ]
        if len(params) != 1 or len(counters) != 1:
            return None
        mapping[words[params[0] + 1]] = "$n"
        mapping[counters[0]] = "$i"
        blocks = [i for i, w in enumerate(words) if w in ("block", "loop")]
        if len(blocks) != 2 or words[blocks[0]] != "block" or words[blocks[1]] != "loop":
            return None
        mapping[words[blocks[0] + 1]] = "$exit"
        mapping[words[blocks[1] + 1]] = "$loop"
        if len(mapping) != 16:
            return None
        words = [mapping.get(w, w) for w in words if w != "nop"]
        # Validate declaration inventory by type, independently of order.
        declarations = []
        normalized = []
        i = 0
        while i < len(words):
            if words[i : i + 2] == ["(", "local"]:
                if words[i + 4] != ")":
                    return None
                declarations.append(tuple(words[i + 2 : i + 4]))
                i += 5
            else:
                normalized.append(words[i])
                i += 1
        expected = [(f"${role}{j}", "i64") for role in ("a", "b", "t") for j in range(4)] + [
            ("$i", "i32")
        ]
        if sorted(declarations) != sorted(expected):
            return None
        # Canonicalize proven non-overlapping parallel limb transfers.
        for src, dst in (("b", "a"), ("t", "b")):
            interleaved = [
                w for j in range(4) for w in ("local.get", f"${src}{j}", "local.set", f"${dst}{j}")
            ]
            grouped = [w for j in range(4) for w in ("local.get", f"${src}{j}")] + [
                w for j in reversed(range(4)) for w in ("local.set", f"${dst}{j}")
            ]
            for i in range(len(normalized) - len(interleaved) + 1):
                if normalized[i : i + len(interleaved)] == interleaved:
                    normalized[i : i + len(interleaved)] = grouped
        coefficients = []
        if grounded:
            for i, w in enumerate(normalized):
                if (
                    w == "i64.const"
                    and i + 2 < len(normalized)
                    and normalized[i + 2] == "i256.mul_scalar"
                ):
                    coefficients.append(int(normalized[i + 1]))
            if len(coefficients) != 2 or any(abs(c) > 1000 for c in coefficients):
                return None
            rebuilt = []
            i = 0
            while i < len(normalized):
                if (
                    normalized[i] == "i64.const"
                    and i + 2 < len(normalized)
                    and normalized[i + 2] == "i256.mul_scalar"
                ):
                    rebuilt.append("i64.const_?")
                    i += 2
                else:
                    rebuilt.append(normalized[i])
                    i += 1
            normalized = rebuilt
        template = tokenize_wat(order2_template(*values))
        without_declarations = []
        i = 0
        while i < len(template):
            if template[i : i + 2] == ["(", "local"]:
                i += 5
            else:
                without_declarations.append(template[i])
                i += 1
        if normalized != without_declarations:
            return None
        return values, tuple(coefficients)
    except (ValueError, IndexError):
        return None


def recognize(source):
    recognized = _structure(source)
    return recognized[0] if recognized else None


def recognize_grounded(source):
    return _structure(source, grounded=True)


@lru_cache(maxsize=128)
def lag_system(visible, fit_indices=None, order=2, transform_sha256=None):
    if order != 2 or (fit_indices is not None and fit_indices != tuple(range(len(visible)))):
        raise ValueError("unsupported recurrence indexing/order")
    if not 3 <= len(visible) <= 20:
        raise ValueError("recurrence prefix length")
    return tuple((visible[n - 1], visible[n - 2]) for n in range(2, len(visible))), tuple(
        visible[2:]
    )


def reference(initial, coefficients, count):
    values = []
    for n in range(count):
        a, b = initial
        for _ in range(n):
            x, y = b * coefficients[0], a * coefficients[1]
            if any(not -(1 << 255) <= v < (1 << 255) for v in (x, y, x + y)):
                raise OverflowError("i256 recurrence intermediate")
            a, b = b, x + y
        values.append(a)
    return values
