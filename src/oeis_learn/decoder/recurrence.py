"""Narrow order-two recognizer. Every declaration/transfer/guard is part of the template."""
from functools import lru_cache
from oeis_learn.decoder.wat_grammar import tokenize_wat


def order2_template(first=0, second=1):
    def get(name): return " ".join(f"local.get ${name}{j}" for j in range(4))
    def put(name): return " ".join(f"local.set ${name}{j}" for j in reversed(range(4)))
    decl = " ".join(f"(local ${name}{j} i64)" for name in ("a","b","t") for j in range(4))
    return f'''(module (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
    {decl} (local $i i32)
    i256.const {first} {put('a')} i256.const {second} {put('b')}
    (block $exit (loop $loop local.get $i local.get $n i32.ge_s br_if $exit
    {get('b')} i64.const_? i256.mul_scalar {get('a')} i64.const_? i256.mul_scalar i256.add {put('t')}
    {get('b')} {put('a')} {get('t')} {put('b')}
    local.get $i i32.const 1 i32.add local.set $i br $loop)) {get('a')}))'''


def recognize(source):
    if not isinstance(source,str) or len(source.encode()) > 65536 or ";" in source: return None
    words = tokenize_wat(source)
    try:
        values = [int(words[i+1]) for i,w in enumerate(words) if w == 'i256.const']
        if len(values) != 2 or any(not -(1<<255) <= v < (1<<255) for v in values): return None
        return tuple(values) if words == tokenize_wat(order2_template(*values)) else None
    except (ValueError,IndexError): return None


@lru_cache(maxsize=128)
def lag_system(visible):
    if not 3 <= len(visible) <= 20: raise ValueError('recurrence prefix length')
    return tuple((visible[n-1],visible[n-2]) for n in range(2,len(visible))), tuple(visible[2:])


def reference(initial, coefficients, count):
    values = []
    for n in range(count):
        a,b = initial
        for _ in range(n):
            x,y = b*coefficients[0],a*coefficients[1]
            if any(not -(1<<255) <= v < (1<<255) for v in (x,y,x+y)): raise OverflowError('i256 recurrence intermediate')
            a,b = b,x+y
        values.append(a)
    return values
