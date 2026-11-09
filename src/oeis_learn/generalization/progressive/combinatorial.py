"""Reviewed local permutation statistics; no source-provided code is imported."""

import itertools
import json
import math
import time
from .contracts import seal, digest

ALLOWLIST = (
    "inversions",
    "descents",
    "ascents",
    "major_index",
    "fixed_points",
    "excedances",
    "cycles",
    "records",
)


def statistic(name, p):
    if name == "inversions":
        return sum(a > b for i, a in enumerate(p) for b in p[i + 1 :])
    if name == "descents":
        return sum(a > b for a, b in zip(p, p[1:]))
    if name == "ascents":
        return sum(a < b for a, b in zip(p, p[1:]))
    if name == "major_index":
        return sum(i for i in range(1, len(p)) if p[i - 1] > p[i])
    if name == "fixed_points":
        return sum(i == v for i, v in enumerate(p))
    if name == "excedances":
        return sum(v > i for i, v in enumerate(p))
    if name == "records":
        return sum(all(v > x for x in p[:i]) for i, v in enumerate(p))
    if name == "cycles":
        seen = set()
        count = 0
        for i in range(len(p)):
            if i not in seen:
                count += 1
                while i not in seen:
                    seen.add(i)
                    i = p[i]
        return count
    raise ValueError("unapproved statistic")


def _alternate(n):
    # Insertion enumeration is independent of itertools permutation traversal.
    if n == 0:
        yield ()
    else:
        for p in _alternate(n - 1):
            for i in range(n):
                yield p[:i] + (n - 1,) + p[i:]


def prepare(
    statistics,
    *,
    moments=(1, 2),
    max_objects=46234,
    max_work=100_000_000,
    max_bytes=1 << 20,
    seconds=30,
):
    if (
        type(statistics) is not list
        or not 1 <= len(statistics) <= 32
        or len(set(statistics)) != len(statistics)
        or set(statistics) - set(ALLOWLIST)
        or not moments
        or len(set(moments)) != len(moments)
        or any(type(m) is not int or m not in (1, 2) for m in moments)
    ):
        raise ValueError("frozen allowlisted statistics and moments required")
    for value, cap in [(max_objects, 46234), (max_work, 100_000_000), (max_bytes, 1 << 20)]:
        if type(value) is not int or not 1 <= value <= cap:
            raise ValueError("invalid/enlarged source budget")
    if type(seconds) not in (int, float) or not 0 < seconds <= 30:
        raise ValueError("bounded enumeration deadline")
    objects = sum(math.factorial(n) for n in range(9))
    # Reserve conservative pair comparisons, traversals and moment arithmetic.
    work = objects * len(statistics) * (8 * 8 + 2 * len(moments))
    if objects > max_objects or work > max_work:
        raise TimeoutError("enumeration reservation exhausted")
    deadline = time.monotonic() + seconds
    totals = {(s, m): [] for s in statistics for m in moments}
    for n in range(9):
        acc = {k: 0 for k in totals}
        count = 0
        for p in itertools.permutations(range(n)):
            if count % 256 == 0 and time.monotonic() >= deadline:
                raise TimeoutError("enumeration deadline")
            count += 1
            for s in statistics:
                v = statistic(s, p)
                for m in moments:
                    acc[s, m] += v**m
        if count != math.factorial(n):
            raise ValueError("incomplete collection")
        if n <= 5:
            checked = {k: 0 for k in totals}
            for p in _alternate(n):
                # Reviewed alternate enumeration checks order/cardinality; the
                # closed-form inversion expectation below checks semantics too.
                for s in statistics:
                    for m in moments:
                        checked[s, m] += statistic(s, p) ** m
            if checked != acc:
                raise ValueError("alternate enumeration differs")
        if ("inversions", 1) in acc and acc["inversions", 1] != math.factorial(n) * n * (
            n - 1
        ) // 4:
            raise ValueError("independent inversion total differs")
        if ("fixed_points", 1) in acc and acc["fixed_points", 1] != (math.factorial(n) if n else 0):
            raise ValueError("independent fixed-point total differs")
        for k in totals:
            totals[k].append(str(acc[k]))
    value = seal(
        dict(
            version="progressive-combinatorial/v1",
            track="external_assisted",
            adapter=digest(__import__("pathlib").Path(__file__).read_text()),
            cardinalities=[str(math.factorial(n)) for n in range(9)],
            objects=objects,
            reserved_work=work,
            zero_size="one empty permutation; all approved statistics zero",
            rows=[
                dict(statistic=s, moment=m, indices=list(map(str, range(9))), values=v)
                for (s, m), v in totals.items()
            ],
        )
    )
    if len(json.dumps(value).encode()) > max_bytes:
        raise TimeoutError("enumeration byte budget")
    if time.monotonic() >= deadline:
        raise TimeoutError("enumeration deadline")
    return value
