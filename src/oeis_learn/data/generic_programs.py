"""Disclosed, target-independent typed sampling for generic_sampler_v1.

Only a frozen config and an integer counter enter sampling. No corpus, sequence
identity, teacher, model, replay or execution feedback enters the distribution.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import copy
import random
import yaml

from oeis_learn.decoder.program_codec import CodecError, decode_body, encode_body
from oeis_learn.evaluation.foundation_cohort import exact_keys
from oeis_learn.experiments.artifacts import compute_canonical_digest as digest, compute_file_hash
from oeis_learn.experiments.config import _UniqueLoader
from oeis_learn.experiments.profiles import profile_digests, ALLOWED_OPERATORS

PRIORS = {
    "statement_count": [1, 32],
    "statement_weights": [70, 15, 15],
    "expression_depth": 4,
    "expression_leaf_probability": [1, 2],
    "loop_depth": 3,
    "structured_depth": 8,
    "wide_registers": 8,
    "i32_temporaries": 8,
    "small_constant_probability": [7, 8],
    "small_constants": [-16, 16],
    "signed_widths": [8, 16, 32, 64, 128, 256],
    "loop_constant_bounds": [0, 32],
    "loop_index_cap": 100,
    "loop_bound_weights": [1, 1],
    "leaf_weights": [1, 1, 1],
    "output_register_weights": [1] * 8,
}
RNG_ID = "python-mt19937/counter-sha256/v1"
BINARY = (
    "add",
    "sub",
    "mul",
    "div_s",
    "div_u",
    "rem_s",
    "rem_u",
    "and",
    "or",
    "xor",
    "shl",
    "shr_s",
    "shr_u",
)
COMPARE = ("eq", "ne", "lt_s", "gt_s", "le_s", "ge_s")
OPS = {
    "wide": [
        ("i256.add", ("wide", "wide")),
        ("i256.sub", ("wide", "wide")),
        ("i256.mul_scalar", ("wide", "i64")),
    ],
    "i64": [("i64." + op, ("i64", "i64")) for op in BINARY]
    + [("i64.extend_i32_s", ("i32",)), ("i64.extend_i32_u", ("i32",))],
    "i32": [("i32." + op, ("i32", "i32")) for op in (*BINARY, *COMPARE)]
    + [("i64." + op, ("i64", "i64")) for op in COMPARE]
    + [("i32.eqz", ("i32",)), ("i64.eqz", ("i64",)), ("i32.wrap_i64", ("i64",))],
}


def default_config():
    return {
        "schema_version": "foundation/v1",
        "profile": "generic_sampler_v1",
        "seed": 20260913,
        "target_count": 64,
        "max_attempts": 10000,
        "budget_ns": 300_000_000_000,
        "workers": 4,
        "chunk_size": 8,
        "priors": copy.deepcopy(PRIORS),
    }


def validate_config(config):
    exact_keys(config, default_config(), "generic sampler config")
    if (
        config["schema_version"] != "foundation/v1"
        or config["profile"] != "generic_sampler_v1"
        or digest(config["priors"]) != digest(PRIORS)
    ):
        raise ValueError("unknown sampler version or changed frozen priors")
    for key, low, high in (
        ("seed", 0, 2**64 - 1),
        ("target_count", 1, 64),
        ("max_attempts", 1, 10000),
        ("budget_ns", 1, 300_000_000_000),
        ("workers", 1, 8),
        ("chunk_size", 8, 8),
    ):
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError(f"{key}: exact integer in {low}..{high} required")
    return copy.deepcopy(config)


def load_sampler_config(path):
    with Path(path).open() as stream:
        return validate_config(yaml.load(stream, Loader=_UniqueLoader))


@dataclass(frozen=True)
class Expression:
    type: str
    op: str
    children: tuple[Expression, ...] = ()
    value: int | None = None

    def source(self):
        if self.op == "local":
            if self.type == "wide":
                return " ".join(f"local.get $v{4 * self.value + i}" for i in range(4))
            return f"local.get ${'v' if self.type == 'i64' else 'c'}{self.value}"
        if self.op == "index":
            return {
                "i32": "local.get $n",
                "i64": "local.get $n i64.extend_i32_u",
                "wide": "local.get $n i64.extend_i32_u i64.const 0 i64.const 0 i64.const 0",
            }[self.type]
        if self.op == "constant":
            if self.type == "wide":
                return "i256.zero" if self.value == 0 else f"i256.const {self.value}"
            return f"{self.type}.const {self.value}"
        return " ".join([*(child.source() for child in self.children), self.op])


@dataclass(frozen=True)
class Statement:
    kind: str
    expression: Expression
    register: int = 0
    body: tuple[Statement, ...] = ()
    otherwise: tuple[Statement, ...] = ()
    bound: int | None = None

    def source(self):
        body = " ".join(s.source() for s in self.body)
        if self.kind == "assignment":
            stores = " ".join(f"local.set $v{4 * self.register + i}" for i in reversed(range(4)))
            return self.expression.source() + " " + stores
        if self.kind == "conditional":
            other = " ".join(s.source() for s in self.otherwise)
            return f"{self.expression.source()} ( if ( then {body} ) ( else {other} ) )"
        # Two fresh temporaries per nesting level: counter and frozen bound.
        counter, limit = f"$c{2 * self.register}", f"$c{2 * self.register + 1}"
        bound = (
            f"i32.const {self.bound}"
            if self.bound is not None
            else "local.get $n i32.const 100 i32.lt_s "
            "( if ( result i32 ) ( then local.get $n ) ( else i32.const 100 ) )"
        )
        return (
            f"{bound} local.set {limit} i32.const 0 local.set {counter} "
            f"( block ( loop local.get {counter} local.get {limit} i32.ge_s br_if 1 "
            f"{body} local.get {counter} i32.const 1 i32.add local.set {counter} br 0 ) )"
        )


class GenericSampler:
    def __init__(self, config):
        self.config = validate_config(config)
        # Execution concurrency, yield targets and time limits cannot change
        # the generic draw stream. They remain pinned by the build manifest.
        self.config_hash = digest(
            {key: self.config[key] for key in ("schema_version", "profile", "priors")}
        )
        self.revision = compute_file_hash(Path(__file__))
        self.rng_identity = RNG_ID

    def sample(self, counter):
        if type(counter) is not int or not 0 <= counter < self.config["max_attempts"]:
            raise ValueError("sample counter outside frozen bounds")
        seed = int(
            digest([RNG_ID, self.revision, self.config_hash, self.config["seed"], counter])[7:23],
            16,
        )
        rng = random.Random(seed)
        kinds, leaves, operators, constants = Counter(), Counter(), Counter(), Counter()
        stats = {"statements": 0, "loop_depth": 0, "structured_depth": 0}

        def expression(typ, depth=0):
            if depth == 4 or rng.randrange(2) == 0:
                leaf = rng.choice(("local", "index", "constant"))
                leaves[typ + ":" + leaf] += 1
                if leaf == "local":
                    return Expression(typ, leaf, value=rng.randrange(32 if typ == "i64" else 8))
                if leaf == "index":
                    return Expression(typ, leaf)
                width = {"wide": 256, "i64": 64, "i32": 32}[typ]
                if rng.randrange(8) < 7:
                    value = rng.randint(-16, 16)
                    constants["small"] += 1
                else:
                    bits = rng.choice([w for w in PRIORS["signed_widths"] if w <= width])
                    value = rng.randrange(-(2 ** (bits - 1)), 2 ** (bits - 1))
                    constants[str(bits)] += 1
                return Expression(typ, leaf, value=value)
            op, inputs = rng.choice(OPS[typ])
            operators[op] += 1
            return Expression(typ, op, tuple(expression(t, depth + 1) for t in inputs))

        def block(count, depth=0, loops=0):
            result = []
            stats["structured_depth"] = max(stats["structured_depth"], depth)
            stats["loop_depth"] = max(stats["loop_depth"], loops)
            while count:
                options, weights = ["assignment"], [70]
                if count >= 3 and depth < 8:
                    options.append("conditional")
                    weights.append(15)
                if count >= 2 and depth + 2 <= 8 and loops < 3:
                    options.append("counted_loop")
                    weights.append(15)
                kind = rng.choices(options, weights=weights)[0]
                kinds[kind] += 1
                stats["statements"] += 1
                count -= 1
                if kind == "assignment":
                    result.append(Statement(kind, expression("wide"), rng.randrange(8)))
                elif kind == "conditional":
                    used = rng.randint(2, count)
                    left = rng.randint(1, used - 1)
                    condition = expression("i32")
                    result.append(
                        Statement(
                            kind,
                            condition,
                            body=block(left, depth + 1, loops),
                            otherwise=block(used - left, depth + 1, loops),
                        )
                    )
                    count -= used
                else:
                    used = rng.randint(1, count)
                    bound = rng.randint(0, 32) if rng.randrange(2) == 0 else None
                    result.append(
                        Statement(
                            kind,
                            Expression("i32", "constant", value=0),
                            register=loops,
                            body=block(used, depth + 2, loops + 1),
                            bound=bound,
                        )
                    )
                    count -= used
            return tuple(result)

        statements = block(rng.randint(1, 32))
        source = " ".join(s.source() for s in statements)
        source += " " + Expression("wide", "local", value=rng.randrange(8)).source()
        failure, tokens = None, []
        try:
            tokens = encode_body(source)
            source = decode_body(tokens)
        except CodecError as exc:
            failure = (
                "token_or_source_limit"
                if "cap" in str(exc) or "64 KiB" in str(exc)
                else "invalid_sample"
            )
        stats.update(
            statement_kinds=dict(kinds),
            leaves=dict(leaves),
            operators=dict(operators),
            constants=dict(constants),
            emitted_operators=dict(
                Counter(word for word in source.split() if word in ALLOWED_OPERATORS)
            ),
        )
        return {
            "generator_revision": self.revision,
            "generator_config_hash": self.config_hash,
            "sample_seed": seed,
            "sample_counter": counter,
            "origin": "generic_sample",
            "language_profile": profile_digests()["language"],
            "codec_profile": profile_digests()["codec"],
            "canonical_source": source,
            "body_tokens": tokens,
            "statistics": stats,
            "sampling_failure": failure,
        }
