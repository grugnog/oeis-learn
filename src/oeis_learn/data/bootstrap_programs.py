"""Versioned, target-independent connected expressions with parameter variants.

No OEIS values, sequence names, model output or execution feedback enter draws.
All statements contribute to the returned value; semantic cancellations are still
possible and existing admission/conditioning deduplication remains mandatory.
"""

import copy
import random
from pathlib import Path

from oeis_learn.decoder.program_codec import encode_body, decode_body
from oeis_learn.experiments.artifacts import compute_canonical_digest as digest, compute_file_hash
from oeis_learn.experiments.profiles import profile_digests

PRIORS = {
    "variants": 16,
    "stages": 4,
    "constant_range": [-16, 16],
    "scalar_choices": [-3, -2, -1, 1, 2, 3],
    "loop_cap": 100,
    "index_scalar_probability": [1, 2],
    "coupled_state_probability": [1, 2],
}


def default_config():
    return dict(
        schema_version="foundation/v1",
        profile="connected_sampler_v1",
        seed=20261016,
        target_count=64,
        max_attempts=10000,
        budget_ns=300_000_000_000,
        workers=4,
        chunk_size=8,
        priors=copy.deepcopy(PRIORS),
    )


class ConnectedSampler:
    def __init__(self, config):
        from .generic_programs import validate_config, default_config as old_default

        translated = dict(config, profile="generic_sampler_v1", priors=old_default()["priors"])
        validate_config(translated)
        if (
            set(config) != set(default_config())
            or config["profile"] != "connected_sampler_v1"
            or config["priors"] != PRIORS
        ):
            raise ValueError("changed connected sampler priors")
        self.config = copy.deepcopy(config)
        self.config_hash = digest({k: config[k] for k in ("schema_version", "profile", "priors")})
        self.revision = compute_file_hash(Path(__file__))
        self.rng_identity = "connected-counter-parameter/v1"

    def sample(self, counter):
        if type(counter) is not int or not 0 <= counter < self.config["max_attempts"]:
            raise ValueError("sample counter outside frozen bounds")
        structure, variant = divmod(counter, PRIORS["variants"])
        stage = structure % 4
        seed = int(
            digest([self.revision, self.config_hash, self.config["seed"], structure])[7:23], 16
        )
        rng = random.Random(seed)
        parameter_seed = int(digest(["connected-parameter/v1", seed, variant])[7:23], 16)
        parameters = random.Random(parameter_seed)
        index = "local.get $n i64.extend_i32_u i64.const 0 i64.const 0 i64.const 0"
        state = "local.get $v0 local.get $v1 local.get $v2 local.get $v3"
        stores = "local.set $v3 local.set $v2 local.set $v1 local.set $v0"

        def constant():
            return f"i256.const {parameters.randint(-16, 16)}"

        def expression(depth, leaf):
            if depth == 0:
                return leaf
            op = rng.choice(("add", "sub", "mul_scalar"))
            left = expression(depth - 1, leaf)
            if op == "mul_scalar":
                scalar = f"i64.const {parameters.choice(PRIORS['scalar_choices'])}"
                if stage > 0 and rng.randrange(2):
                    # Generic index-valued scalar multiplication permits nonlinear
                    # compositions. In a loop it can depend on the iteration too.
                    scalar = (
                        "local.get $c0 i32.const 1 i32.add i64.extend_i32_u"
                        if stage == 3 and rng.randrange(2)
                        else "local.get $n i64.extend_i32_u"
                    )
                return f"{left} {scalar} i256.mul_scalar"
            # Every subtree has a live path to the output. Branching increases
            # composition diversity without unused registers or dead stores.
            right = expression(depth - 1, index) if depth > 1 and rng.randrange(2) else constant()
            return f"{left} {right} i256.{op}"

        if stage < 2:
            source = expression(rng.randint(1, 2) if stage == 0 else rng.randint(3, 5), index)
        elif stage == 2:
            left, right = expression(rng.randint(1, 3), index), expression(rng.randint(1, 3), index)
            modulus = parameters.randint(2, 7)
            threshold = parameters.randrange(modulus)
            source = (
                f"local.get $n i32.const {modulus} i32.rem_u i32.const {threshold} i32.le_s "
                f"( if ( result i64 i64 i64 i64 ) ( then {left} ) ( else {right} ) )"
            )
        else:
            coupled = bool(rng.randrange(2))
            other = "local.get $v4 local.get $v5 local.get $v6 local.get $v7"
            other_stores = "local.set $v7 local.set $v6 local.set $v5 local.set $v4"
            temporary = "local.get $v8 local.get $v9 local.get $v10 local.get $v11"
            temporary_stores = "local.set $v11 local.set $v10 local.set $v9 local.set $v8"
            leaf = f"{state} {other} i256.{rng.choice(('add', 'sub'))}" if coupled else state
            update = expression(rng.randint(1, 3), leaf)
            initial = f"{constant()} {stores}"
            if coupled:
                initial += f" {constant()} {other_stores}"
                update += f" {temporary_stores} {state} {other_stores} {temporary} {stores}"
            else:
                update += f" {stores}"
            source = (
                f"{initial} i32.const 0 local.set $c0 "
                "( block ( loop local.get $c0 local.get $n i32.ge_s br_if 1 "
                f"local.get $c0 i32.const 100 i32.ge_s br_if 1 {update} "
                "local.get $c0 i32.const 1 i32.add local.set $c0 br 0 ) ) " + state
            )
        tokens = encode_body(source)
        source = decode_body(tokens)
        return dict(
            generator_revision=self.revision,
            generator_config_hash=self.config_hash,
            sample_seed=parameter_seed,
            sample_counter=counter,
            origin="generic_sample",
            language_profile=profile_digests()["language"],
            codec_profile=profile_digests()["codec"],
            canonical_source=source,
            body_tokens=tokens,
            statistics=dict(
                stage=stage,
                structure_counter=structure,
                parameter_variant=variant,
                connected_dataflow=True,
                body_tokens=len(tokens),
            ),
            sampling_failure=None,
        )
