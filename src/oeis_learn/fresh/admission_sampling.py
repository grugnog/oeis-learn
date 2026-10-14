"""Bounded independent generic streams for fresh diagnostic admission."""

from oeis_learn.data.generic_programs import GenericSampler
from oeis_learn.experiments.artifacts import compute_canonical_digest


class SamplerShards:
    def __init__(self, config, shards=1):
        if type(shards) is not int or not 1 <= shards <= 64:
            raise ValueError("sampling_shards: exact integer in 1..64 required")
        self.base = GenericSampler(config)
        self.shards = shards
        self.limit = self.base.config["max_attempts"] * shards
        self.samplers = {0: self.base}

    def select(self, counter):
        if type(counter) is not int or not 0 <= counter < self.limit:
            raise ValueError("global admission counter outside frozen bounds")
        shard, local = divmod(counter, self.base.config["max_attempts"])
        if shard not in self.samplers:
            seed = int(
                compute_canonical_digest(
                    ["fresh-sampler-shard/v1", self.base.config["seed"], shard]
                )[7:23],
                16,
            )
            self.samplers[shard] = GenericSampler(dict(self.base.config, seed=seed))
        return self.samplers[shard], local
