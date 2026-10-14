import json

import pytest

from oeis_learn.data.generic_programs import GenericSampler, default_config, validate_config
from oeis_learn.data.program_admission import validate_sample
from oeis_learn.fresh.admission_sampling import SamplerShards
from tests.unit.test_fresh_cli import cli, config


def test_first_shard_keeps_existing_stream_and_later_shards_replay():
    cfg = default_config()
    series = SamplerShards(cfg, 64)
    legacy = GenericSampler(cfg)
    assert series.limit == 640000
    for counter in (0, 99, 9999):
        sampler, local = series.select(counter)
        assert sampler.sample(local) == legacy.sample(counter)
    sampler, local = series.select(10000)
    assert local == 0 and sampler.config["seed"] != cfg["seed"]
    # Each packet remains independently replayable by the unchanged strict sampler.
    packet = sampler.sample(local)
    assert GenericSampler(sampler.config).sample(local) == packet
    assert validate_sample(packet, sampler) == validate_sample(
        packet, GenericSampler(sampler.config)
    )
    replay, local2 = SamplerShards(cfg, 64).select(10000)
    assert replay.sample(local2) == packet
    seeds = {series.select(n * 10000)[0].config["seed"] for n in range(64)}
    assert len(seeds) == 64
    with pytest.raises(ValueError, match="outside"):
        series.select(640000)
    with pytest.raises(ValueError, match="max_attempts"):
        validate_config(dict(cfg, max_attempts=10001))


@pytest.mark.parametrize("shards", [0, 65, True, 1.5, "64"])
def test_shard_bounds_reject(shards):
    with pytest.raises(ValueError, match="sampling_shards"):
        SamplerShards(default_config(), shards)


@pytest.mark.parametrize("shards", [1, 64, 0, 65, True, "64"])
def test_workflow_shard_configuration_is_bounded(tmp_path, shards):
    cfg = config()
    cfg["archive"]["sampling_shards"] = shards
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    if type(shards) is int and 1 <= shards <= 64:
        assert cli.configuration(path)["archive"]["sampling_shards"] == shards
    else:
        with pytest.raises(ValueError):
            cli.configuration(path)
