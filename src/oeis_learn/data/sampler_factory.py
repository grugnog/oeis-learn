"""Explicit dispatch without changing the historical sampler's source identity."""

from pathlib import Path
from .generic_programs import GenericSampler


def sampler(config):
    if config.get("profile") == "connected_sampler_v1":
        from .bootstrap_programs import ConnectedSampler

        return ConnectedSampler(config)
    return GenericSampler(config)


def load_sampler_config(path):
    import yaml
    from oeis_learn.experiments.config import _UniqueLoader

    with Path(path).open() as stream:
        return sampler(yaml.load(stream, Loader=_UniqueLoader)).config
