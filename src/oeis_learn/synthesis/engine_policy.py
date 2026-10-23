"""Fixed engine policy shared with Rust; no runtime override or resource escalation."""

import json
from importlib.resources import files

from .contracts import identity


def production_policy():
    return json.loads(files(__package__).joinpath("engine-policy.json").read_text())


def engine_configuration(guest_sha256, *, zero_control=False):
    policy = production_policy()
    if zero_control:
        policy.update(
            version="bigint-engine-policy/zero-control-v1", dynamic_memory_reserved_for_growth=0
        )
    manifest = dict(policy)
    manifest.update(
        version="bigint-engine-configuration/v2",
        policy_version=policy["version"],
        policy_sha256=identity(policy),
        purpose="qualification-zero-growth-control" if zero_control else "production",
        runtime_wasm_sha256=guest_sha256,
    )
    return dict(manifest=manifest, sha256=identity(manifest))
