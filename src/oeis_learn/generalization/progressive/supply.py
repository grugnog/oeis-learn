"""Outcome-blind finite refreshed-support feasibility at the frozen token horizon."""

from .contracts import seal


def plan(package, target, *, arm=None):
    arm = package.get("arm") if arm is None else arm
    if arm == "static" or package.get("task_profile") == "combinatorial-short/v1":
        return None
    if arm not in {"refreshed", "replay", "continued_synthetic", "library", "no_library", "with_source", "without_source"}:
        raise ValueError("frozen support arm required")
    shards = package["shards"]
    batch = package["update_policy"]["effective_batch"]
    cadence = package["refresh_updates"]
    if type(target) is not int or target < 0 or type(cadence) is not int or cadence < 1:
        raise ValueError("frozen supply horizon required")
    # The smallest possible whole-example debit gives the conservative maximum
    # number of committed updates before reaching the milestone. Discarded work
    # increases charged exposure and cannot purchase additional useful updates.
    minimum = min(len(r["tokens"]) for shard in shards for r in shard)
    if not minimum:
        raise ValueError("nonempty target token debit required")
    updates = (target + minimum * batch - 1) // (minimum * batch)
    needed = (updates + cadence - 1) // cadence
    if needed > len(shards):
        raise ValueError(
            f"refreshed preparation deficit: {needed} frozen shards required, {len(shards)} supplied"
        )
    return seal(
        dict(
            version="progressive-support-supply/v1",
            target_tokens=target,
            minimum_update_tokens=minimum * batch,
            maximum_updates=updates,
            required_shards=needed,
            available_shards=len(shards),
            refresh_updates=cadence,
        )
    )


def shard_index(arm, bin_name, updates, cadence, count):
    if (
        # Replay and component arms retain the same refreshed synthetic support
        # cursor; their separate mixers select the additional training rows.
        arm not in {"static", "refreshed", "replay", "continued_synthetic", "library", "no_library"}
        or type(updates) is not int
        or updates < 0
        or type(cadence) is not int
        or cadence < 1
        or type(count) is not int
        or count < 1
    ):
        raise ValueError("frozen deterministic support cursor required")
    index = 0 if arm == "static" or bin_name == "familiar" else updates // cadence
    if index >= count:
        raise TimeoutError("preallocated refreshed support exhausted; no static fallback")
    return index
