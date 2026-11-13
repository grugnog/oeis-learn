"""Fixed-bin, eight-update adapter over the existing Graves EXP3.S kernel."""

import copy
import random

from oeis_learn.generalization.curriculum import Exp3S, rng_restore, rng_state

from .contracts import digest
from .metrics import _seed


def _layout(layout):
    if not isinstance(layout, dict) or set(layout) != {"bins", "probes"}:
        raise ValueError("closed fixed-bin layout required")
    bins = layout["bins"]
    probes = layout["probes"]
    if (
        not isinstance(bins, list)
        or not 2 <= len(bins) <= 32
        or len(set(bins)) != len(bins)
        or not {"familiar", "frontier"} <= set(bins)
        or set(probes) != set(bins)
        or any(not isinstance(b, str) or not b for b in bins)
    ):
        raise ValueError("fixed bins and complete probe roster required")
    for p in probes.values():
        if (
            not isinstance(p, dict)
            or set(p) != {"role", "id"}
            or p["role"] != "training"
            or not isinstance(p["id"], str)
            or not p["id"]
        ):
            raise ValueError("training-only fixed probes required")
    if len({p["id"] for p in probes.values()}) != len(bins):
        raise ValueError("disjoint training probes required")
    return copy.deepcopy(layout)


class Controller:
    def __init__(self, layout, *, seed):
        self.layout = _layout(layout)
        self.seed = _seed(seed)
        self.kernel = Exp3S(self.layout["bins"], seed=seed)
        self.rng = random.Random(seed)
        self.updates = 0
        self.charged_tokens = 0
        self.discarded_tokens = 0
        self.pending = None
        self.rewards = []
        self.block_position = 0
        self.shard = 0
        self.cursor = 0
        self.rejection_watermark = 0

    def probabilities(self):
        base = self.kernel.probabilities()
        return {
            b: 0.7 * p + (0.2 if b == "familiar" else 0.1 if b == "frontier" else 0)
            for b, p in zip(self.layout["bins"], base)
        }

    def select(self):
        if self.block_position == 8:
            raise ValueError("training probe feedback required before another block")
        if self.pending is None:
            selected = self.kernel.choose()
            p = self.probabilities()
            u = self.rng.random()
            total = 0
            chosen = self.layout["bins"][-1]
            for b in self.layout["bins"]:
                total += p[b]
                if u < total:
                    chosen = b
                    break
            # The shared kernel update must use the actual mixture probability.
            self.kernel.pending = dict(selected, family=chosen, probability=p[chosen])
            self.pending = {"bin": chosen, "probability": p[chosen], "action": selected["action"]}
        return copy.deepcopy(self.pending)

    def commit(self, *, tokens, success):
        if (
            self.pending is None
            or self.block_position >= 8
            or type(tokens) is not int
            or tokens <= 0
            or type(success) is not bool
        ):
            raise ValueError("reserved complete logical update required")
        self.charged_tokens += tokens
        if success:
            self.updates += 1
            self.block_position += 1
        else:
            self.discarded_tokens += tokens

    def feedback(self, *, role, probe, gain):
        if (
            role != "training"
            or self.pending is None
            or self.block_position != 8
            or probe != self.layout["probes"][self.pending["bin"]]["id"]
        ):
            raise ValueError("matched training probe at eight-update boundary required")
        # Exp3S.observe implements the shared Graves update. Restore's historical
        # selection check remains unchanged; this adapter explicitly restores its
        # mixture probability rather than passing new states to old validators.
        self.kernel.observe(
            self.kernel.pending, gain, event_id=f"feedback/{self.pending['action']}", role=role
        )
        self.rewards.append(
            {"action": self.pending["action"], "bin": self.pending["bin"], "gain": gain}
        )
        self.rewards = self.rewards[-64:]
        self.pending = None
        self.block_position = 0

    def set_stream_cut(self, *, shard, cursor, rejection_watermark):
        for v in (shard, cursor, rejection_watermark):
            if type(v) is not int or v < 0:
                raise ValueError("nonnegative stream cut required")
        if (shard, cursor) < (
            self.shard,
            self.cursor,
        ) or rejection_watermark < self.rejection_watermark:
            raise ValueError("stream watermark cannot rewind")
        self.shard, self.cursor, self.rejection_watermark = shard, cursor, rejection_watermark

    def snapshot(self):
        return {
            "version": "progressive-fixed-bin-controller/v1",
            "layout": digest(self.layout),
            "seed": self.seed,
            "kernel": self.kernel.snapshot(),
            "rng": rng_state(self.rng),
            "rewards": copy.deepcopy(self.rewards),
            "updates": self.updates,
            "charged_tokens": self.charged_tokens,
            "discarded_tokens": self.discarded_tokens,
            "pending": copy.deepcopy(self.pending),
            "block_position": self.block_position,
            "shard": self.shard,
            "cursor": self.cursor,
            "rejection_watermark": self.rejection_watermark,
        }

    @classmethod
    def restore(cls, state, layout):
        c = cls(layout, seed=state["seed"])
        if (
            state.get("version") != "progressive-fixed-bin-controller/v1"
            or state["layout"] != digest(c.layout)
            or set(state) != set(c.snapshot())
        ):
            raise ValueError("fixed controller policy changed on resume")
        k = copy.deepcopy(state["kernel"])
        pending = k["pending"]
        k["pending"] = None
        c.kernel = Exp3S.restore(k)
        c.kernel.pending = pending
        c.rng = rng_restore(state["rng"])
        c.pending = copy.deepcopy(state["pending"])
        c.rewards = copy.deepcopy(state["rewards"])
        if not isinstance(c.rewards, list) or len(c.rewards) > 64:
            raise ValueError("bounded reward history required")
        for name in (
            "updates",
            "charged_tokens",
            "discarded_tokens",
            "block_position",
            "shard",
            "cursor",
            "rejection_watermark",
        ):
            value = state[name]
            if type(value) is not int or value < 0:
                raise ValueError("invalid controller counters")
            setattr(c, name, value)
        if not 0 <= c.block_position <= 8 or c.discarded_tokens > c.charged_tokens:
            raise ValueError("invalid update accounting")
        if (c.pending is None) != (pending is None) or (c.updates - c.block_position) % 8:
            raise ValueError("inconsistent block cut")
        if c.pending is not None and (
            pending is None
            or c.pending["bin"] != pending["family"]
            or c.pending["probability"] != c.probabilities()[c.pending["bin"]]
            or pending["probability"] != c.pending["probability"]
        ):
            raise ValueError("mixture selection differs on resume")
        return c
