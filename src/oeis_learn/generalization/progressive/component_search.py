"""Bounded typed component proposals for slots 4–7 of the existing portfolio.

Two deterministic typed AST candidates per slot, ranked by the unchanged decoder
on fully expanded tokens. No truth, witness, new vocabulary or fitting authority.
"""

import copy
import math
import time

from .components import expand, validate_library

# Conservative pre-work charge per expansion, including template validation.
EXPANSION_WORK = 65 * 4 * 4096
SLOT_TOKENS = 2 * 2 * 1024


class Proposals:
    def __init__(self, library, *, decoder=None, memory=None, charge=None):
        self.library = copy.deepcopy(validate_library(library))
        self.components = sorted(self.library["payload"]["components"], key=lambda c: c["id"])
        self.decoder, self.memory, self.charge = decoder, memory, charge
        if (decoder is None) != (memory is None):
            raise ValueError("model scoring requires decoder and visible memory together")
        self.used = set()
        self.report = dict(
            library=library["identity"],
            proposals=[],
            expanded_nodes=0,
            expanded_tokens=0,
            expansion_work_reserved=0,
            scoring_tokens=0,
            elapsed_us=0,
            scientific_status="UNMEASURED",
            discovery_verification=library["payload"]["verification"],
        )

    def __call__(self, position, deadline):
        if type(position) is not int or position not in range(4, 8) or position in self.used:
            raise ValueError("component proposals own exactly four nonrepeatable slots")
        self.used.add(position)
        start = time.monotonic()
        empty = dict(
            source="",
            tokens=[],
            complete=False,
            generation_seconds=0.0,
            generation_batch_size=1,
            failure="component_search_exhausted",
        )
        if not self.components:
            return empty
        from oeis_learn.decoder.bigint_codec import encode_body, decode_body, BOS_ID, PAD_ID

        candidates = []
        for variant in range(2):
            if time.monotonic() >= deadline:
                break
            if self.charge:
                self.charge(tokens=2048, native_work=EXPANSION_WORK)
            self.report["expansion_work_reserved"] += EXPANSION_WORK
            c = self.components[(position - 4) % len(self.components)]
            args = []
            for i, a in enumerate(c["arguments"]):
                # Ordinary typed AST operands; preserve their order and every call.
                choices = {
                    "bigint": ["$index", ["int", "1"], ["add", "$index", ["int", "1"]]],
                    "bool": [["true"], ["false"]],
                    "control": [["control", "0"], ["control", "1"]],
                }
                pool = choices[a["type"]]
                args.append(
                    copy.deepcopy(
                        pool[(i + variant + (position - 4) // len(self.components)) % len(pool)]
                    )
                )
            expression = ["call", c["id"], *args]
            if variant and c["returns"] == "bigint":
                expression = ["add", expression, ["int", "1"]]
            row = dict(position=position, variant=variant, component=c["id"], expression=expression)
            self.report["proposals"].append(row)
            try:
                value = expand(self.library, expression)
                tokens = encode_body(value["source"])
                value["source"] = decode_body(tokens)
            except ValueError as exc:
                row.update(status="rejected", reason=str(exc))
                continue
            self.report["expanded_nodes"] += value["expanded_nodes"]
            self.report["expanded_tokens"] += len(tokens)
            score = 0.0
            if self.decoder is not None and time.monotonic() < deadline:
                import torch

                self.report["scoring_tokens"] += len(tokens)
                with torch.no_grad():
                    ids = torch.tensor([[BOS_ID, *tokens]], device=self.memory.device)
                    logits = self.decoder(
                        ids[:, :-1], self.memory, tgt_key_padding_mask=(ids[:, :-1] == PAD_ID)
                    )
                    score = float(
                        torch.log_softmax(logits.float(), -1)
                        .gather(-1, ids[:, 1:].unsqueeze(-1))
                        .sum()
                        .item()
                    )
            row.update(
                source=value["source"],
                expanded_nodes=value["expanded_nodes"],
                expanded_bytes=value["expanded_bytes"],
                expanded_tokens=len(tokens),
                model_score=score,
                status="eligible",
            )
            if not math.isfinite(score):
                raise ValueError("nonfinite component model score")
            if time.monotonic() < deadline:
                candidates.append((score, value["source"], tokens))
        elapsed = time.monotonic() - start
        self.report["elapsed_us"] += int(elapsed * 1e6)
        if not candidates or time.monotonic() >= deadline:
            return dict(empty, generation_seconds=elapsed)
        _, source, tokens = max(candidates, key=lambda v: (v[0], v[1]))
        return dict(
            source=source,
            tokens=tokens,
            complete=True,
            generation_seconds=elapsed,
            generation_batch_size=1,
        )


def search_input(store, library):
    """Export a committed cut with original cost/proof evidence, no source solutions."""
    import json
    from .contracts import load, seal, verify_refs

    if library not in store.committed():
        raise ValueError("original committed library required")
    commits = [load(p) for p in sorted(store.path.glob("commit-*.json"))]
    commit = next(
        c for c in commits if json.loads(verify_refs(store.path, [c["library"]])[0]) == library
    )
    value = seal(
        dict(
            version="progressive-component-search/v1",
            library=library,
            commit=commit,
            discovery_work=[json.loads(s) for s in store.work.read_text().splitlines()],
            proofs=[
                json.loads(b)
                for b in verify_refs(store.path, library["payload"]["verification"]["proofs"])
            ],
        )
    )
    return validate_input(value)


def validate_input(value):
    import hashlib
    import json
    from .contracts import seal

    if (
        set(value) != {"identity", "version", "library", "commit", "discovery_work", "proofs"}
        or value["version"] != "progressive-component-search/v1"
        or seal(value) != value
    ):
        raise ValueError("closed committed component search input required")
    library = validate_library(value["library"])
    commit = value["commit"]
    if (
        seal(commit) != commit
        or commit.get("status") != "COMMITTED"
        or commit.get("version") != "progressive-library-commit/v1"
    ):
        raise ValueError("original committed component cut required")

    def matches(item, ref):
        raw = (json.dumps(item, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        return (
            len(raw) == ref["bytes"]
            and "sha256:" + hashlib.sha256(raw).hexdigest() == ref["sha256"]
        )

    if not matches(library, commit["library"]):
        raise ValueError("committed library bytes differ")
    refs = library["payload"]["verification"]["proofs"]
    if len(refs) != len(value["proofs"]) or any(
        not matches(p, r) for p, r in zip(value["proofs"], refs)
    ):
        raise ValueError("original expansion verification evidence required")
    work = value["discovery_work"]
    if not isinstance(work, list) or not work or len(work) > 131072:
        raise ValueError("bounded original discovery work required")
    for i, row in enumerate(work):
        if (
            set(row) != {"identity", "ordinal", "phase", "key", "units"}
            or seal(row) != row
            or row["ordinal"] != i
            or type(row["units"]) is not int
            or row["units"] < 1
        ):
            raise ValueError("original spent discovery work required")
    return value
