"""Shared closed learner-row validation for preparation and installed training."""
from .contracts import seal

GENERATOR_LOOKS = (0, 1_000_000, 4_000_000)


def validate_row(row, *, native_verified, short=False):
    from oeis_learn.decoder.bigint_codec import encode_body

    fields = {
        "version",
        "identity",
        "ordinal",
        "draw_identity",
        "source_policy",
        "source",
        "structure",
        "behavior",
        "indices",
        "values",
        "target_source",
        "tokens",
        "target_disposition",
        "membership",
        "native_verified",
        "control_events",
        "branch_events",
        "operations",
        "detector_limit",
    }
    if set(row) != fields:
        raise ValueError("closed training row; evaluation metadata excluded")
    if (
        seal(row) != row
        or row.get("version") != "progressive-training-row/v1"
        or row.get("membership") != "training"
    ):
        raise ValueError("sealed training rows only")
    if native_verified and row.get("native_verified") is not True:
        raise ValueError("native verification missing")
    if (
        len(row["values"]) != (4 if short else 100)
        or len(row["indices"]) != (4 if short else 100)
        or encode_body(row["target_source"]) != row["tokens"]
    ):
        raise ValueError("complete target/codec binding required")
    return row


def first_divergence(positive, negative, positives):
    """The negative logit must be available at the positive's teacher-forced context."""
    for i, (good, bad) in enumerate(zip(positive, negative)):
        if good != bad:
            if any(seq[:i + 1] == negative[:i + 1] for seq in positives):
                return None
            return dict(position=i, positive=good, negative=bad)
    return None


def held_out_probes(layout):
    return {probe["id"] for probe in layout["probes"].values()}


def verified_training_pairs(shards, *, excluded=()):
    """Mine only the registered strict training stream for same-index, prefix-matching alternatives."""
    from collections import defaultdict
    import re
    from oeis_learn.decoder.bigint_codec import decode_body
    from oeis_learn.generalization.targets import ground
    from oeis_learn.synthesis.reference import evaluate
    from oeis_learn.synthesis.reference_parser import parse
    from .shared_support import validated_rows

    rows = validated_rows(shards, native_verified=True)
    groups = defaultdict(list)
    for row in rows.values():
        groups[tuple(row["indices"]), tuple(row["values"][:20])].append(row)
    pairs = []
    for group in groups.values():
        if len({tuple(r["values"][20:]) for r in group}) < 2:
            continue
        verified = []
        for row in sorted(group, key=lambda r: r["draw_identity"]):
            decoded = decode_body(row["tokens"])
            target = parse(decoded, allow_parameters=True)
            if target != parse(row["target_source"], allow_parameters=True):
                raise ValueError("decoded training target differs")
            if "param.get" in row["target_source"]:
                # Preparation numbers literal occurrences in source order.
                constants = re.findall(r"\(int (-?(?:0|[1-9][0-9]*))\)", row["source"])
                assignment = {f"p{i}": v for i, v in enumerate(constants)}
                decoded = ground(decoded, assignment)
            if parse(decoded) != parse(row["source"]):
                raise ValueError("training target grounding differs from the verified program")
            result = evaluate(decoded, row["indices"])
            if result.outcome != "ok":
                continue
            if list(result.values) != row["values"]:
                raise ValueError("independent training continuation differs")
            verified.append(row)
        for pos in verified:
            if pos["draw_identity"] in excluded:
                continue
            same = [r["tokens"] for r in group if r["values"] == pos["values"]]
            for neg in verified:
                if neg["draw_identity"] in excluded or neg["values"] == pos["values"]:
                    continue
                label = first_divergence(pos["tokens"], neg["tokens"], same)
                if label is not None:
                    pairs.append(dict(positive_row=pos["draw_identity"], negative_row=neg["draw_identity"],
                                      family=pos["structure"], label=label))
                    break
    return dict(eligible_rows=len(pairs), eligible_families=len({p["family"] for p in pairs}),
                positions=len({p["label"]["position"] for p in pairs}), pairs=pairs)

def pair_seeds(registration, ordinal):
    from .contracts import digest
    import re
    if (not isinstance(registration, str) or not re.fullmatch("sha256:[0-9a-f]{64}", registration)
            or type(ordinal) is not int or ordinal not in (0, 1)):
        raise ValueError("registered generator pair and parent required")
    return {name: int(digest([registration, name, ordinal])[7:15], 16)
            for name in ("initialization", "training_families")}


def matched_stream(shards, layout, registration, ordinal, *, law=None):
    """Commit one outcome-independent cyclic order and its actual verified labels."""
    from .contracts import digest
    from .shared_support import validated_rows

    seeds = pair_seeds(registration, ordinal)
    if law is not None:
        from .generator import validate_policy
        if validate_policy(law)["seed"] != seeds["training_families"]:
            raise ValueError("mechanical draw law belongs to another paired seed")
    rows = validated_rows(shards, native_verified=True)
    excluded = held_out_probes(layout)
    if not excluded <= rows.keys():
        raise ValueError("missing held-out training probe")
    order = sorted(rows.keys() - excluded, key=lambda key: (digest([seeds["training_families"], key]), key))
    if not order:
        raise ValueError("empty matched training stream")
    support = verified_training_pairs(shards, excluded=excluded)
    return seal(dict(version="round028-matched-stream/v1", registration=registration,
                     pair=ordinal, seeds=seeds, **({"law":law} if law is not None else {}),
                     sources=sorted({r["source_policy"] for r in rows.values()}),
                     row_identities=digest([rows[key]["identity"] for key in order]),
                     order=order, **support))


def matched_batch(stream, cursor, count):
    """Address a full update by its committed example cursor on either arm."""
    if type(cursor) is not int or cursor < 0 or type(count) is not int or count < 1:
        raise ValueError("complete matched batch coordinates required")
    order = stream["order"]
    return [order[(cursor + offset) % len(order)] for offset in range(count)]


def exposure_at(stream, rows, updates, size=32):
    """Independently replay committed positive IDs and token counts without running a model."""
    from .contracts import digest
    if type(updates) is not int or updates < 0 or type(size) is not int or size != 32:
        raise ValueError("complete matched updates of 32 required")
    consumed, tokens, final = digest(None), 0, 0
    for update in range(updates):
        batch = matched_batch(stream, size * update, size)
        final = sum(len(rows[key]["tokens"]) for key in batch)
        if not final:
            raise ValueError("empty registered decoder target")
        tokens += final
        consumed = digest([consumed, batch])
    return dict(order_digest=consumed, useful_tokens=tokens, last_update_tokens=final,
                updates=updates, cursor=updates * size)


def matched_milestone(package, rows, checkpoint, restored, useful, target):
    """Bind a spending observation to the complete private recovery cut."""
    from .contracts import digest
    stream = package["matched_generator"]
    state = restored["states"]
    selection = state["selection"]
    ledger = restored["ledger"]
    if type(target) is not int or target not in GENERATOR_LOOKS:
        raise ValueError("registered generator token look required")
    expected = exposure_at(stream, rows, ledger.updates)
    if (selection["matched_stream"] != stream["identity"] or digest(selection) != restored["cuts"]["selection"]
            or expected["order_digest"] != restored["cuts"]["consumed"]
            or expected["useful_tokens"] != useful or state["data"]["cursor"] != ledger.updates
            or state["scheduler"]["controller"]["cursor"] != expected["cursor"]
            or useful + restored["discarded_tokens"] != ledger.consumed
            or state["optimizer"]["policy"] != package["update_policy"]["identity"]):
        raise ValueError("checkpoint disagrees with the independent matched exposure replay")
    return seal(dict(version="round028-generator-milestone/v1", parent=checkpoint["identity"],
                     registration=stream["registration"], pair=stream["pair"], seeds=stream["seeds"],
                     arm=package["update_policy"]["objective"], stream=stream["identity"],
                     initial=selection["initial_weights"], codec=package["profile"]["codec"],
                     common_policy=digest({
                         "update":{k:v for k,v in package["update_policy"].items()
                                   if k not in {"objective", "identity", "model_profile"}},
                         "model":{k:v for k,v in package["profile"]["model"].items() if k != "seed"},
                         "inputs":package["profile"]["inputs"],
                     }), target=target,
                     charged=ledger.consumed, discarded=restored["discarded_tokens"],
                     overshoot=max(0, ledger.consumed-target), **expected))


def matching_proof(tokens, donor, *, client=None, replay=None):
    """Classify a donor under the *decoded* target's live grounding policy."""
    from oeis_learn.decoder.bigint_codec import decode_body, encode_body, fitting_policy, TOKEN_TO_ID
    from oeis_learn.synthesis.evidence import classify_scope
    from oeis_learn.synthesis.reference import evaluate
    from oeis_learn.synthesis.replay import replay_native
    from .contracts import digest

    source = decode_body(tokens)
    if encode_body(source) != list(tokens):
        raise ValueError("matching target did not round-trip")
    policy = fitting_policy(source)
    binding = dict(target=digest(tokens), prompt=donor.sha256, policy=policy)
    # No integer or parameter syntax means there is no possible adjustable coefficient.
    if not {TOKEN_TO_ID["int"], TOKEN_TO_ID["param.get"]}.intersection(tokens):
        result = evaluate(source, list(donor.indices))
        outcome = "unsupported" if result.outcome != "ok" else (
            "equivalent" if tuple(result.values) == donor.values else "eligible")
        return seal(dict(**binding, status=outcome, proof="independent_fixed_target"))
    if client is None:
        return seal(dict(**binding, status="needs_native_replay", proof=None))
    fit = client.fit(source, donor, policy=policy)
    native = fit.get("native") or {}
    outcome = fit["outcome"]
    proof = None
    if outcome == "unsat_in_scope":
        if classify_scope(native) != "unsat_in_scope" or native.get("coefficient_bits") != 4096:
            status = "partial_scope"
        else:
            proof = dict(indices=list(donor.indices), values=list(donor.values), profile="bigint-body/v1",
                         original_source=source, policy=policy, result=fit,
                         runtime=client.runtime_identity("bigint-body/v1"), reference=client.reference_sha256)
            proof["identity_sha256"] = digest(proof)
            status = "eligible" if (replay or replay_native)(proof) else "unverified"
    elif outcome == "verified_solution":
        status = "regroundable"
    elif outcome in {"resource_exhausted", "unknown", "timeout"}:
        status = "unfinished"
    else:
        status = "unsupported"
    return seal(dict(**binding, status=status, proof=proof))


def matching_pairs(shards, order, *, seed, client=None):
    """Freeze a within-band derangement. All positives retain their original stream slots."""
    from collections import Counter, defaultdict
    from .contracts import digest
    from .shared_support import validated_rows
    from oeis_learn.synthesis.admission import VisiblePrompt
    from oeis_learn.synthesis.contracts import canonical_integer

    rows = validated_rows(shards, native_verified=True)
    if len(set(order)) != len(order) or not set(order) <= rows.keys():
        raise ValueError("verified unique training support required")
    bands = defaultdict(list)
    for key in order:
        values = [canonical_integer(v) for v in rows[key]["values"][:20]]
        signs = tuple((v > 0) - (v < 0) for v in values)
        bands[len(values), max(abs(v).bit_length() for v in values), signs].append(key)
    statuses = Counter()
    pairs = {}
    for members in bands.values():
        # Contiguous identical visible values and rotation by the largest group size
        # yield a complete differing-value permutation whenever one exists.
        groups = defaultdict(list)
        for key in members:
            groups[tuple(rows[key]["values"][:20])].append(key)
        largest = max(map(len, groups.values()))
        ordered = [k for values in sorted(groups, key=lambda v: digest([seed, v]))
                   for k in sorted(groups[values], key=lambda k: digest([seed, k]))]
        for i, key in enumerate(ordered):
            donor = ordered[(i + largest) % len(ordered)]
            row = rows[key]
            values = rows[donor]["values"][:20]
            if values == row["values"][:20]:
                statuses["no_differing_donor"] += 1
                continue
            proof = matching_proof(row["tokens"], VisiblePrompt(tuple(row["indices"][:20]), tuple(values)),
                                   client=client)
            statuses[proof["status"]] += 1
            if proof["status"] == "eligible":
                pairs[key] = dict(donor=donor, family=row["structure"], proof=proof)
    return seal(dict(version="round029-training-pairs/v1", seed=seed, order=digest(order),
                     positive_rows=len(order), eligible_rows=len(pairs), dispositions=dict(statuses),
                     eligible_families=len({rows[k]["structure"] for k in pairs}), pairs=pairs))
