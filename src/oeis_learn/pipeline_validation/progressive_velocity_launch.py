"""Read paid training and publication measurements for the supported round029 launch."""

import hashlib
import json
import os
from pathlib import Path
import zipfile

from oeis_learn.generalization.progressive.contracts import load, seal
from oeis_learn.generalization.progressive.program_search_costing import trajectory_overheads
from oeis_learn.generalization.progressive.velocity_records import SESSION, operation_key
from .identity import confined
from . import progressive_velocity_broad_host as broad
from . import progressive_velocity_checkpoint_host as checkpoints
from .progressive_pilot_capacity import owned_allocations, velocity_mounts
from .progressive_velocity_donor import ARCHIVES, _archives


def paid_training(root, fresh):
    folder = confined(root, "run-state/discovery/" + SESSION)
    pointer = checkpoints.origin(folder)
    area = confined(folder, "qualification/" + pointer["grant"][7:])
    grant = load(confined(area, "grant.json"))
    out = broad.attempt(area, broad.coordinates(grant, "measure"), grant)
    producer = load(confined(out, "producer.json"))
    request = load(confined(area, "device-request.json"))
    if (
        fresh != seal(fresh)
        or pointer["receipt"] != fresh["identity"]
        or producer != seal(producer)
        or producer["identity"] != fresh["device"]["producer"]
        or request != seal(request)
        or producer["request"] != request["identity"]
    ):
        raise ValueError("original fresh optimizer samples and request required")
    samples = producer["matrix"]["methods"]
    starts = {m: samples[m]["startup_us"] for m in fresh["device"]["methods"]}
    counts = {m: sum(c["updates_at_look"] for c in request["design"]["cells"][m]) for m in starts}
    return starts, counts


def release_measurement(root, policy, info):
    """Read the source-pinned successful release already checked by donor preflight."""
    if info["archives"]["release"].get("reader") != policy["release_reader"]:
        return None
    paths, _ = _archives(root, policy)
    with paths["release"].open("rb") as raw:
        if hashlib.file_digest(raw, "sha256").hexdigest() != ARCHIVES["release"][1]:
            raise ValueError("original release archive changed")
        raw.seek(0)
        with zipfile.ZipFile(raw) as archive:
            report = json.loads(archive.read("release.json"))
    if report != seal(report) or report["candidate"] != policy["release_reader"]:
        raise ValueError("original release measurement or pinned reader differs")
    result = report["stages"]["scale"]["result"]
    return dict(
        receipt=report["identity"],
        candidate=report["candidate"],
        floor_bytes=report["admission"]["floor_bytes"],
        **{k: result[k] for k in ("corpora", "journal", "peak_allocated_bytes", "peak_inodes")},
    )


def publication_proxy(release):
    """One production-size write/read and ledger replay at the original measured scale."""
    if release is None:
        return None
    corpora, ledger = release["corpora"], release["journal"]
    spans = [c["cold_us"] + c["warm_us"] for c in corpora.values()]
    return dict(
        receipt=release["receipt"],
        corpora={k: dict(c) for k, c in corpora.items()},
        journal=dict(ledger),
        expected_us=(sum(spans) + 1) // 2 + ledger["cold_us"],
        tail_us=max(spans) + ledger["cold_us"],
        scope="One historical-scale write, expanded-member verification and a cold journal replay. "
        "The two original encodings supply the midpoint and slower observed price.",
    )


def checkpoint_owners(cycles, trajectory):
    """Compare all retained treatment cuts and one staged write to current free space."""
    path = "run-state/discovery-learning"
    owners = []
    for method, row in cycles["methods"].items():
        # The measured tree already contains the initial and both updated generations.
        # Charge a whole observed tree per extra hourly generation until a single
        # generation's allocated size is measured. One future write may be pending.
        copies = 1 + trajectory["methods"][method]["hourly_saves"]["tail"]
        observed = {u: row["checkpoint_" + u] for u in ("bytes", "inodes")}
        owners.append(
            dict(
                owner="learner_" + method,
                path=path,
                retained=[],
                observed=observed,
                projected={u: n * copies for u, n in observed.items()},
                revision=cycles["candidate"],
                scope="retained treatment checkpoints",
                harm="loss of a resumable learner cut",
                exhaustion="price one treatment",
                rehearsal="T060 peak coexistence",
            )
        )
    ordinary = next(o for o in owners if o["owner"] == "learner_00")
    staged = dict(ordinary, owner="pending_checkpoint", scope="one concurrent checkpoint write")
    staged["observed"] = {u: 0 for u in ("bytes", "inodes")}
    staged["projected"] = {u: max(o["observed"][u] for o in owners) for u in ("bytes", "inodes")}
    first = dict(ordinary, projected=ordinary["observed"])
    first_stage = dict(staged, projected=first["observed"])
    return [*owners, staged], [first, first_stage]


def owner_floors(root, owners, floor, registered):
    by_device = {row["device"]: row for row in registered or ()}
    floors = []
    for name in sorted({owner["path"] for owner in owners}):
        path = confined(root, name, exists=False)
        while not path.exists():
            path = path.parent
        original = by_device.get(path.stat().st_dev, {})
        floors.append(
            dict(
                path=name,
                bytes=max(floor, original.get("reserved_bytes", 0)),
                inodes=original.get("reserved_inodes", 0),
            )
        )
    return floors


def checkpoint_space(root, cycles, trajectory, floor, registered=()):
    full, prefix = checkpoint_owners(cycles, trajectory)
    floors = owner_floors(root, full, floor, registered)
    full = velocity_mounts(root, full, floors)
    prefix = velocity_mounts(root, prefix, floors)
    return dict(
        checkpoint_receipt=cycles["receipt"],
        wave=full,
        first_ordinary_cut=prefix,
        scope="Four retained checkpoint trees plus a single pending write, with provisional headroom. "
        "The first ordinary cut has its own smaller storage check.",
    )


def settled_area(root, parent, name, identity, *, field="identity"):
    """Locate the unique matching receipt, then check its plan, authorization and debit."""
    from oeis_learn.generalization.progressive.budgets import BudgetJournal
    from oeis_learn.generalization.progressive.velocity_records import (
        physical_root_id,
        validate_grant,
    )

    matches = []
    for path in parent.glob("*/" + name + ".json"):
        row = load(confined(root, path))
        if row.get(field) == identity:
            if row != seal(row):
                raise ValueError("changed preparation record")
            matches.append(path.parent)
    if len(matches) != 1:
        raise ValueError("one matching preparation scope required")
    area = matches[0]
    plan, grant = (load(confined(area, n + ".json")) for n in ("plan", "grant"))
    if plan != seal(plan) or grant["identity"][7:] != area.name:
        raise ValueError("original preparation plan or grant differs")
    auth = load(
        confined(
            root,
            "run-state/discovery-authorizations/" + grant["authorizing_record_ref"][7:] + ".json",
        )
    )
    validate_grant(
        grant, auth, root=physical_root_id(root), parent=plan["identity"], action="qualify"
    )
    state = BudgetJournal.inspect_existing(
        area / "accounting", grant["identity"], grant["resources"]
    )
    original_grant = confined(
        root, "run-state/discovery-authorizations/grant-" + area.name + ".json"
    )
    if grant != load(original_grant) or state["pending"]:
        raise ValueError("preparation has unsettled operations")
    if name == "receipt":
        record = load(confined(area, "receipt.json"))
        if (
            record["plan"] != plan["identity"]
            or record["grant"] != grant["identity"]
            or record["charged"] != state["consumed"]
        ):
            raise ValueError("preparation receipt differs from its original debit")
    return area, plan, grant, state


def role_storage(area, grant, state, coords, *, recovered=None):
    block = confined(area, "operations/" + operation_key(coords, 0)["logical"][7:])
    req = load(confined(area, f"inputs/{coords['method']}/{coords['role']}/request.json"))
    if req != seal(req) or req["plan"] != grant["registration_or_parent_ref"]:
        raise ValueError("role sizing input differs from the original plan")
    # A search may have failed the original verifier. Its settled files still
    # occupy storage. Only a finished search + scorer can supply a whole-block price.
    if recovered is not None:
        reader = dict(coords, role="verify-paid-search")
        replay_block = confined(area, "operations/" + operation_key(reader, 0)["logical"][7:])
        receipt = load(confined(replay_block, "receipt.json"))
        proof = receipt["evidence"]["original"]
        reader_key = f"{operation_key(reader, 0)['logical']}/dispatch/{receipt['attempt']}"
        if (
            receipt != seal(receipt)
            or receipt["coordinates"] != reader
            or receipt["grant"] != grant["identity"]
            or proof["identity"] != recovered
            or reader_key not in state["committed"]
        ):
            raise ValueError("recovered search requires its paid reader and score barrier")
    else:
        proof = load(confined(block, "receipt.json"))
    if (
        proof != seal(proof)
        or proof["coordinates"] != coords
        or proof["request"] != req["identity"]
        or proof["grant"] != grant["identity"]
    ):
        raise ValueError("role sizing receipt differs from the original request")
    attempt = proof["attempt"]
    logical = operation_key(coords, attempt)["logical"]
    if f"{logical}/dispatch/{attempt}" not in state["committed"]:
        raise ValueError("role sizing requires a settled operation")
    return [
        confined(block, f"attempt-{attempt:04d}"),
        confined(area, f"inputs/{coords['method']}/{coords['role']}"),
    ]


def sample_storage(root, wave):
    from .progressive_research_inputs import checked_file

    parent = confined(root, "run-state/discovery/" + SESSION + "/preparation")
    paired, plan, grant, state = settled_area(
        root, parent / "paired-conditioning", "plan", wave["checkpoint_receipt"], field="checkpoint"
    )
    if plan["version"] != "round029-paired-conditioning-prices/v1":
        raise ValueError("paired training-control cost rehearsal required")
    _, bank_plan, _, _ = settled_area(
        root, parent / "matched-bank-prices", "receipt", plan["previous"]
    )
    cpu, cpu_plan, cpu_grant, cpu_state = settled_area(
        root, parent / "isolated-control-prices", "receipt", bank_plan["controls"]
    )
    if any(p["checkpoint"] != wave["checkpoint_receipt"] for p in (bank_plan, cpu_plan)):
        raise ValueError("control sizing belongs to another checkpoint matrix")
    roles = {}
    for method in (
        "template",
        "polynomial",
        "linear_recurrence",
        "ordinary-common",
        "ordinary-full",
        "D-full",
    ):
        control = method in cpu_plan["methods"]
        area, terms, ledger = (cpu, cpu_grant, cpu_state) if control else (paired, grant, state)
        coords = dict(
            study=terms["identity"],
            stage="qualification",
            method=method,
            seed=29 if control else plan["seed"],
            look=4_000_000,
            panel="exposed_development",
            profile="common" if method.endswith("-common") else "full",
            input_mode="true" if control else "paired",
            ordinal=0,
        )
        try:
            search = dict(coords, role="search")
            paths = role_storage(area, terms, ledger, dict(coords, role="score"))
            try:
                paths += role_storage(area, terms, ledger, search)
            except FileNotFoundError:
                # Successful CPU scoring binds the recovered search to its barrier.
                path = confined(area, method + ".json")
                group = load(path)
                if control or group != seal(group) or group["group"] != method:
                    raise ValueError("original recovered group required")
                paths += role_storage(area, terms, ledger, search, recovered=group["search"])
            roles[method] = owned_allocations(paths)
        except FileNotFoundError:
            continue
    # Runtime staging on the cached-image scenario. Use the verified wheel that
    # was mounted in the model role and count both a staged and installed copy.
    model = dict(
        study=grant["identity"],
        stage="qualification",
        method="paired-models",
        seed=plan["seed"],
        look=4_000_000,
        panel="exposed_development",
        profile="models",
        input_mode="paired",
        role="models",
        ordinal=0,
    )
    out = role_storage(paired, grant, state, model)[0]
    inspection = load(confined(out, "container-inspect.json"))[0]
    (wheel,) = [
        confined(root, m["Source"])
        for m in inspection["Mounts"]
        if m["Destination"].endswith(".whl") and m["RW"] is False
    ]
    checked_file(
        root,
        dict(path=str(wheel.relative_to(root)), bytes=wheel.stat().st_size, sha256=plan["wheel"]),
    )
    block = os.statvfs(paired).f_frsize
    with zipfile.ZipFile(wheel) as z:
        members = z.infolist()
    dirs = {p for m in members for p in Path(m.filename).parents}
    allocated = sum((m.file_size + block - 1) // block for m in members) + len(dirs)
    return dict(
        cases=8,
        checkpoint=wave["checkpoint_receipt"],
        grant=grant["identity"],
        runtime=dict(bytes=2 * block * allocated, inodes=2 * (len(dirs) + len(members))),
        roles=roles,
    )


def wave_space(
    root, wave, cycles, trajectory, sample, release, *, floor, updates=None, registered=()
):
    """Provisional cached-image, serial-wave coexistence and its first resumable prefix."""
    from oeis_learn.generalization.budgets import integer

    if sample["checkpoint"] != cycles["receipt"] or sample["cases"] != 8:
        raise ValueError("same checkpoint and eight-family storage sample required")
    full, prefix = checkpoint_owners(cycles, trajectory)
    chunks = dict.fromkeys(("ordinary-common", "ordinary-full", "D-full"), 0)
    ops = wave["operations"]
    for worker, profile, proxy in (
        ("ordinary", "one", "ordinary-common"),
        ("learner", "one", "ordinary-common"),
        ("ordinary", "full", "ordinary-full"),
        ("learner", "full", "ordinary-full"),
        ("D", "full", "D-full"),
    ):
        key = f"{worker}.{profile}.search."
        roles = integer(ops[key + "startup"], 1)
        cases = sum(n for k, n in ops.items() if k.startswith(key) and k != key + "startup")
        per_block, remainder = divmod(cases, roles)
        if remainder or per_block < 1:
            raise ValueError("unmatched per-role wave footprint")
        chunks[proxy] += roles * ((per_block + 7) // 8)
    for method in ("template", "polynomial", "linear_recurrence"):
        chunks[method] = chunks["D-full"]
    for group in ("ordinary-common", "ordinary-full"):
        chunks[group] += len(wave["methods"])
    chunks["D-full"] += 1
    missing = sorted(chunks.keys() - sample["roles"].keys())
    private = {
        u: sum(chunks[g] * integer(v[u], 1) for g, v in sample["roles"].items())
        for u in ("bytes", "inodes")
    }

    def owner(name, path, size, count, scope):
        return dict(
            owner=name,
            path=path,
            observed=dict(bytes=0, inodes=0),
            projected=dict(bytes=size, inodes=count),
            retained=[],
            revision=cycles["candidate"],
            scope=scope,
            harm="unfinished workstation output",
            exhaustion="price the next whole look",
            rehearsal="T060 peak coexistence",
        )

    session = "run-state/discovery/" + SESSION
    if private["bytes"]:
        full.append(
            owner(
                "private_portfolios",
                session,
                private["bytes"],
                private["inodes"],
                "retained search and scorer outputs at the matched eight-case footprint",
            )
        )
    runtime = owner(
        "worker_staging",
        session,
        sample["runtime"]["bytes"],
        sample["runtime"]["inodes"],
        "one serial cached worker installation",
    )
    full.append(runtime)
    prefix.append(runtime)
    if release is None:
        missing.append("release_and_journal")
    else:
        # Use the varied historical transport as a floor. Allow a full private-size
        # source, a staged ZIP and its published copy, rounded in their own units.
        shape = release["corpora"]["varied"]
        size = max(shape["expanded_bytes"], private["bytes"])
        files = max(shape["files"], private["inodes"])
        archive = (size * shape["compressed_bytes"] + shape["expanded_bytes"] - 1) // shape[
            "expanded_bytes"
        ]
        full.append(
            owner(
                "release_staging",
                session + "/share",
                size + 2 * archive,
                files + 4,
                "one expanded release, staged and published ZIP and manifest",
            )
        )
        # BudgetJournal appends many events to one file. Price bytes per event,
        # and one directory, lock, binding, journal and staging inode per ledger.
        journal = release["journal"]

        def ledger(name, destination, events, ledgers):
            size = (events * journal["bytes"] + journal["events"] - 1) // journal["events"]
            return owner(
                name,
                destination,
                size,
                5 * ledgers,
                "append-only debit ledgers including bindings and pending writes",
            )

        # Each resident chunk covers a generator and a scorer, each with a reserve/commit.
        full.append(
            ledger("role_accounts", "run-state/discovery-accounting", 4 * sum(chunks.values()), 1)
        )
        if updates:
            if set(updates) != set(wave["methods"]):
                raise ValueError("update counts required for all retained recipes")
            full.append(
                ledger(
                    "learner_accounts",
                    full[0]["path"],
                    2 * sum(integer(n, 1) for n in updates.values()),
                    len(updates),
                )
            )
            prefix.append(ledger("first_account", prefix[0]["path"], 2 * updates["00"], 1))
        else:
            missing.append("learner_accounting")
    mounts = velocity_mounts(root, full, owner_floors(root, full, floor, registered))
    first = velocity_mounts(root, prefix, owner_floors(root, prefix, floor, registered))
    fits = not any(m["deficit_bytes"] or m["deficit_inodes"] for m in mounts)
    return dict(
        wave=mounts,
        first_ordinary_cut=first,
        chunks=chunks,
        missing=missing,
        fits_with_reserve=False if not fits else None if missing else True,
        basis="Cached image, existing data and serial roles. A full paid paired block bounds one "
        "single-mode eight-case chunk. Size each resident block before rounding. Preserve all "
        "output trees, one pending checkpoint, installation scratch, report staging and journals.",
    )


def launch_preflight(root, policy, info, *, floor):
    prices = info["prices"]
    cycles, fresh = prices["checkpoint_cycles"], prices["fresh_start"]
    result = dict(
        checkpoint_receipt=cycles["receipt"],
        fresh_receipt=fresh["identity"],
        trajectory=None,
        publication_proxy=None,
        checkpoint_space=None,
        registered_mounts=info["capacity"],
        missing={},
    )
    release = None
    try:
        release = release_measurement(root, policy, info)
        result["publication_proxy"] = publication_proxy(release)
        if release is None:
            result["missing"]["publication_proxy"] = "original verified round028 release required"
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        result["missing"]["publication_proxy"] = type(exc).__name__
    bookkeeping = None
    try:
        from . import progressive_velocity_ledger_prices

        accounting = progressive_velocity_ledger_prices.inventory(root, fresh, cycles)
        if accounting:
            result["accounting_receipt"] = accounting["identity"]
            bookkeeping = {
                m: accounting["batches"][str(row["batch"])]["price"]
                for m, row in cycles["methods"].items()
            }
    except (OSError, ValueError, KeyError) as exc:
        result["missing"]["accounting"] = type(exc).__name__
    try:
        if cycles.get("device_charge_us") is None:
            raise ValueError("original successful checkpoint dispatch has no proven wall interval")
        starts, counts = paid_training(root, fresh)
        result["updates_to_4M"] = counts
        trajectory = trajectory_overheads(
            fresh["device"]["methods"], starts, cycles, bookkeeping=bookkeeping
        )
        result["trajectory"] = trajectory
        result["checkpoint_space"] = checkpoint_space(
            root, cycles, trajectory, floor, info["capacity"]["mounts"]
        )
    except (OSError, ValueError, KeyError) as exc:
        result["missing"]["trajectory_or_capacity"] = type(exc).__name__
    if result["trajectory"]:
        try:
            sample = sample_storage(root, prices["supported_wave"])
            result["wave_space"] = wave_space(
                root,
                prices["supported_wave"],
                cycles,
                trajectory,
                sample,
                release,
                floor=floor,
                updates=counts,
                registered=info["capacity"]["mounts"],
            )
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            result["missing"]["wave_space"] = type(exc).__name__
    return result
