"""Stdlib offline reconciliation; independent arithmetic over public tables only."""

from collections import Counter, defaultdict
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import random
import stat
import zipfile
from .contracts import seal, digest
from .report import MAX_BYTES, MAX_PART, KINDS, validate_manifest, validate_rows, validate_producer


def _json(blob):
    def pairs(items):
        d = {}
        for k, v in items:
            if k in d:
                raise ValueError("duplicate JSON key")
            d[k] = v
        return d

    return json.loads(
        blob,
        object_pairs_hook=pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
    )


def reduce_outcomes(rows, roster, seeds, design, *, costs=None):
    members = {r["task"]: r for r in roster}
    if len(members) != len(roster) or not members or not seeds:
        raise ValueError("unique nonempty frozen roster/seeds required")
    found = {}
    for row in rows:
        key = row["seed"], row["arm"], row["task"]
        if (
            key in found
            or row["seed"] not in seeds
            or row["task"] not in members
            or any(row[k] != members[row["task"]][k] for k in ("group", "stratum"))
        ):
            raise ValueError("outcome rectangle/membership differs")
        found[key] = row["outcome"]
    cells = defaultdict(list)
    for seed in sorted(seeds):
        for arm in ("baseline", "candidate"):
            for task, m in members.items():
                cells[seed, arm, m["stratum"], m["group"]].append(
                    found.get((seed, arm, task), "unresolved")
                )
    primary = tuple(design.get("primary", ("composition", "unseen_structure")))
    if primary not in {("composition", "unseen_structure"), ("oeis",)} or (primary == ("oeis",) and design["claim"] != "capability"):
        raise ValueError("unsupported frozen endpoint")
    vectors = {s: {} for s in primary}
    effects = []
    completion = []
    retention = []
    for seed in sorted(seeds):
        seed_cells = []
        for s in primary:
            gs = sorted({m["group"] for m in roster if m["stratum"] == s})
            diffs = []
            for g in gs:
                a, b = (cells[seed, arm, s, g] for arm in ("baseline", "candidate"))
                n = len(a)
                if n != len(b) or not n:
                    raise ValueError("unpaired group")
                lo = F(b.count("success") - a.count("success") - a.count("unresolved"), n)
                mid = F(b.count("success") - a.count("success"), n)
                hi = F(b.count("success") + b.count("unresolved") - a.count("success"), n)
                vectors[s].setdefault(g, []).append((lo, mid, hi))
                diffs.append(mid)
            if diffs:
                seed_cells.append(sum(diffs) / len(diffs))
            for arm in ("baseline", "candidate"):
                all_values = [
                    v
                    for (sd, ar, st, _), vs in cells.items()
                    if (sd, ar, st) == (seed, arm, s)
                    for v in vs
                ]
                completion.append(
                    1 - F(all_values.count("unresolved"), len(all_values)) if all_values else 0
                )
        effects.append(sum(seed_cells) / len(primary) if len(seed_cells) == len(primary) else None)
        a, b = (
            [
                v
                for (sd, ar, st, _), vs in cells.items()
                if (sd, ar, st) == (seed, arm, "retention")
                for v in vs
            ]
            for arm in ("baseline", "candidate")
        )
        retention.append(
            F(len(a) - a.count("failure"), len(a)) - F(b.count("success"), len(b))
            if a and b
            else None
        )
    nodes = {s + "/" + g for s in primary for g in vectors[s]}
    parent = {n: n for n in nodes}

    def root(n):
        while parent[n] != n:
            n = parent[n]
        return n

    for link in design["links"]:
        if (
            len(link) != 2
            or any(n not in nodes for n in link)
            or link[0].split("/")[0] == link[1].split("/")[0]
        ):
            raise ValueError("invalid frozen bootstrap link")
        a, b = sorted((root(link[0]), root(link[1])))
        parent[b] = a
    units = {n: root(n) for n in nodes}
    support = min((len({units[s + "/" + g] for g in vectors[s]}) for s in primary), default=0)
    metrics = dict(
        mean_gain=float(sum(effects) / len(effects))
        if all(x is not None for x in effects)
        else None,
        groups=support,
        completion=float(min(completion)),
        lower=None,
        upper=None,
    )
    if len(seeds) < 2 or support < 16 or any(x is None for x in retention + effects):
        return metrics, "INCOMPLETE"
    vectors = {
        s: {
            g: tuple(sum(v[j] for v in vs) / len(seeds) for j in range(3))
            for g, vs in table.items()
        }
        for s, table in vectors.items()
    }

    def bounds(selected=None):
        rng = random.Random(design["bootstrap_seed"])
        lower = []
        upper = []
        blocks = [primary] if design["links"] else [(s,) for s in primary]
        for _ in range(10000):
            total = [F(0), F(0)]
            for block in blocks:
                labels = sorted({units[s + "/" + g] for s in block for g in vectors[s]})
                for _retry in range(128):
                    weights = Counter(labels[rng.randrange(len(labels))] for _ in labels)
                    den = {s: sum(weights[units[s + "/" + g]] for g in vectors[s]) for s in block}
                    if all(den.values()):
                        break
                else:
                    raise ValueError("bootstrap budget exhausted")
                for s in block:
                    if selected is not None and s != selected:
                        continue
                    for i, j in enumerate((0, 2)):
                        total[i] += (
                            sum(weights[units[s + "/" + g]] * v[j] for g, v in vectors[s].items())
                            / den[s]
                            / (1 if selected else len(primary))
                        )
            lower.append(total[0])
            upper.append(total[1])
        lower.sort()
        upper.sort()
        d = 20 * design["comparisons"]
        return float(lower[max(1, 10000 // d) - 1]), float(
            upper[min(10000, (10000 * (d - 1) + d - 1) // d) - 1]
        )

    lo, hi = bounds()
    metrics.update(lower=lo, upper=hi)
    if min(completion) < F(95, 100):
        return metrics, "INCOMPLETE"
    if hi < 0:
        return metrics, "REGRESSION"
    retained = max(retention) <= F(2, 100)
    if (
        design["claim"] == "capability"
        and retained
        and metrics["mean_gain"] >= 0.02
        and min(effects) > 0
        and lo > 0
    ):
        return metrics, "CONFIRMED_GAIN"
    if design["claim"] == "efficiency" and costs is None:
        return metrics, "INCOMPLETE"
    if design["claim"] == "efficiency" and retained and costs:
        if 5 * (costs["baseline"] - costs["candidate"]) >= costs["baseline"] and all(
            bounds(s)[0] >= -0.01 for s in primary
        ):
            return metrics, "CONFIRMED_EFFICIENCY"
    return metrics, "NO_MATERIAL_GAIN" if hi < 0.02 else "INCONCLUSIVE"


def verify(path, *, max_bytes=MAX_BYTES):
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_BYTES:
        raise ValueError("verification byte cap")
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > MAX_BYTES:
        raise ValueError("archive byte cap")
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        if (
            len(names) > 4097
            or len(set(names)) != len(names)
            or sum(i.file_size for i in infos) > max_bytes
        ):
            raise ValueError("ZIP inventory/size bound")
        for i in infos:
            if (
                i.filename.startswith("/")
                or "\\" in i.filename
                or any(p in {"", "..", "."} for p in i.filename.split("/"))
                or stat.S_ISLNK(i.external_attr >> 16)
                or i.flag_bits & 1
            ):
                raise ValueError("unsafe ZIP entry")
        if "manifest.json" not in names or z.getinfo("manifest.json").file_size > MAX_PART:
            raise ValueError("bounded manifest required")
        m = validate_manifest(_json(z.read("manifest.json")))
        inventory = {r["path"]: r for r in m["inventory"]}
        if len(inventory) != len(m["inventory"]) or set(names) != {"manifest.json", *inventory}:
            raise ValueError("closed ZIP inventory differs")
        tables = {k: [] for k in KINDS}
        producers = {}
        scope_authority = None
        for name, r in inventory.items():
            if r["bytes"] > MAX_PART or z.getinfo(name).file_size != r["bytes"]:
                raise ValueError("public partition byte cap")
            import re

            match = re.fullmatch(
                r"tables/(roster|outcomes|accounting|ancestry|qualifications)-[0-9]{4}\.json", name
            )
            if name == "authority/scope.json":
                blob = z.read(name)
                if (
                    len(blob) != r["bytes"]
                    or "sha256:" + hashlib.sha256(blob).hexdigest() != r["sha256"]
                ):
                    raise ValueError("frozen scope bytes differ")
                scope_authority = _json(blob)
                continue
            producer_name = re.fullmatch(r"producers/[0-9a-f]{64}\.json", name)
            if (
                (not match and not producer_name)
                or r["bytes"] > MAX_PART
                or z.getinfo(name).file_size != r["bytes"]
            ):
                raise ValueError("table partition bound")
            blob = z.read(name)
            if "sha256:" + hashlib.sha256(blob).hexdigest() != r["sha256"]:
                raise ValueError("public table digest differs")
            if producer_name:
                producers[name] = _json(blob)
            else:
                tables[match[1]].extend(validate_rows(match[1], _json(blob)))
    if (
        m["summary"] != {k: len(v) for k, v in tables.items()}
        or sum(map(len, tables.values())) > 262144
    ):
        raise ValueError("forged table count")
    missing = list(m["missing"])
    if m["campaign_record"] is None or m["scope_record"] is None or scope_authority is None:
        missing.append("frozen_public_scope_authority")
    else:
        from .campaign import validate_campaign

        campaign = validate_campaign(m["campaign_record"])
        if campaign["identity"] != m["campaign"] or m["scope_record"] not in [
            r["artifact"] for r in campaign["payload"]["references"] if r["role"] == "verifier"
        ]:
            raise ValueError("report scope not frozen in campaign")
        if (
            set(scope_authority)
            != {
                "version",
                "identity",
                "source",
                "scope",
                "seeds",
                "design",
                "roster",
                "qualification_producers",
            }
            or seal(scope_authority) != scope_authority
            or scope_authority["version"] != "progressive-public-scope/v1"
        ):
            raise ValueError("closed original public scope required")
        if any(
            scope_authority[k] != m[k] for k in ("source", "scope", "seeds", "design")
        ) or scope_authority["roster"] != digest(tables["roster"]):
            raise ValueError("report roster/design/scope differs from frozen authority")
        if any(
            inventory["authority/scope.json"][k] != m["scope_record"][k]
            for k in ("bytes", "sha256")
        ):
            raise ValueError("original scope reference differs")
        for q in tables["qualifications"]:
            if scope_authority["qualification_producers"].get(q["scope"]) != q["identity"]:
                raise ValueError("qualification producer not prospectively frozen")
    frozen_caps = (
        {digest(t["id"]): t["caps"] for t in m["campaign_record"]["payload"]["tasks"]}
        if m["campaign_record"] is not None
        else {}
    )
    accounting = {}
    measured_costs = {"baseline": 0, "candidate": 0}
    for r in tables["accounting"]:
        if (
            r["task"] in accounting
            or r["tokens"] != r["useful_tokens"] + r["discarded_tokens"]
            or r["tokens"] > r["token_cap"]
            or r["elapsed_us"] > r["elapsed_cap_us"]
        ):
            raise ValueError("accounting cap/decomposition differs")
        if r["task"] not in m["scope"]["tasks"]:
            raise ValueError("accounting task outside frozen scope")
        if (
            r["task"] not in frozen_caps
            or r["token_cap"] != frozen_caps[r["task"]]["tokens"]
            or r["elapsed_cap_us"] != frozen_caps[r["task"]]["elapsed_us"]
        ):
            raise ValueError("accounting ceiling differs from frozen task")
        accounting[r["task"]] = r
        measured_costs[r["arm"]] += r["elapsed_us"]
    missing.extend("accounting:" + t for t in m["scope"]["tasks"] if t not in accounting)
    parents = defaultdict(list)
    for edge in tables["ancestry"]:
        if edge["child"] not in m["scope"]["tasks"] or edge["parent"] not in m["scope"]["tasks"]:
            raise ValueError("ancestry closure missing")
        parents[edge["child"]].append(edge["parent"])
    done = set()

    def visit(n, active):
        if n in active:
            raise ValueError("cyclic report ancestry")
        if n not in done:
            for p in parents[n]:
                visit(p, active | {n})
            done.add(n)

    for t in list(parents):
        visit(t, set())
    required_producers = set()
    for q in tables["qualifications"]:
        name = "producers/" + q["identity"].split(":")[1] + ".json"
        required_producers.add(name)
        if name not in producers:
            raise ValueError("missing qualification producer")
        validate_producer(producers[name], q)
        raw_ref = q["receipt"]
        original_entry = inventory[name]
        if any(raw_ref[k] != original_entry[k] for k in ("bytes", "sha256")):
            raise ValueError("original producer reference differs")
    if set(producers) != required_producers:
        raise ValueError("unclaimed qualification producer")
    for scope in m["scope"]["qualification_scopes"]:
        producers = [
            q
            for q in tables["qualifications"]
            if q["scope"] == scope
            and q["source"] == m["source"]
            and q["device"] == "actual_HIP"
            and q["status"] == "PASS"
            and q["requested"]
            and q["requested"] == q["executed"]
        ]
        if len(producers) != 1:
            missing.append("qualification:" + scope)
    if not tables["roster"] or not m["seeds"]:
        missing.append("frozen_roster_and_seeds")
        metrics, verdict = {}, "INCOMPLETE"
    else:
        metrics, verdict = reduce_outcomes(
            tables["outcomes"],
            tables["roster"],
            m["seeds"],
            m["design"],
            costs=measured_costs if all(measured_costs.values()) else None,
        )
    for decision in m["decisions"]:
        if decision["verdict"] != verdict or (
            decision["action"] == "ADVANCE_REGISTERED"
            and verdict not in {"CONFIRMED_GAIN", "CONFIRMED_EFFICIENCY"}
        ):
            raise ValueError("forged decision/gate")
        if decision["action"] == "ADVANCE_REGISTERED" and not any(
            e["condition"] == verdict and e["action"] == "ADVANCE_REGISTERED"
            for e in m["campaign_record"]["payload"]["graph"]["edges"]
        ):
            raise ValueError("unreserved gate action")
    return seal(
        dict(
            version="progressive-verification/v1",
            report=m["identity"],
            status="INCOMPLETE" if missing else "VERIFIED",
            missing=sorted(set(missing)),
            decision=verdict,
            metrics=metrics,
            producer_error=m["producer_error"],
            collector_errors=m["collector_errors"],
        )
    )
