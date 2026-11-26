"""Finite stage selection and leaf contingencies; no arbitrary executable DAG."""

from oeis_learn.generalization.budgets import integer

from .contracts import _check, schema

STAGES = ("capability", "diversity", "tuning", "scaling", "oeis", "components", "source_pilot")
DEPENDENCIES = {
    "capability": set(),
    "diversity": {"capability"},
    "tuning": {"diversity"},
    "scaling": {"tuning"},
    "oeis": {"scaling"},
    "components": {"scaling"},
    "source_pilot": {"capability"},
}


def validate_graph(graph, *, caps):
    _check(schema("common")["$defs"]["graph"], graph, schema("common"))
    selected = set(graph["selected_stages"])
    if any(not DEPENDENCIES[s] <= selected for s in selected):
        raise ValueError("missing selected-stage dependency")
    nodes = {n["id"]: n for n in graph["nodes"]}
    if len(nodes) != len(graph["nodes"]):
        raise ValueError("duplicate task slot")
    if {n["stage"] for n in nodes.values()} != selected:
        raise ValueError("selected stage slot coverage")
    done = set()
    active = set()

    def visit(name):
        if name in active:
            raise ValueError("cyclic stage DAG")
        if name in done:
            return
        if name not in nodes:
            raise ValueError("unresolved dependency")
        active.add(name)
        for dependency in nodes[name]["dependencies"]:
            visit(dependency)
        active.remove(name)
        done.add(name)

    for name in nodes:
        visit(name)
    for edge in graph["edges"]:
        if edge["source"] not in nodes or edge["target"] not in nodes:
            raise ValueError("unknown edge or hidden assessment")
        if edge["source"] not in nodes[edge["target"]]["dependencies"]:
            raise ValueError("edge lacks declared dependency")
        if nodes[edge["source"]]["kind"] == "diagnostic":
            raise ValueError("recursive diagnostic forbidden")
        if edge["action"] == "ADVANCE_REGISTERED" and edge["condition"] not in {
            "COMPLETE",
            "CONFIRMED_GAIN",
            "CONFIRMED_EFFICIENCY",
        }:
            raise ValueError("failed claim cannot authorize confirmed expansion")
        if (
            nodes[edge["target"]]["stage"] in {"scaling", "oeis"}
            and edge["action"] == "ADVANCE_REGISTERED"
            and edge["condition"] != "CONFIRMED_GAIN"
        ):
            raise ValueError("confirmed scaling requires executable gain")
        if edge["action"] == "EXPLORE_RESERVED" and nodes[edge["target"]]["kind"] != "diagnostic":
            raise ValueError("diagnostic reservation required")
        if nodes[edge["target"]]["kind"] == "diagnostic" and edge["action"] != "EXPLORE_RESERVED":
            raise ValueError("diagnostic cannot inherit a passed gate")
    for node in nodes.values():
        if (
            node["stage"] in {"scaling", "oeis"}
            and node["kind"] == "routine"
            and not any(
                e["target"] == node["id"]
                and e["condition"] == "CONFIRMED_GAIN"
                and e["action"] == "ADVANCE_REGISTERED"
                for e in graph["edges"]
            )
        ):
            raise ValueError("confirmed scaling requires a frozen executable gain edge")
        if node["kind"] == "diagnostic" and not any(
            e["target"] == node["id"] for e in graph["edges"]
        ):
            raise ValueError("untriggered diagnostic task")
        required = DEPENDENCIES[node["stage"]]
        ancestor_stages = set()

        def ancestors(name):
            for dep in nodes[name]["dependencies"]:
                ancestor_stages.add(nodes[dep]["stage"])  # noqa: B023 -- invoked before the next loop iteration
                ancestors(dep)

        ancestors(node["id"])
        if not required <= ancestor_stages:
            raise ValueError("stage bypass")
    for key, cap in caps.items():
        integer(cap)
        if sum(n["caps"][key] for n in nodes.values()) > cap:
            raise ValueError("child reservations exceed campaign " + key)
    diagnostics = [n for n in nodes.values() if n["kind"] == "diagnostic"]
    if len(diagnostics) > 4 or any(
        n["caps"]["elapsed_us"] > 12 * 3600 * 1_000_000 for n in diagnostics
    ):
        raise ValueError("diagnostic task cap")
    for key, cap in graph["diagnostic_pool"].items():
        if sum(n["caps"][key] for n in diagnostics) > cap:
            raise ValueError("diagnostic pool exceeded")
    if graph["diagnostic_pool"]["elapsed_us"] > 24 * 3600 * 1_000_000:
        raise ValueError("diagnostic pool maximum")
    scaling = [n for n in nodes.values() if n["stage"] == "scaling" and n["kind"] == "routine"]
    if len(scaling) > 8 or sum(n["caps"]["elapsed_us"] for n in scaling) > 168 * 3600 * 1_000_000:
        raise ValueError("routine scaling reserve")
    return graph


def generator_look(panels, *, registration, pair, arm, target, split):
    """Compile the single registered search request from an already frozen panel roster."""
    from .contracts import digest, seal
    from .cohorts import STRATA
    from .training_rows import GENERATOR_LOOKS, pair_seeds
    seeds = pair_seeds(registration, pair)
    if (arm not in {"ordinary", "first_divergence", "incumbent"} or type(target) is not int
            or target not in GENERATOR_LOOKS or split not in {"development", "replication"}
            or ((arm == "incumbent" or split == "replication") and target != GENERATOR_LOOKS[-1])):
        raise ValueError("unregistered generator arm, split or look")
    names = {"development", "development_retention", "replication", "replication_retention"}
    if set(panels) != names:
        raise ValueError("development and common replication reservations with separate retention required")
    owners = set()
    group_owners = {}
    for name, members in panels.items():
        strata = {"retention"} if name.endswith("_retention") else set(STRATA)
        if (not isinstance(members, list) or {r["stratum"] for r in members} != strata
                or any(set(r) != {"task", "stratum", "group"} or not all(isinstance(v, str) and v for v in r.values())
                       for r in members)):
            raise ValueError("complete and outcome-blind reserved panel membership required")
        if not name.endswith("_retention"):
            for stratum in strata:
                selected = [r for r in members if r["stratum"] == stratum]
                if len(selected) != 32:
                    raise ValueError("target panel requires 32 registered cases in every development/replication stratum")
                if stratum in ("composition", "unseen_structure") and len({r["group"] for r in selected}) < 16:
                    raise ValueError("primary family supply deficit")
                if stratum != "parameter" and any(group_owners.setdefault(r["group"], name) != name for r in selected):
                    raise ValueError("linked evaluation family crosses panel roles")
        keys = {r["task"] for r in members}
        if len(keys) != len(members) or owners & keys:
            raise ValueError("case reused across frozen panel roles")
        owners.update(keys)
    full = target == GENERATOR_LOOKS[-1]
    if full:
        members = [*panels[split], *panels[split + "_retention"]]
    else:
        members = []
        families = set()
        for row in panels[split]:
            key = row["stratum"], row["group"]
            if key not in families:
                members.append(row)
                families.add(key)
    return seal(dict(version="round028-generator-look/v1", registration=registration, pair=pair,
                     seed=seeds["initialization"], arm=arm, target=target, split=split, panels=digest(panels),
                     members=members, attempts=8 if full else 1, portfolio_us=60_000_000, fit_us=4_000_000,
                     max_tokens=1024, profile="bigint-20-100/v1", spending=full))
