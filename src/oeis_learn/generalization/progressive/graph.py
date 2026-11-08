"""Finite stage selection and leaf contingencies; no arbitrary executable DAG."""

from .contracts import _check, schema
from oeis_learn.generalization.budgets import integer

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
        if node["kind"] == "diagnostic" and not any(
            e["target"] == node["id"] for e in graph["edges"]
        ):
            raise ValueError("untriggered diagnostic task")
        required = DEPENDENCIES[node["stage"]]
        ancestor_stages = set()

        def ancestors(name):
            for dep in nodes[name]["dependencies"]:
                ancestor_stages.add(nodes[dep]["stage"])
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
