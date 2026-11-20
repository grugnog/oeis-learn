"""Independent three-structure component support and durable publication cases."""

import pytest

from oeis_learn.generalization.progressive.contracts import atomic_write, digest, reference, seal

SOURCES = [
    "(program (return (add (mul $index (int 2)) (int 1))))",
    "(program (return (sub (add (mul $index (int 3)) (int 4)) (int 5))))",
    "(program (return (mul (add (mul $index (int 4)) (int 5)) $index)))",
]


def sources(root, client=None, programs=SOURCES):
    from oeis_learn.pipeline_validation.progressive_preparation import verify_draw
    from oeis_learn.generalization.progressive.generator import structure_id

    refs = []
    for i, source in enumerate(programs):
        row, reason = verify_draw(
            dict(
                source=source,
                ordinal=i,
                draw_identity=digest(["fixture", i, source]),
                source_policy=digest("component-test"),
                structure=structure_id(source),
                detector_limit="DIAGNOSTIC_ONLY hand-checked sources",
            ),
            client=client,
        )
        assert row is not None, reason
        atomic_write(root / f"source-{i}.json", row)
        refs.append(reference(root, f"source-{i}.json"))
    return refs


def test_mines_alpha_normalized_connected_subtree_and_expands_exactly(tmp_path):
    from oeis_learn.generalization.progressive.component_mining import LibraryStore
    from oeis_learn.generalization.progressive.components import expand
    from oeis_learn.synthesis.reference import evaluate

    refs = sources(tmp_path)
    store = LibraryStore(tmp_path, "library", purpose="acceptance", track="synthetic_strict")
    library = store.publish(refs, client=None)
    motif = ["add", ["mul", ["arg", "h0"], ["arg", "h1"]], ["arg", "h2"]]
    component = next(c for c in library["payload"]["components"] if c["body"] == motif)
    assert len({o["structure"] for o in component["origins"]}) == 3
    assert len({o["group"] for o in component["origins"]}) >= 2
    result = expand(library, ["call", component["id"], ["int", "2"], ["int", "7"], ["int", "3"]])
    assert result["source"] == "(program (return (add (mul (int 2) (int 7)) (int 3))))"
    assert evaluate(result["source"], ["0", "1", "2"]).values == ("17", "17", "17")
    assert library["payload"]["verification"]["status"] == "DIAGNOSTIC_ONLY"
    before = store.work.read_bytes()
    assert store.publish(list(reversed(refs)), client=None) == library
    assert store.work.read_bytes() == before


def test_numeric_variants_do_not_invent_structural_support(tmp_path):
    from oeis_learn.generalization.progressive.component_mining import LibraryStore

    refs = sources(
        tmp_path, programs=[SOURCES[0].replace("(int 2)", f"(int {n})") for n in (2, 3, 4)]
    )
    library = LibraryStore(
        tmp_path, "library", purpose="acceptance", track="synthetic_strict"
    ).publish(refs)
    assert library["payload"]["components"] == []


def test_failed_or_exposed_origins_cannot_publish_and_work_stays_spent(tmp_path):
    from oeis_learn.generalization.progressive.component_mining import LibraryStore
    from oeis_learn.generalization.progressive.contracts import load

    refs = sources(tmp_path)
    row = load(tmp_path / refs[0]["path"])
    row["membership"] = "confirmation"
    atomic_write(tmp_path / "bad.json", seal(row))
    refs[0] = reference(tmp_path, "bad.json")
    store = LibraryStore(tmp_path, "library", purpose="acceptance", track="synthetic_strict")
    with pytest.raises(ValueError, match="training"):
        store.publish(refs)
    assert not store.committed()
    assert store.spent() > 0


def test_interrupted_verify_is_not_a_cut_and_resume_is_deterministic(tmp_path):
    from oeis_learn.generalization.progressive.component_mining import LibraryStore

    refs = sources(tmp_path)
    store = LibraryStore(tmp_path, "library", purpose="acceptance", track="synthetic_strict")
    with pytest.raises(InterruptedError):
        store.publish(refs, fault="after-verify")
    spent = store.spent()
    assert not store.committed()
    recovered = LibraryStore(tmp_path, "library", purpose="acceptance", track="synthetic_strict")
    library = recovered.publish(refs)
    assert recovered.spent() >= spent
    assert recovered.committed() == [library]


def test_reservation_exhaustion_and_native_disagreement_do_not_publish(tmp_path):
    from oeis_learn.generalization.progressive.component_mining import LibraryStore, policy

    refs = sources(tmp_path)
    store = LibraryStore(
        tmp_path,
        "small",
        purpose="acceptance",
        track="synthetic_strict",
        selected=policy(maximum_work=1),
    )
    with pytest.raises(TimeoutError):
        store.publish(refs)
    assert not store.committed()

    class Disagreement:
        def execute(self, source, indices, **kwargs):
            return dict(outcome="ok", values=["0"] * len(indices))

        def runtime_identity(self, *args):
            return digest("not-a-production-receipt")

    store = LibraryStore(tmp_path, "disagreement", purpose="acceptance", track="synthetic_strict")
    with pytest.raises(ValueError, match="native/reference"):
        store.publish(refs, client=Disagreement())
    assert not store.committed()


def test_full_expansions_use_actual_native_and_independent_reference(tmp_path, fitting_client):
    from oeis_learn.generalization.progressive.component_mining import LibraryStore

    refs = sources(tmp_path, client=fitting_client)
    store = LibraryStore(tmp_path, "library", purpose="acceptance", track="synthetic_strict")
    library = store.publish(refs, client=fitting_client)
    assert library["payload"]["verification"]["status"] == "NATIVE_VERIFIED"
    assert library["payload"]["verification"]["native"] == fitting_client.runtime_identity(
        "bigint-body/v1"
    )
    assert library["payload"]["verification"]["proofs"]


def test_library_versions_keep_prior_definitions_and_cuts(tmp_path):
    from oeis_learn.generalization.progressive.component_mining import LibraryStore
    from oeis_learn.generalization.progressive.components import validate_library

    refs = sources(tmp_path)
    store = LibraryStore(tmp_path, "library", purpose="acceptance", track="synthetic_strict")
    first = store.publish(refs)
    second = store.publish(refs, previous=first)
    assert second["payload"]["previous"] == first["identity"]
    assert second["payload"]["components"] == first["payload"]["components"]
    assert {v["identity"] for v in store.committed()} == {first["identity"], second["identity"]}
    assert validate_library(first) == first


def test_installed_component_roles_use_the_prepaid_native_path(tmp_path, fitting_client):
    import time
    from tests.unit.test_progressive_freeze import campaign
    from oeis_learn.generalization.progressive.budgets import BudgetJournal, role_work_requirements
    from oeis_learn.generalization.progressive.component_mining import policy
    from oeis_learn.generalization.progressive.contracts import record, load
    from oeis_learn.generalization.progressive.projections import describe, project
    from oeis_learn.pipeline_validation.progressive_roles import REGISTRY
    from oeis_learn.pipeline_validation.progressive_role import execute
    from pathlib import Path

    refs = sources(tmp_path, client=fitting_client)
    paths = []
    for i, ref in enumerate(refs):
        name = f"channel-{i}.json"
        atomic_write(
            tmp_path / name,
            seal(
                dict(
                    version="discovery-channel/v1",
                    kind="typed_training_source",
                    role="component",
                    data=dict(
                        membership="training",
                        artifact=ref,
                        profile="bigint-body/v1",
                        compatibility=digest("component-test"),
                    ),
                )
            ),
        )
        paths.append(name)
    atomic_write(
        tmp_path / "policy.json",
        seal(
            dict(
                version="discovery-channel/v1",
                kind="component_policy",
                role="component",
                data=dict(policy=policy(maximum_work=8192), previous=None),
            )
        ),
    )
    paths.append("policy.json")
    artifacts = {"archive": [describe(tmp_path, p) for p in paths]}
    project(
        tmp_path,
        "inputs",
        "extract-components",
        artifacts,
        registry=REGISTRY,
        authorized=[a.ref for a in artifacts["archive"]],
    )
    values = [load(tmp_path / p) for p in paths]
    c = campaign()
    p = c["payload"]["tasks"][0]
    p["startup_us"] = p["finalization_us"] = 1000
    for key in ("attempts", "rejections", "native_work"):
        p["caps"][key] = 32768
        for node in c["payload"]["graph"]["nodes"]:
            node["caps"][key] = 32768
        c["payload"]["caps"][key] = 65536
    c["payload"]["storage"]["required_inodes"] = 16384
    c["payload"]["storage"]["available_inodes"] = 32768
    c = seal(c)
    t = record(
        "task",
        dict(p, campaign=c["identity"]),
        created_at=c["created_at"],
        purpose=c["purpose"],
        track=c["track"],
    )
    journal = BudgetJournal(tmp_path / "accounting", c["identity"], c["payload"]["caps"])
    journal.task_remaining(t)
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    results = []
    for i, operation in enumerate(("extract-components", "verify-components")):
        cost = role_work_requirements(operation, values)
        cost["elapsed_us"] = 60_000_000
        reserved = journal.reserve(
            p["id"], operation, i, cost, boot=boot, monotonic_us=time.monotonic_ns() // 1000
        )
        event = journal._read()[0][-1]
        authority = seal(
            dict(
                version="progressive-role-budget/v1",
                campaign=c,
                task=t,
                operation=operation,
                inputs=digest(sorted(values, key=lambda v: v["identity"])),
                reservation=event,
            )
        )
        authority_path = tmp_path / f"authority-{i}.json"
        atomic_write(authority_path, authority)
        result = execute(
            operation,
            inputs=tmp_path / "inputs",
            output=tmp_path / f"out-{i}",
            authority=authority_path,
            component_state=tmp_path / "component-state",
        )
        journal.commit(
            reserved["key"],
            boot=boot,
            monotonic_us=time.monotonic_ns() // 1000,
            termination_proven=True,
        )
        assert result["mechanical_status"] == "COMPLETE"
        assert result["metrics"]["components"] > 0
        results.append(result["metrics"]["library"])
    assert results[0] == results[1]


def test_research_rejects_unbound_mechanical_provenance_before_native_execution(tmp_path):
    from oeis_learn.generalization.progressive.component_mining import LibraryStore
    from oeis_learn.generalization.progressive.contracts import load

    refs = sources(tmp_path)
    promoted = []
    for i, ref in enumerate(refs):
        value = load(tmp_path / ref["path"])
        value["native_verified"] = True
        atomic_write(tmp_path / f"unproven-{i}.json", seal(value))
        promoted.append(reference(tmp_path, f"unproven-{i}.json"))

    class MustNotExecute:
        def runtime_identity(self, *args):
            return digest("not-a-native-receipt")

        def execute(self, *args, **kwargs):
            raise AssertionError("unbound source reached execution")

    store = LibraryStore(tmp_path, "unbound", purpose="research", track="synthetic_strict")
    with pytest.raises(ValueError, match="mechanical"):
        store.publish(promoted, client=MustNotExecute())
    assert not store.committed()


def test_mechanical_generation_law_replays_draw_identity(tmp_path):
    from oeis_learn.generalization.progressive.component_mining import LibraryStore
    from oeis_learn.generalization.progressive.generator import default_policy, draw
    from oeis_learn.pipeline_validation.progressive_preparation import verify_draw
    from oeis_learn.generalization.progressive.contracts import load

    law = default_policy(seed=13)
    refs = []
    for ordinal in range(32):
        row, _ = verify_draw(draw(law, ordinal))
        if row is not None:
            atomic_write(tmp_path / f"draw-{ordinal}.json", row)
            refs.append(reference(tmp_path, f"draw-{ordinal}.json"))
        if len(refs) == 3:
            break
    store = LibraryStore(
        tmp_path,
        "mechanical",
        purpose="acceptance",
        track="synthetic_strict",
        generation_policies={digest(law): law},
    )
    store.publish(refs)
    changed = load(tmp_path / refs[0]["path"])
    changed["draw_identity"] = digest("relabelled after outcomes")
    atomic_write(tmp_path / "changed-draw.json", seal(changed))
    refs[0] = reference(tmp_path, "changed-draw.json")
    other = LibraryStore(
        tmp_path,
        "altered",
        purpose="acceptance",
        track="synthetic_strict",
        generation_policies={digest(law): law},
    )
    with pytest.raises(ValueError, match="mechanical"):
        other.publish(refs)
