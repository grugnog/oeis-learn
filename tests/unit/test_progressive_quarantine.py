import copy
import pytest
from oeis_learn.generalization.progressive.quarantine import Journal, refute

GRAPH = [
    dict(id=i, kind=k, parents=p)
    for i, k, p in [
        ("t", "target", []),
        ("a", "archive", ["t"]),
        ("l", "library", ["a"]),
        ("u", "consumed_update", ["l"]),
        ("c", "checkpoint", ["u"]),
        ("d", "dossier", ["c"]),
        ("clean", "checkpoint", []),
    ]
]


def test_complete_closure_and_append_only_overlay(tmp_path):
    graph = copy.deepcopy(GRAPH)
    j = Journal(tmp_path)
    j.register(graph)
    before = [p.read_bytes() for p in sorted(j.root.glob("event-*.json"))]
    overlay = j.refute(["t"], reason="independent counterexample")
    assert overlay["affected"] == ["a", "c", "d", "l", "t", "u"]
    assert graph == GRAPH
    assert [p.read_bytes() for p in sorted(j.root.glob("event-*.json"))][: len(before)] == before
    for i in ["a", "c", "d", "l", "t", "u"]:
        with pytest.raises(ValueError, match="quarantined"):
            j.require_clean([i])
    j.require_clean(["clean"])
    j.register([dict(id="later", kind="checkpoint", parents=["c"])])
    with pytest.raises(ValueError):
        j.require_clean(["later"])
    assert j.recovery(["c", "clean"], spent={"tokens": 42}) == dict(
        checkpoint="clean", status="CLEAN_ANCESTOR", spent={"tokens": 42}
    )
    assert j.recovery(["c"], spent={"tokens": 42})["status"] == "CONTAMINATED"


@pytest.mark.parametrize(
    "graph",
    [
        GRAPH + [GRAPH[0]],
        [dict(id="a", kind="target", parents=["missing"])],
        [dict(id="a", kind="target", parents=["a"])],
        [dict(id="a", kind="unknown", parents=[])],
    ],
)
def test_malformed_lineage_fails_closed(graph):
    with pytest.raises(ValueError):
        refute(graph, roots=["a"])


def test_real_checkpoint_restore_refuses_refuted_cut_without_refunding(tmp_path):
    from tests.unit.test_progressive_recovery_transactions import setup
    from tests.integration.test_progressive_recovery import states
    from oeis_learn.generalization.progressive.recovery import Recovery
    from oeis_learn.generalization.progressive.contracts import digest

    t, budget, cuts = setup(tmp_path)
    with Recovery(budget, t, digest("learning")) as recovery:
        record = recovery.initialize(states(), cuts)
        recovery.begin(boot="b", monotonic_us=0)
        recovery.reserve(["row"], 3, boot="b", monotonic_us=1)
        before = budget.path.read_bytes()
        Journal(tmp_path).refute([cuts["archive"]], reason="independent mismatch")
        with pytest.raises(ValueError, match="quarantined"):
            recovery.restore()
        assert budget.path.read_bytes() == before
        assert recovery.state["checkpoint"] == record


def test_dossier_overlay_retains_original_claim(tmp_path):
    j = Journal(tmp_path)
    j.register(GRAPH)
    dossier = {"identity": "review", "payload": {"claim": {"statement": "proved"}, "library": None}}
    before = copy.deepcopy(dossier)
    j.dossier(dossier, checkpoints=["c"])
    j.refute(["t"], reason="false training target")
    assert dossier == before
    with pytest.raises(ValueError):
        j.require_clean(["review"])


def test_null_archive_and_library_sentinels_are_not_false_dependencies(tmp_path):
    from tests.unit.test_progressive_recovery_transactions import setup
    from tests.integration.test_progressive_recovery import states
    from oeis_learn.generalization.progressive.recovery import Recovery
    from oeis_learn.generalization.progressive.contracts import digest

    t, budget, cuts = setup(tmp_path)
    cuts = {k: digest(None) for k in cuts}
    with Recovery(budget, t, digest("learning")) as recovery:
        record = recovery.initialize(states(), cuts)
        assert recovery.restore()["ledger"].consumed == 0
        graph, affected, _, _ = Journal(tmp_path).read()
        assert graph == [dict(id=record["identity"], kind="checkpoint", parents=[])]
        assert affected == set()


def test_cumulative_roots_survive_reopen_and_new_descendants(tmp_path):
    journal = Journal(tmp_path)
    journal.register([dict(id=str(i), kind='target', parents=[]) for i in range(257)])
    journal.refute([str(i) for i in range(256)], reason='first batch')
    journal.refute(['256'], reason='second batch')
    reopened = Journal(tmp_path)
    assert len(reopened.read()[1]) == 257
    reopened.register([dict(id='later', kind='checkpoint', parents=['256'])])
    assert 'later' in reopened.read()[1]
