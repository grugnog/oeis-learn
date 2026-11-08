import pytest

from oeis_learn.generalization.progressive.archive import Archive
from oeis_learn.generalization.progressive.contracts import digest, record


def admitted(task, structure, tokens=3):
    return record(
        "admission",
        {
            "member": digest(task),
            "task": task,
            "group": task,
            "source_snapshot": digest("snapshot"),
            "source": "(program (return $index))",
            "source_hash": digest("source"),
            "indices": list(map(str, range(100))),
            "values_hash": digest("truth"),
            "structure": digest(structure),
            "tokens": list(range(tokens)),
            "target_source": "literal",
            "target_disposition": "literal",
            "native_verified": True,
            "native": digest("native"),
            "visible_values": list(map(str, range(20))),
            "reference": digest("reference"),
            "limits": digest("limits"),
            "validity": "finite_100_terms_only",
            "membership": "training",
            "status": "COMMITTED",
        },
        created_at="2026-11-08T00:00:00Z",
        purpose="acceptance",
        track="oeis_derived",
    )


def test_task_balance_duplicates_structural_alternatives_and_resume(tmp_path):
    a = Archive(tmp_path)
    rows = [
        admitted("a", "first", 2),
        admitted("a", "first", 4),
        admitted("a", "second", 3),
        admitted("a", "third", 8),
        admitted("b", "only", 2),
    ]
    cut = a.publish(rows + rows)
    assert len(a.entries(cut)) == 3
    chosen = a.sample(cut, ordinal=0, count=64)
    assert {r["task"] for r in chosen} == {"a", "b"}
    assert chosen == Archive(tmp_path).sample(cut, ordinal=0, count=64)
    assert all(r["charged_tokens"] == len(r["tokens"]) for r in chosen)
    with pytest.raises(ValueError):
        Archive(tmp_path, capacity=2).entries(cut)


def test_uncommitted_or_exposed_target_cannot_enter_archive(tmp_path):
    from oeis_learn.pipeline_validation.identity import seal

    r = admitted("a", "a")
    r["payload"]["status"] = "RESERVED"
    with pytest.raises(ValueError):
        Archive(tmp_path).publish([seal(r)])
    r = admitted("a", "a")
    r["payload"]["membership"] = "assessment"
    with pytest.raises(ValueError):
        Archive(tmp_path).publish([seal(r)])
