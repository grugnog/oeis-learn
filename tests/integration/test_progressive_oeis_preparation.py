import pytest

from oeis_learn.generalization.progressive.contracts import atomic_write, record, reference
from oeis_learn.pipeline_validation.identity import file_hash
from oeis_learn.pipeline_validation.progressive_preparation import prepare_oeis_pool


def test_existing_bulk_producer_preserves_offset_and_private_horizon(tmp_path):
    source = tmp_path / "bulk"
    source.mkdir()
    terms = source / "terms.txt"
    terms.write_text("".join(f"{i} {i * i}\n" for i in range(-2, 98)))
    atomic_write(source / "metadata.json", {"offset": -2, "finite": False, "table": False})
    atomic_write(
        source / "manifest.json",
        {
            "version": "indexed-source/v1",
            "release": "pinned",
            "records": [
                {
                    "record_id": "OEIS:A000001",
                    "terms": {"path": "terms.txt", "sha256": file_hash(terms)},
                    "metadata": {
                        "path": "metadata.json",
                        "sha256": file_hash(source / "metadata.json"),
                    },
                }
            ],
        },
    )
    exposure = record(
        "exposure",
        {"previous": None, "detector_limits": ["finite windows only"], "events": []},
        created_at="2026-11-08T00:00:00Z",
        purpose="acceptance",
        track="oeis_derived",
    )
    result = prepare_oeis_pool(
        tmp_path,
        reference(tmp_path, source / "manifest.json"),
        ["OEIS:A000001"],
        exposure=exposure,
        output="prepared",
        count=1,
    )
    assert result["status"] == "COMPLETE"
    from oeis_learn.generalization.progressive.contracts import load

    visible = load(next((tmp_path / "prepared/visible").glob("*.json")))["data"]
    truth = load(next((tmp_path / "prepared/private-admission").glob("*.json")))["data"]
    assert len(visible["values"]) == 20 and visible["indices"][0] == "-2"
    assert len(truth["values"]) == 100 and truth["indices"][-1] == "97"
    assert set(visible) == {"id", "indices", "values"}
    terms.write_text("0 0\n")
    with pytest.raises(ValueError):
        prepare_oeis_pool(
            tmp_path,
            reference(tmp_path, source / "manifest.json"),
            ["OEIS:A000001"],
            exposure=exposure,
            output="other",
            count=1,
        )
