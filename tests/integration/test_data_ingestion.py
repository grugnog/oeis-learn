"""Integration tests for OEIS data ingestion, LZ complexity, and DuckDB storage."""

import os
import tempfile
import pytest
from oeis_learn.data.dataset import OeisSequenceDataset, collate_sequence_batch
from oeis_learn.data.ingest import OeisIngestionPipeline, infer_curriculum_stage
from oeis_learn.data.lz_complexity import lempel_ziv_complexity, normalized_lz_complexity
from oeis_learn.data.models import SequenceRecord, parse_names_line, parse_stripped_line


def test_lz_complexity():
    # Constant sequence has low complexity
    const_seq = [1] * 50
    # Alternating/pseudo-random has higher complexity
    alt_seq = [n % 5 for n in range(50)]

    c_const = lempel_ziv_complexity(const_seq)
    c_alt = lempel_ziv_complexity(alt_seq)

    assert c_const > 0
    assert c_alt > c_const
    assert normalized_lz_complexity(const_seq) > 0.0


def test_parsing_lines():
    stripped_line = "A000045 ,0,1,1,2,3,5,8,13,21,34,55,89"
    parsed = parse_stripped_line(stripped_line)
    assert parsed is not None
    oeis_id, terms = parsed
    assert oeis_id == "A000045"
    assert terms == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

    names_line = "A000045 Fibonacci numbers: F(n) = F(n-1) + F(n-2) with F(0) = 0 and F(1) = 1."
    parsed_name = parse_names_line(names_line)
    assert parsed_name is not None
    assert parsed_name[0] == "A000045"
    assert "Fibonacci numbers" in parsed_name[1]


def test_infer_curriculum_stage():
    assert infer_curriculum_stage(["core", "easy"], "Polynomial", [1, 4, 9, 16]) == 1
    assert infer_curriculum_stage(["frac"], "Fibonacci numbers", [0, 1, 1, 2, 3, 5]) == 2
    assert infer_curriculum_stage(["nice", "tabl"], "Catalan numbers", [1, 1, 2, 5, 14]) == 3
    assert infer_curriculum_stage(["hard"], "Number of partitions of n", [1, 1, 2, 3, 5, 7]) == 4
    assert infer_curriculum_stage(["bref"], "Graph chromatic invariant", [1, 2, 3]) == 5


def test_ingestion_and_dataset_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_oeis.duckdb")
        pipeline = OeisIngestionPipeline(db_path=db_path)

        inserted = pipeline.generate_synthetic_curriculum_dataset(num_per_stage=10)
        assert inserted == 50
        pipeline.close()

        # Query all records via Dataset
        dataset = OeisSequenceDataset(db_path=db_path)
        assert len(dataset) == 50

        # Query Stage 1 only
        stage1_ds = OeisSequenceDataset(db_path=db_path, stage_subset=[1])
        assert len(stage1_ds) == 10

        item = stage1_ds[0]
        assert item["stage"] == 1
        assert len(item["terms"]) > 0

        # Test batch collation
        batch = [stage1_ds[0], stage1_ds[1]]
        collated = collate_sequence_batch(batch)
        assert len(collated["oeis_ids"]) == 2
        assert collated["stages"].tolist() == [1, 1]
