"""Unit tests for Elite Demonstration Buffer (EDB) dormancy-weighted replay sampling."""

from __future__ import annotations

import pytest
from oeis_learn.rl.elite_buffer import EliteSeedDemonstrationBuffer


def test_edb_deduplication_and_capacity():
    edb = EliteSeedDemonstrationBuffer(capacity_per_seq=4)

    wat_1 = "(module (func (export \"compute\") (param $n i32) (result i64) (i64.const 1)))"
    wat_2 = "(module (func (export \"compute\") (param $n i32) (result i64) (i64.const 2)))"
    wat_3 = "(module (func (export \"compute\") (param $n i32) (result i64) (i64.const 3)))"
    wat_4 = "(module (func (export \"compute\") (param $n i32) (result i64) (i64.const 4)))"
    wat_5 = "(module (func (export \"compute\") (param $n i32) (result i64) (i64.const 5)))"

    # Ingest 5 solutions for A000012
    edb.add_canonical_entry("A000012", wat_1, terms=[1]*20, fuel=10, step=1)
    edb.add_canonical_entry("A000012", wat_2, terms=[1]*20, fuel=20, step=2)
    edb.add_canonical_entry("A000012", wat_3, terms=[1]*20, fuel=30, step=3)
    edb.add_canonical_entry("A000012", wat_4, terms=[1]*20, fuel=40, step=4)
    edb.add_canonical_entry("A000012", wat_5, terms=[1]*20, fuel=50, step=5)

    entries = edb.get_entries_for_sequence("A000012")
    assert len(entries) <= 4  # Capacity capped at 4


def test_edb_dormancy_sampling():
    edb = EliteSeedDemonstrationBuffer()
    edb.add_canonical_entry("A999001", "(wat 1)", terms=[1]*20, step=10)
    edb.add_canonical_entry("A999002", "(wat 2)", terms=[2]*20, step=500)

    # Current step is 1000 -> A999001 dormancy is 990, A999002 dormancy is 500
    sampled = edb.sample_dormancy_vulnerable_batch(batch_size=2, current_step=1000)
    assert len(sampled) == 2
    sampled_ids = [s[0] for s in sampled]
    assert len(set(sampled_ids)) == 2
