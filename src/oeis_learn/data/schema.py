"""Database schema initialization and connection management for DuckDB / SQLite."""

from __future__ import annotations

import os
import sqlite3
from typing import Any, Union
import duckdb

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sequences (
    oeis_id VARCHAR(10) PRIMARY KEY,
    name TEXT NOT NULL,
    terms_json TEXT NOT NULL,
    term_count INTEGER NOT NULL,
    tags TEXT NOT NULL,
    curriculum_stage INTEGER NOT NULL,
    joeis_class VARCHAR(255),
    generating_formula TEXT,
    lz_complexity DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sequences_stage ON sequences(curriculum_stage);
CREATE INDEX IF NOT EXISTS idx_sequences_tags ON sequences(tags);

CREATE TABLE IF NOT EXISTS synthesis_benchmarks (
    benchmark_id VARCHAR(64) PRIMARY KEY,
    oeis_id VARCHAR(10) NOT NULL,
    curriculum_stage INTEGER NOT NULL,
    wat_code TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    consumed_fuel INTEGER NOT NULL,
    status VARCHAR(32) NOT NULL,
    extrapolation_passed BOOLEAN NOT NULL,
    mdl_ratio DOUBLE PRECISION NOT NULL,
    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_benchmarks_oeis ON synthesis_benchmarks(oeis_id);
CREATE INDEX IF NOT EXISTS idx_benchmarks_stage ON synthesis_benchmarks(curriculum_stage);

CREATE TABLE IF NOT EXISTS discovered_relations (
    relation_id VARCHAR(64) PRIMARY KEY,
    relation_type VARCHAR(32) NOT NULL,
    sequence_ids TEXT NOT NULL,
    vector_distance DOUBLE PRECISION NOT NULL,
    pslq_vector TEXT,
    pslq_confidence DOUBLE PRECISION,
    symbolic_proof TEXT,
    status VARCHAR(32) NOT NULL,
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

SQLITE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sequences (
    oeis_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    terms_json TEXT NOT NULL,
    term_count INTEGER NOT NULL,
    tags TEXT NOT NULL,
    curriculum_stage INTEGER NOT NULL,
    joeis_class TEXT,
    generating_formula TEXT,
    lz_complexity REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sequences_stage ON sequences(curriculum_stage);
CREATE INDEX IF NOT EXISTS idx_sequences_tags ON sequences(tags);

CREATE TABLE IF NOT EXISTS synthesis_benchmarks (
    benchmark_id TEXT PRIMARY KEY,
    oeis_id TEXT NOT NULL,
    curriculum_stage INTEGER NOT NULL,
    wat_code TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    consumed_fuel INTEGER NOT NULL,
    status TEXT NOT NULL,
    extrapolation_passed INTEGER NOT NULL,
    mdl_ratio REAL NOT NULL,
    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_benchmarks_oeis ON synthesis_benchmarks(oeis_id);
CREATE INDEX IF NOT EXISTS idx_benchmarks_stage ON synthesis_benchmarks(curriculum_stage);

CREATE TABLE IF NOT EXISTS discovered_relations (
    relation_id TEXT PRIMARY KEY,
    relation_type TEXT NOT NULL,
    sequence_ids TEXT NOT NULL,
    vector_distance REAL NOT NULL,
    pslq_vector TEXT,
    pslq_confidence REAL,
    symbolic_proof TEXT,
    status TEXT NOT NULL,
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def init_database(db_path: str = "data/oeis_learn.duckdb", use_sqlite: bool = False) -> Any:
    """Initialize the DuckDB or SQLite database with all required schema tables."""
    if db_path != ":memory:":
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

    if use_sqlite or db_path.endswith((".sqlite", ".db")):
        sqlite_conn = sqlite3.connect(db_path)
        sqlite_conn.executescript(SQLITE_SCHEMA_SQL)
        sqlite_conn.commit()
        return sqlite_conn
    else:
        duck_conn = duckdb.connect(db_path)
        # Execute each statement
        for stmt in SCHEMA_SQL.strip().split(";"):
            cleaned = stmt.strip()
            if cleaned:
                duck_conn.execute(cleaned)
        return duck_conn
