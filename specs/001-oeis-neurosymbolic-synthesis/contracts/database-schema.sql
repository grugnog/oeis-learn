-- OEIS Learn Local Storage Schema (DuckDB / SQLite)

CREATE TABLE IF NOT EXISTS sequences (
    oeis_id VARCHAR(10) PRIMARY KEY,
    name TEXT NOT NULL,
    terms_json TEXT NOT NULL,         -- JSON array of initial integers e.g. [0, 1, 1, 2, 3, 5, 8]
    term_count INTEGER NOT NULL,
    tags TEXT NOT NULL,               -- Comma-separated tags e.g. "core,easy,nonn"
    curriculum_stage INTEGER NOT NULL CHECK (curriculum_stage BETWEEN 1 AND 5),
    joeis_class VARCHAR(255),
    generating_formula TEXT,
    lz_complexity DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sequences_stage ON sequences(curriculum_stage);
CREATE INDEX IF NOT EXISTS idx_sequences_tags ON sequences(tags);

CREATE TABLE IF NOT EXISTS synthesis_benchmarks (
    benchmark_id VARCHAR(64) PRIMARY KEY,
    oeis_id VARCHAR(10) NOT NULL REFERENCES sequences(oeis_id),
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
    sequence_ids TEXT NOT NULL,       -- Comma-separated A-numbers e.g. "A000045,A000032,A000213"
    vector_distance DOUBLE PRECISION NOT NULL,
    pslq_vector TEXT,                 -- JSON array of integers e.g. [1, 1, -1]
    pslq_confidence DOUBLE PRECISION,
    symbolic_proof TEXT,
    status VARCHAR(32) NOT NULL,      -- 'CONJECTURED', 'PSLQ_VERIFIED', 'PROVEN', 'REJECTED'
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
