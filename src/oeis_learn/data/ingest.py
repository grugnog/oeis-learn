"""OEIS Data Ingestion & Indexing Pipeline into DuckDB / SQLite."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple
from oeis_learn.data.lz_complexity import normalized_lz_complexity
from oeis_learn.data.models import SequenceRecord, parse_names_line, parse_stripped_line
from oeis_learn.data.preprocessing import (
    analyze_log_linearity,
    check_finite_difference_polynomial_degree,
)
from oeis_learn.data.schema import init_database


def infer_curriculum_stage(tags: Sequence[str], name: str, terms: Sequence[int], joeis_class: Optional[str] = None) -> int:
    """Classify an OEIS sequence into one of 5 taxonomic curriculum stages.

    Uses metadata keywords, Lempel-Ziv complexity, finite difference polynomial checks,
    and log-linear growth analysis (from IntSeqBERT).
    """
    tag_set = {t.lower() for t in tags}
    name_lower = name.lower()

    # Stage 5: Search, Graphs, Invariants, Bref
    if "bref" in tag_set or "graph" in name_lower or "game" in name_lower or "search" in name_lower:
        return 5

    # Stage 4: Combinatorics & Number Theory (Hard, primes, partition, base)
    if "hard" in tag_set or "base" in tag_set or "eigen" in tag_set or "prime" in name_lower or "partition" in name_lower:
        return 4

    # Stage 3: Holonomic & D-Finite Sequences (Hypergeometric, factorial, catalan, cofr, tabl)
    if "cofr" in tag_set or "tabl" in tag_set or "tabf" in tag_set or "factorial" in name_lower or "catalan" in name_lower or "hypergeometric" in name_lower:
        return 3

    # Stage 2: Linear Recurrences & Rational GFs (Fibonacci, Pell, Lucas, frac, cons, mult)
    if "frac" in tag_set or "cons" in tag_set or "mult" in tag_set or "fibonacci" in name_lower or "lucas" in name_lower or "recurrence" in name_lower:
        return 2

    # If no explicit tags, perform arithmetic analysis on terms
    if len(terms) >= 6:
        # Check if polynomial (constant d-th difference)
        poly_degree = check_finite_difference_polynomial_degree(terms, max_degree=3)
        if poly_degree is not None:
            return 1

        # Check if exponential/linear recurrence growth (log-linear)
        is_log_lin, _ = analyze_log_linearity(terms)
        if is_log_lin:
            return 2

    # Stage 1: Primitives & Polynomials (Default / core, easy, nonn)
    return 1


class OeisIngestionPipeline:
    """Pipeline for ingesting raw OEIS files or mock sequence generators into DuckDB."""

    def __init__(self, db_path: str = "data/oeis_learn.duckdb", use_sqlite: bool = False):
        self.db_path = db_path
        self.use_sqlite = use_sqlite
        self.conn = init_database(db_path, use_sqlite=use_sqlite)

    def close(self) -> None:
        """Closes the underlying database connection."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def insert_records(self, records: Sequence[SequenceRecord]) -> int:
        """Batch insert sequence records into the database."""
        if not records:
            return 0

        rows = []
        for r in records:
            lz = r.lz_complexity if r.lz_complexity > 0 else normalized_lz_complexity(r.terms)
            rows.append((
                r.oeis_id,
                r.name,
                r.terms_json,
                r.term_count,
                r.tags_str,
                r.curriculum_stage,
                r.joeis_class,
                r.generating_formula,
                lz,
            ))

        if self.use_sqlite or self.db_path.endswith((".sqlite", ".db")):
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO sequences 
                (oeis_id, name, terms_json, term_count, tags, curriculum_stage, joeis_class, generating_formula, lz_complexity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self.conn.commit()
        else:
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO sequences 
                (oeis_id, name, terms_json, term_count, tags, curriculum_stage, joeis_class, generating_formula, lz_complexity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def ingest_from_files(
        self,
        stripped_path: Optional[str] = None,
        names_path: Optional[str] = None,
        stage_filter: Optional[int] = None,
        max_records: Optional[int] = None,
    ) -> int:
        """Ingest stripped and names files from disk."""
        names_dict: Dict[str, str] = {}
        if names_path and os.path.exists(names_path):
            with open(names_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parsed = parse_names_line(line)
                    if parsed:
                        names_dict[parsed[0]] = parsed[1]

        records: List[SequenceRecord] = []
        count = 0

        if stripped_path and os.path.exists(stripped_path):
            with open(stripped_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parsed_stripped = parse_stripped_line(line)
                    if not parsed_stripped:
                        continue
                    oeis_id, terms = parsed_stripped
                    if len(terms) < 5:
                        continue
                    name = names_dict.get(oeis_id, f"Sequence {oeis_id}")
                    stage = infer_curriculum_stage([], name, terms)
                    if stage_filter is not None and stage != stage_filter:
                        continue

                    rec = SequenceRecord(
                        oeis_id=oeis_id,
                        name=name,
                        terms=terms,
                        tags=["nonn", "core"] if stage == 1 else [],
                        curriculum_stage=stage,
                        lz_complexity=normalized_lz_complexity(terms),
                    )
                    records.append(rec)
                    if len(records) >= 1000:
                        self.insert_records(records)
                        count += len(records)
                        records.clear()
                    if max_records and count >= max_records:
                        break

        if records:
            self.insert_records(records)
            count += len(records)

        return count

    def generate_synthetic_curriculum_dataset(
        self,
        num_per_stage: int = 100,
        elite_buffer: Optional[Any] = None,
    ) -> int:
        """Generates representative synthetic OEIS benchmark sequences across all 5 stages for testing."""
        from oeis_learn.data.models import EliteReplayBufferEntry

        records: List[SequenceRecord] = []

        # Stage 1: Polynomials & Linear progressions
        for i in range(num_per_stage):
            oeis_id = f"A{100000 + i:06d}"
            # n^2 + c or a*n + b
            a, b, c = (i % 5) + 1, (i % 7), (i % 3)
            terms = [a * n * n + b * n + c for n in range(50)]
            wat_code = f"""(module
  (func (export "compute") (param $n i32) (result i64)
    (local $n64 i64)
    (local.set $n64 (i64.extend_i32_u (local.get $n)))
    (i64.add
      (i64.mul (i64.const {a}) (i64.mul (local.get $n64) (local.get $n64)))
      (i64.add (i64.mul (i64.const {b}) (local.get $n64)) (i64.const {c}))
    )
  )
)"""
            records.append(
                SequenceRecord(
                    oeis_id=oeis_id,
                    name=f"Polynomial sequence {a}n^2 + {b}n + {c}",
                    terms=terms,
                    tags=["core", "easy", "nonn"],
                    curriculum_stage=1,
                    generating_formula=f"a(n) = {a}*n^2 + {b}*n + {c}",
                    lz_complexity=normalized_lz_complexity(terms),
                )
            )
            if elite_buffer is not None:
                elite_buffer.add_entry(
                    EliteReplayBufferEntry(
                        oeis_id=oeis_id,
                        terms=terms,
                        wat_code=wat_code,
                        byte_size=len(wat_code.encode("utf-8")),
                        extrapolation_passed=True,
                        mdl_ratio=0.85,
                        source="SYNTHETIC_CURRICULUM",
                    )
                )

        # Stage 2: Linear Recurrences (Fibonacci-like: F(n) = a*F(n-1) + b*F(n-2))
        for i in range(num_per_stage):
            oeis_id = f"A{200000 + i:06d}"
            c1, c2 = (i % 3) + 1, (i % 2) + 1
            terms = [0, 1]
            for n in range(2, 50):
                terms.append(c1 * terms[-1] + c2 * terms[-2])
            wat_code = f"""(module
  (func (export "compute") (param $n i32) (result i64)
    (local $a i64)
    (local $b i64)
    (local $temp i64)
    (local $i i32)
    (local.set $a (i64.const 0))
    (local.set $b (i64.const 1))
    (local.set $i (i32.const 0))
    (block $exit
      (loop $loop
        (br_if $exit (i32.ge_s (local.get $i) (local.get $n)))
        (local.set $temp (i64.add (i64.mul (local.get $b) (i64.const {c1})) (i64.mul (local.get $a) (i64.const {c2}))))
        (local.set $a (local.get $b))
        (local.set $b (local.get $temp))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )
    (local.get $a)
  )
)"""
            records.append(
                SequenceRecord(
                    oeis_id=oeis_id,
                    name=f"Linear recurrence a(n) = {c1}*a(n-1) + {c2}*a(n-2)",
                    terms=terms,
                    tags=["core", "frac", "cons"],
                    curriculum_stage=2,
                    generating_formula=f"a(n) = {c1}*a(n-1) + {c2}*a(n-2)",
                    lz_complexity=normalized_lz_complexity(terms),
                )
            )
            if elite_buffer is not None:
                elite_buffer.add_entry(
                    EliteReplayBufferEntry(
                        oeis_id=oeis_id,
                        terms=terms,
                        wat_code=wat_code,
                        byte_size=len(wat_code.encode("utf-8")),
                        extrapolation_passed=True,
                        mdl_ratio=0.90,
                        source="SYNTHETIC_CURRICULUM",
                    )
                )

        # Stage 3: Holonomic (Factorials, Catalan, (n+1)*a(n-1))
        for i in range(num_per_stage):
            oeis_id = f"A{300000 + i:06d}"
            terms = [1]
            for n in range(1, 40):
                terms.append(terms[-1] * (n + (i % 3)))
            records.append(
                SequenceRecord(
                    oeis_id=oeis_id,
                    name=f"Holonomic sequence a(n) = (n+{i%3})*a(n-1)",
                    terms=terms,
                    tags=["nice", "cofr", "tabl"],
                    curriculum_stage=3,
                    lz_complexity=normalized_lz_complexity(terms),
                )
            )

        # Stage 4: Combinatorics & Number Theory (Primes, divisor count, etc.)
        for i in range(num_per_stage):
            oeis_id = f"A{400000 + i:06d}"
            terms = []
            for n in range(1, 51):
                # Divisor count + offset
                divisors = sum(1 for d in range(1, n + 1) if n % d == 0)
                terms.append(divisors + (i % 5))
            records.append(
                SequenceRecord(
                    oeis_id=oeis_id,
                    name=f"Combinatorial divisor count sequence variant {i}",
                    terms=terms,
                    tags=["hard", "base", "eigen"],
                    curriculum_stage=4,
                    lz_complexity=normalized_lz_complexity(terms),
                )
            )

        # Stage 5: Search & Graph Invariants
        for i in range(num_per_stage):
            oeis_id = f"A{500000 + i:06d}"
            terms = [n ^ (n >> 1) for n in range(50)]  # Gray code
            records.append(
                SequenceRecord(
                    oeis_id=oeis_id,
                    name=f"Graph/Search invariant Gray code variant {i}",
                    terms=terms,
                    tags=["hard", "bref", "more"],
                    curriculum_stage=5,
                    lz_complexity=normalized_lz_complexity(terms),
                )
            )

        return self.insert_records(records)
