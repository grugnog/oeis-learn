#!/usr/bin/env python3
"""Exact horizon census; the explicit database is opened read-only."""

import argparse
import json
from pathlib import Path

import duckdb
from oeis_learn.evaluation.source_policy import (
    analyze_sequence_cohort,
    bit_width,  # noqa: F401 (historical API)
    classify_growth_profile,  # noqa: F401 (historical API)
)
from oeis_learn.experiments.artifacts import compute_file_hash


def generate_markdown_report(analysis):
    lines = [
        "# Source precision census",
        "",
        "Coverage uses complete contiguous horizons only.",
        "Growth labels describe the entire observed series; they are finite descriptions.",
        "",
        f"Database: {analysis.get('database_path', 'unavailable')}",
        "",
        "| Cohort | Horizon | Complete denominator | Signed256 covered |",
        "| --- | --- | ---: | ---: |",
    ]
    for name, cohort in analysis.get("cohorts", {}).items():
        for horizon, row in cohort["horizons"].items():
            lines.append(
                f"| {name} | {horizon} | {row['sequences_evaluated']} | {row['coverage_counts']['signed256']} |"
            )
    return "\n".join(lines) + "\n"


def run_precision_analysis(
    db_path,
    output_json="reports/oeis_precision_census.json",
    output_md="reports/oeis_precision_census.md",
):
    path = Path(db_path).resolve(strict=True)
    records, invalid = [], []
    with duckdb.connect(str(path), read_only=True) as con:
        cursor = con.execute("SELECT * FROM sequences")
        columns = [d[0] for d in cursor.description]
        if not {"oeis_id", "terms_json"} <= set(columns):
            raise ValueError("sequences requires oeis_id and terms_json")
        for values in cursor.fetchall():
            row = dict(zip(columns, values))
            try:
                terms = json.loads(row["terms_json"])
                if not isinstance(terms, list) or any(type(v) is not int for v in terms):
                    raise ValueError("noninteger terms")
                records.append({**row, "terms": terms, "offset": row.get("offset")})
            except (ValueError, TypeError):
                invalid.append(row["oeis_id"])
    census = analyze_sequence_cohort(records)
    report = {
        "schema_version": "source-census/v1",
        "database_path": str(path),
        "source_sha256": compute_file_hash(path),
        "total_sequences": len(records) + len(invalid),
        "invalid_records": len(invalid),
        "invalid_record_ids": invalid,
        "source_metadata": census["source_metadata"],
        "cohorts": {"All Sequences": census},
    }
    for output, data in (
        (output_json, json.dumps(report, indent=2)),
        (output_md, generate_markdown_report(report)),
    ):
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(data)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--output-json", default="reports/oeis_precision_census.json")
    parser.add_argument("--output-md", default="reports/oeis_precision_census.md")
    args = parser.parse_args()
    run_precision_analysis(args.db, args.output_json, args.output_md)


if __name__ == "__main__":
    main()
