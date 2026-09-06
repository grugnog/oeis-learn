#!/usr/bin/env python3
"""Extracts, indexes, and enriches OEIS corpus data into local storage.

Streams and fuses:
1. OEIS stripped sequence data (terms, counts, signed integers)
2. OEIS sequence names & definitions
3. jOEIS (archmageirvine/joeis) implementation status & structural classes
4. Automatic mathematical tags, complexity, and taxonomic curriculum stage inference

Outputs:
- Parquet / DuckDB database for analytical querying and training
- Standalone enriched JSON / Parquet datasets
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import json
import logging
import os
import re
import sys
import time
import urllib.request
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import duckdb

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger("extract_oeis_data")

DEFAULT_CACHE_DIR = "data/oeis_cache"
OEIS_STRIPPED_URL = "https://oeis.org/stripped.gz"
OEIS_NAMES_URL = "https://oeis.org/names.gz"
JOEIS_TREE_API_URL = "https://api.github.com/repos/archmageirvine/joeis/git/trees/master?recursive=1"


def download_file(url: str, dest_path: str, force: bool = False) -> str:
    """Downloads a file if not already present or forced."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    if os.path.exists(dest_path) and not force:
        logger.info(f"File already cached: {dest_path} ({os.path.getsize(dest_path)/(1024*1024):.1f} MB)")
        return dest_path

    logger.info(f"Downloading {url} -> {dest_path}...")
    req = urllib.request.Request(url, headers={"User-Agent": "oeis-learn-extractor/1.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out_f.write(chunk)
    logger.info(f"Downloaded {dest_path} successfully ({os.path.getsize(dest_path)/(1024*1024):.1f} MB)")
    return dest_path


def fetch_joeis_classes(cache_dir: str = DEFAULT_CACHE_DIR, force: bool = False) -> Dict[str, str]:
    """Fetches or loads the map of OEIS ID -> jOEIS class path."""
    dest_path = os.path.join(cache_dir, "joeis_classes.json")
    if os.path.exists(dest_path) and not force:
        try:
            with open(dest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} jOEIS classes from {dest_path}")
            return data
        except Exception as e:
            logger.warning(f"Could not load cached jOEIS classes: {e}, refetching...")

    logger.info(f"Fetching jOEIS repository tree from GitHub API: {JOEIS_TREE_API_URL}...")
    req = urllib.request.Request(JOEIS_TREE_API_URL, headers={"User-Agent": "oeis-learn-extractor/1.0"})
    class_map: Dict[str, str] = {}
    try:
        with urllib.request.urlopen(req) as resp:
            tree_data = json.load(resp)
        tree = tree_data.get("tree", [])
        for item in tree:
            p = item.get("path", "")
            m = re.match(r"^src/(irvine/oeis/a\d+/(A\d+))\.java$", p)
            if m:
                java_pkg = m.group(1).replace("/", ".")
                oeis_id = m.group(2)
                class_map[oeis_id] = java_pkg
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(class_map, f)
        logger.info(f"Extracted and cached {len(class_map)} jOEIS classes to {dest_path}")
    except Exception as e:
        logger.warning(f"Failed to fetch jOEIS classes from GitHub API: {e}")
        # Check if simpler ID list exists
        alt_path = os.path.join(cache_dir, "joeis_implemented_ids.json")
        if os.path.exists(alt_path):
            with open(alt_path, "r", encoding="utf-8") as f:
                ids = json.load(f)
            class_map = {oid: f"irvine.oeis.{oid.lower()}" for oid in ids}
            logger.info(f"Fell back to {len(class_map)} IDs from {alt_path}")
    return class_map


def stream_names(names_path: str) -> Iterator[Tuple[str, str]]:
    """Streams (oeis_id, name) tuples from names.gz."""
    with gzip.open(names_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("A"):
                continue
            parts = line.split(" ", 1)
            oeis_id = parts[0]
            name = parts[1].strip() if len(parts) > 1 else ""
            yield oeis_id, name


def stream_stripped(stripped_path: str) -> Iterator[Tuple[str, List[int]]]:
    """Streams (oeis_id, terms) tuples from stripped.gz."""
    with gzip.open(stripped_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("A"):
                continue
            parts = line.split(" ,", 1)
            if len(parts) != 2:
                parts = line.split(",", 1)
            if len(parts) < 2:
                continue
            oeis_id = parts[0].strip()
            raw_terms = parts[1].split(",")
            terms: List[int] = []
            for t in raw_terms:
                t = t.strip()
                if not t:
                    continue
                try:
                    terms.append(int(t))
                except ValueError:
                    pass
            if terms:
                yield oeis_id, terms


def infer_tags_and_stage(
    oeis_id: str,
    name: str,
    terms: List[int],
    joeis_class: Optional[str] = None,
) -> Tuple[List[str], int, Optional[str]]:
    """Infers metadata tags, curriculum stage (1..5), and generating formula."""
    tags = ["nonn"]
    name_lower = name.lower()
    formula: Optional[str] = None

    # Formula extraction from name
    m_formula = re.search(r"a\(n\)\s*=\s*([^,\.;]+)", name)
    if m_formula:
        formula = f"a(n) = {m_formula.group(1).strip()}"

    # Curriculum stage classification
    stage = 1

    # Stage 5: Graphs, Games, Bitwise Search Invariants
    if any(k in name_lower for k in ["graph", "game", "gray code", "automaton", "cellular"]):
        stage = 5
        tags.extend(["hard", "bref", "graph"])
    # Stage 4: Primes, Divisors, Partitions, Number Theory
    elif any(k in name_lower for k in ["prime", "divisor", "partition", "totient", "gcd", "lcm", "sigma"]):
        stage = 4
        tags.extend(["hard", "base", "eigen", "prime"])
    # Stage 3: Holonomic, Catalan, Factorial, Hypergeometric, Motzkin
    elif any(k in name_lower for k in ["factorial", "catalan", "holonomic", "derangement", "motzkin", "hypergeometric"]):
        stage = 3
        tags.extend(["nice", "tabl", "holonomic"])
    # Stage 2: Recurrences, Fibonacci, Lucas, Pell, Geometric
    elif any(k in name_lower for k in ["fibonacci", "lucas", "pell", "jacobsthal", "recurrence", "tribonacci", "powers of"]):
        stage = 2
        tags.extend(["frac", "cons", "recurrence"])
    else:
        # Check polynomial differences
        if len(terms) >= 6:
            # Check constant first/second/third differences
            diff1 = [terms[i+1] - terms[i] for i in range(len(terms)-1)]
            if len(set(diff1)) == 1:
                stage = 1
                tags.extend(["easy", "linear"])
            else:
                diff2 = [diff1[i+1] - diff1[i] for i in range(len(diff1)-1)]
                if len(set(diff2)) == 1:
                    stage = 1
                    tags.extend(["easy", "quadratic"])
                else:
                    diff3 = [diff2[i+1] - diff2[i] for i in range(len(diff2)-1)]
                    if len(set(diff3)) == 1:
                        stage = 1
                        tags.extend(["easy", "cubic"])
                    else:
                        tags.extend(["core", "easy"])
        else:
            tags.extend(["core", "easy"])

    if joeis_class:
        tags.append("joeis")

    return list(dict.fromkeys(tags)), stage, formula


def run_extraction(
    cache_dir: str = DEFAULT_CACHE_DIR,
    output_db: str = "data/oeis_corpus.duckdb",
    output_summary_json: Optional[str] = "reports/oeis_extraction_summary.json",
    limit: Optional[int] = None,
    force_download: bool = False,
) -> Dict[str, Any]:
    """Runs the full OEIS data extraction, indexing, and DuckDB storage pipeline."""
    start_time = time.time()
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_db)), exist_ok=True)

    # 1. Acquire raw data assets
    stripped_path = os.path.join(cache_dir, "stripped.gz")
    names_path = os.path.join(cache_dir, "names.gz")

    download_file(OEIS_STRIPPED_URL, stripped_path, force=force_download)
    download_file(OEIS_NAMES_URL, names_path, force=force_download)
    joeis_map = fetch_joeis_classes(cache_dir=cache_dir, force=force_download)

    # 2. Ingest names dictionary into memory
    logger.info("Indexing sequence names from names.gz...")
    names_dict: Dict[str, str] = {}
    for oeis_id, name in stream_names(names_path):
        names_dict[oeis_id] = name
    logger.info(f"Loaded {len(names_dict):,} sequence names into lookup index.")

    # 3. Setup DuckDB schema
    logger.info(f"Initializing DuckDB table schema in {output_db}...")
    conn = duckdb.connect(output_db)
    conn.execute("DROP TABLE IF EXISTS sequences")
    conn.execute("""
        CREATE TABLE sequences (
            oeis_id VARCHAR PRIMARY KEY,
            name VARCHAR,
            terms_json VARCHAR,
            term_count INTEGER,
            tags VARCHAR,
            curriculum_stage INTEGER,
            joeis_class VARCHAR,
            generating_formula VARCHAR,
            max_term_abs VARCHAR,
            max_term_bits INTEGER,
            has_joeis BOOLEAN,
            has_formula BOOLEAN
        )
    """)

    # 4. Stream stripped data, fuse signals, batch insert into DuckDB
    logger.info("Streaming and enriching sequences from stripped.gz...")
    batch: List[Tuple[Any, ...]] = []
    batch_size = 10000
    total_processed = 0
    stage_counts: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    joeis_matches = 0
    formula_matches = 0

    insert_sql = """
        INSERT INTO sequences VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    for oeis_id, terms in stream_stripped(stripped_path):
        name = names_dict.get(oeis_id, f"Sequence {oeis_id}")
        j_class = joeis_map.get(oeis_id)
        tags, stage, formula = infer_tags_and_stage(oeis_id, name, terms, joeis_class=j_class)

        # Bit width & maximum term
        max_abs = max(abs(x) for x in terms) if terms else 0
        max_bits = max_abs.bit_length()

        has_j = j_class is not None
        has_f = formula is not None
        if has_j:
            joeis_matches += 1
        if has_f:
            formula_matches += 1
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

        batch.append((
            oeis_id,
            name,
            json.dumps(terms),
            len(terms),
            ",".join(tags),
            stage,
            j_class,
            formula,
            str(max_abs),
            max_bits,
            has_j,
            has_f,
        ))

        total_processed += 1
        if len(batch) >= batch_size:
            conn.executemany(insert_sql, batch)
            batch.clear()
            if total_processed % 50000 == 0:
                logger.info(f"Processed {total_processed:,} sequences (jOEIS matches: {joeis_matches:,})...")

        if limit is not None and total_processed >= limit:
            break

    if batch:
        conn.executemany(insert_sql, batch)
        batch.clear()

    total_duration = time.time() - start_time
    logger.info(f"Extraction complete: {total_processed:,} sequences stored in {output_db} in {total_duration:.1f}s.")
    logger.info(f"Coverage Summary: jOEIS implementations = {joeis_matches:,} ({joeis_matches/total_processed*100:.1f}%), Formulas = {formula_matches:,} ({formula_matches/total_processed*100:.1f}%)")
    logger.info(f"Stage breakdown: {stage_counts}")

    summary = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_sequences": total_processed,
        "duration_seconds": total_duration,
        "database_path": output_db,
        "joeis_implementations": joeis_matches,
        "joeis_coverage_pct": round(joeis_matches / max(1, total_processed) * 100, 2),
        "formula_matches": formula_matches,
        "formula_coverage_pct": round(formula_matches / max(1, total_processed) * 100, 2),
        "stage_distribution": stage_counts,
    }

    if output_summary_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_summary_json)), exist_ok=True)
        with open(output_summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved extraction summary to {output_summary_json}")

    conn.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Extract and enrich OEIS data into local storage.")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR, help="Local cache directory for raw files.")
    parser.add_argument("--output-db", type=str, default="data/oeis_corpus.duckdb", help="DuckDB output file path.")
    parser.add_argument("--summary", type=str, default="reports/oeis_extraction_summary.json", help="Summary JSON output.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for rapid testing.")
    parser.add_argument("--force-download", action="store_true", help="Force re-downloading raw files.")

    args = parser.parse_args()
    run_extraction(
        cache_dir=args.cache_dir,
        output_db=args.output_db,
        output_summary_json=args.summary,
        limit=args.limit,
        force_download=args.force_download,
    )


if __name__ == "__main__":
    main()
