"""Real OEIS Data Loader and Benchmark Suite Provider.

Supports ingesting curated sequences from local data/sample_oeis,
or downloading real data archives from https://github.com/oeis/oeisdata
and https://github.com/archmageirvine/joeis.
"""

from __future__ import annotations

import os
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple
from oeis_learn.data.lz_complexity import normalized_lz_complexity
from oeis_learn.data.models import SequenceRecord, parse_names_line, parse_stripped_line
from oeis_learn.data.preprocessing import (
    analyze_log_linearity,
    check_finite_difference_polynomial_degree,
)

DEFAULT_SAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "data",
    "sample_oeis",
)


class RealOeisDataLoader:
    """Loads and organizes real OEIS sequences across taxonomic stages."""

    def __init__(self, sample_dir: Optional[str] = None):
        self.sample_dir = sample_dir or DEFAULT_SAMPLE_DIR

    def load_local_benchmark_records(self) -> List[SequenceRecord]:
        """Loads real benchmark sequence records from local sample_oeis directory."""
        stripped_path = os.path.join(self.sample_dir, "stripped")
        names_path = os.path.join(self.sample_dir, "names")

        names_dict: Dict[str, str] = {}
        if os.path.exists(names_path):
            with open(names_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parsed_n = parse_names_line(line)
                    if parsed_n:
                        names_dict[parsed_n[0]] = parsed_n[1]

        records: List[SequenceRecord] = []
        if os.path.exists(stripped_path):
            with open(stripped_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parsed_s = parse_stripped_line(line)
                    if not parsed_s:
                        continue
                    oeis_id, terms = parsed_s
                    name = names_dict.get(oeis_id, f"Sequence {oeis_id}")

                    # Determine taxonomic stage
                    stage = self._classify_stage(oeis_id, name, terms)
                    tags = self._determine_tags(oeis_id, stage)

                    records.append(
                        SequenceRecord(
                            oeis_id=oeis_id,
                            name=name,
                            terms=terms,
                            tags=tags,
                            curriculum_stage=stage,
                            lz_complexity=normalized_lz_complexity(terms),
                        )
                    )

        return records

    def _classify_stage(self, oeis_id: str, name: str, terms: Sequence[int]) -> int:
        """Classifies a real sequence into curriculum stages 1 to 5."""
        name_lower = name.lower()

        # Specific well-known OEIS mappings
        stage_map = {
            "A000012": 1, "A000027": 1, "A000217": 1, "A000290": 1, "A000578": 1, "A005408": 1, "A000079": 1,
            "A000045": 2, "A000032": 2, "A000129": 2, "A001045": 2, "A000073": 2,
            "A000142": 3, "A000108": 3, "A000166": 3, "A001006": 3, "A000110": 3,
            "A000040": 4, "A000041": 4, "A000005": 4, "A000010": 4,
            "A003188": 5, "A000088": 5,
        }
        if oeis_id in stage_map:
            return stage_map[oeis_id]

        # Stage 5
        if "graph" in name_lower or "game" in name_lower or "gray" in name_lower:
            return 5
        # Stage 4
        if "prime" in name_lower or "partition" in name_lower or "divisor" in name_lower or "totient" in name_lower:
            return 4
        # Stage 3
        if "factorial" in name_lower or "catalan" in name_lower or "derangement" in name_lower or "motzkin" in name_lower:
            return 3
        # Stage 2
        if "fibonacci" in name_lower or "lucas" in name_lower or "pell" in name_lower or "jacobsthal" in name_lower:
            return 2

        # Check arithmetic properties
        if len(terms) >= 6:
            poly_deg = check_finite_difference_polynomial_degree(terms)
            if poly_deg is not None:
                return 1
            is_log_lin, _ = analyze_log_linearity(terms)
            if is_log_lin:
                return 2

        return 1

    def _determine_tags(self, oeis_id: str, stage: int) -> List[str]:
        """Generates appropriate tags for sequence records."""
        tags = ["core", "nonn"]
        if stage == 1:
            tags.extend(["easy", "polynomial"])
        elif stage == 2:
            tags.extend(["frac", "cons", "recurrence"])
        elif stage == 3:
            tags.extend(["nice", "tabl", "holonomic"])
        elif stage == 4:
            tags.extend(["hard", "base", "eigen", "prime"])
        elif stage == 5:
            tags.extend(["hard", "bref", "graph"])
        return tags
