"""PyTorch Dataset and Collation Utilities for OEIS Sequences."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import duckdb
import torch
from torch.utils.data import Dataset
from oeis_learn.data.models import SequenceRecord


class OeisSequenceDataset(Dataset):
    """PyTorch Dataset wrapping OEIS sequences queried from DuckDB or SQLite."""

    def __init__(
        self,
        db_path: str = "data/oeis_learn.duckdb",
        stage_subset: Optional[Sequence[int]] = None,
        max_seq_len: int = 50,
        records: Optional[List[SequenceRecord]] = None,
    ):
        self.max_seq_len = max_seq_len
        self.records: List[SequenceRecord] = []

        if records is not None:
            if stage_subset is not None:
                stage_set = set(stage_subset)
                self.records = [r for r in records if r.curriculum_stage in stage_set]
            else:
                self.records = list(records)
        else:
            if db_path.endswith((".sqlite", ".db")):
                import sqlite3
                sqlite_conn = sqlite3.connect(db_path)
                query = "SELECT oeis_id, name, terms_json, tags, curriculum_stage, joeis_class, generating_formula, lz_complexity FROM sequences"
                if stage_subset is not None:
                    stages_str = ",".join(map(str, stage_subset))
                    query += f" WHERE curriculum_stage IN ({stages_str})"
                query += " ORDER BY oeis_id"
                rows = sqlite_conn.execute(query).fetchall()
                sqlite_conn.close()
            else:
                duck_conn = duckdb.connect(db_path)
                query = "SELECT oeis_id, name, terms_json, tags, curriculum_stage, joeis_class, generating_formula, lz_complexity FROM sequences"
                if stage_subset is not None:
                    stages_str = ",".join(map(str, stage_subset))
                    query += f" WHERE curriculum_stage IN ({stages_str})"
                query += " ORDER BY oeis_id"
                rows = duck_conn.execute(query).fetchall()
                duck_conn.close()

            for row in rows:
                terms = json.loads(row[2])
                tags = [t for t in row[3].split(",") if t]
                self.records.append(
                    SequenceRecord(
                        oeis_id=row[0],
                        name=row[1],
                        terms=terms,
                        tags=tags,
                        curriculum_stage=row[4],
                        joeis_class=row[5],
                        generating_formula=row[6],
                        lz_complexity=row[7] if row[7] is not None else 0.0,
                    )
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        terms = rec.terms[: self.max_seq_len]
        return {
            "oeis_id": rec.oeis_id,
            "name": rec.name,
            "terms": terms,
            "length": len(terms),
            "stage": rec.curriculum_stage,
            "lz_complexity": rec.lz_complexity,
        }


def collate_sequence_batch(batch: List[Dict[str, Any]], pad_value: int = 0) -> Dict[str, Any]:
    """Collates a list of sequence item dictionaries into padded batch tensors."""
    oeis_ids = [item["oeis_id"] for item in batch]
    names = [item["name"] for item in batch]
    stages = torch.tensor([item["stage"] for item in batch], dtype=torch.long)
    lz_complexities = torch.tensor([item["lz_complexity"] for item in batch], dtype=torch.float32)
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)

    max_len = max(item["length"] for item in batch)
    raw_terms_list = [item["terms"] for item in batch]

    # Create padded integer tensor (using torch.float64 or list of ints for extreme values)
    # Since terms can be astronomical, we also store raw python ints
    padded_terms = []
    for terms in raw_terms_list:
        padded = list(terms) + [pad_value] * (max_len - len(terms))
        padded_terms.append(padded)

    return {
        "oeis_ids": oeis_ids,
        "names": names,
        "stages": stages,
        "lz_complexities": lz_complexities,
        "lengths": lengths,
        "raw_terms": raw_terms_list,
        "padded_terms": padded_terms,
    }
