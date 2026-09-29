"""Reviewed symbolic definitions registry loader and formula lookup service."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
import sympy as sp
from oeis_learn.data.models import SequenceRef


class SymbolicDefinitionRegistry:
    """Manages verified mathematical closed-form definitions for OEIS sequences."""

    def __init__(self, registry_path: str = "data/benchmarks/symbolic_definitions_v1.json"):
        self.registry_path = registry_path
        self.definitions_by_oeis_id: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(registry_path):
            self.load_registry(registry_path)

    def load_registry(self, registry_path: str) -> None:
        """Loads and parses reviewed definitions from JSON."""
        with open(registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for d in data.get("definitions", []):
            seq_ref = d.get("sequence_ref", {})
            oeis_id = seq_ref.get("oeis_id")
            if oeis_id:
                # Validate parseability with SymPy
                expr_str = d.get("expression", "")
                try:
                    n = sp.Symbol("n", integer=True)
                    sp.sympify(expr_str)
                    self.definitions_by_oeis_id[oeis_id] = d
                except Exception as e:
                    raise ValueError(f"Malformed definition for {oeis_id} ({expr_str}): {e}")

    def get_definition(self, oeis_id: str) -> Optional[Dict[str, Any]]:
        """Returns verified definition entry for a sequence ID if available."""
        return self.definitions_by_oeis_id.get(oeis_id)

    def has_definition(self, oeis_id: str) -> bool:
        return oeis_id in self.definitions_by_oeis_id
