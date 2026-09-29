"""Minimum Description Length (MDL) Anti-Memorization Complexity Verifier."""

from __future__ import annotations

import math
from typing import Sequence, Tuple, Union
from oeis_learn.data.lz_complexity import lempel_ziv_complexity, normalized_lz_complexity


from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union
from oeis_learn.data.lz_complexity import lempel_ziv_complexity, normalized_lz_complexity


@dataclass
class MdlAssessmentRecord:
    """Detailed evidence of program compactness and MDL assessment."""

    passed: bool
    byte_size: int
    canonical_byte_size: int
    target_complexity: float
    mdl_ratio: float
    threshold: float
    is_table_memorized: bool


def get_wat_byte_size(wat_code: str) -> int:
    """Computes binary WASM or stripped WAT byte size."""
    try:
        import wasmtime
        wasm_bytes = wasmtime.wat2wasm(wat_code)
        return len(wasm_bytes)
    except Exception:
        # Fallback to stripped text bytes
        return len(wat_code.encode("utf-8"))


class MdlVerifier:
    """Verifies that synthesized programs satisfy Minimum Description Length complexity bounds

    (M_MDL <= 1.2) to reject hardcoded lookup tables and memorized polynomial fits.
    """

    def __init__(self, threshold: float = 1.2, min_byte_baseline: int = 40):
        self.threshold = threshold
        self.min_byte_baseline = min_byte_baseline

    def compute_mdl_ratio(self, wat_code: str, sequence: Union[Sequence[int], str]) -> Tuple[float, int, float]:
        """Computes MDL ratio = (WASM byte size) / (LZ complexity proxy * scaling).

        Returns:
            Tuple of (mdl_ratio, byte_size, lz_complexity)
        """
        byte_size = get_wat_byte_size(wat_code)
        lz_comp = lempel_ziv_complexity(sequence)

        # Baseline Kolmogorov proxy for N terms
        lz_proxy_bytes = max(float(self.min_byte_baseline), float(lz_comp * 8.0))
        mdl_ratio = byte_size / lz_proxy_bytes

        return float(mdl_ratio), byte_size, float(lz_comp)

    def assess_compactness(
        self,
        wat_code: str,
        sequence: Union[Sequence[int], str],
        canonical_wat: Optional[str] = None,
    ) -> MdlAssessmentRecord:
        """Returns structured compactness evidence without conflating compactness with correctness."""
        byte_size = get_wat_byte_size(wat_code)
        canonical_bytes = get_wat_byte_size(canonical_wat) if canonical_wat else byte_size

        lz_comp = lempel_ziv_complexity(sequence)
        lz_proxy_bytes = max(float(self.min_byte_baseline), float(lz_comp * 8.0))
        ratio = float(canonical_bytes / lz_proxy_bytes)

        is_table = ("table" in wat_code.lower()) or (wat_code.count("i64.const") > 25)
        passed = (ratio <= self.threshold) and not is_table

        return MdlAssessmentRecord(
            passed=passed,
            byte_size=byte_size,
            canonical_byte_size=canonical_bytes,
            target_complexity=float(lz_comp),
            mdl_ratio=ratio,
            threshold=self.threshold,
            is_table_memorized=is_table,
        )

    def verify(self, wat_code: str, sequence: Union[Sequence[int], str]) -> bool:
        """Returns True if the program passes MDL anti-memorization threshold (mdl_ratio <= threshold)."""
        record = self.assess_compactness(wat_code, sequence)
        return record.passed
