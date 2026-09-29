"""Template-driven forward synthetic program and sequence generator for SFT warmup."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
from oeis_learn.curriculum.mdl_verifier import MdlVerifier
from oeis_learn.data.models import SyntheticDemonstrationPair
from oeis_learn.sandbox.runner import WasmRunner


@dataclass
class SyntheticDemonstrationDataset:
    """Collection of synthetic forward demonstration pairs for SFT warmup."""

    version: str
    total_samples: int
    samples: List[SyntheticDemonstrationPair]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "total_samples": self.total_samples,
            "samples": [s.to_dict() for s in self.samples],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SyntheticDemonstrationDataset:
        samples = [SyntheticDemonstrationPair.from_dict(s) for s in data["samples"]]
        return cls(
            version=data.get("version", "1.0.0"),
            total_samples=data.get("total_samples", len(samples)),
            samples=samples,
        )


class SyntheticDemonstrationGenerator:
    """Generates valid WAT programs paired with their executed integer sequence terms."""

    def __init__(
        self,
        seed: int = 42,
        fuel_budget: int = 20000,
        enable_affine_sweeps: bool = True,
        scale_min_pow: float = 0.0,
        scale_max_pow: float = 5.0,
    ):
        self.rng = random.Random(seed)
        self.wasm_runner = WasmRunner(fuel_budget=fuel_budget)
        self.mdl_verifier = MdlVerifier()
        self.enable_affine_sweeps = enable_affine_sweeps
        self.scale_min_pow = scale_min_pow
        self.scale_max_pow = scale_max_pow

    def generate_sample(self, sample_idx: int, family: Optional[str] = None) -> Optional[SyntheticDemonstrationPair]:
        """Generates a single valid synthetic demonstration pair."""
        families = [
            "POLYNOMIAL_LINEAR",
            "POLYNOMIAL_QUADRATIC",
            "POLYNOMIAL_CUBIC",
            "RECURRENCE_ORDER1",
            "RECURRENCE_FIBONACCI",
            "HOLONOMIC_FACTORIAL",
            "MODULAR_PERIODIC",
        ]
        chosen_family = family or self.rng.choice(families)

        sample_id = f"SYNTH_{chosen_family}_{sample_idx:06d}"
        wat_code, metadata = self._generate_wat_for_family(chosen_family)

        # Execute program to obtain ground truth terms
        n_terms = 20
        res = self.wasm_runner.run_single(wat_code, terms_to_generate=n_terms)
        if res.status != "SUCCESS" or len(res.output) < n_terms:
            return None

        terms = res.output
        _, byte_size, lz = self.mdl_verifier.compute_mdl_ratio(wat_code, terms)

        return SyntheticDemonstrationPair(
            sample_id=sample_id,
            family=chosen_family,
            terms=terms,
            wat_code=wat_code,
            byte_size=byte_size,
            lz_complexity=lz,
            metadata=metadata,
        )

    def _generate_wat_for_family(self, family: str) -> Tuple[str, Dict[str, Any]]:
        """Generates template WAT code and metadata for a specific family with affine scaling sweeps."""
        # Draw dynamic affine multiplier if enabled
        if self.enable_affine_sweeps:
            pow_scale = self.rng.uniform(self.scale_min_pow, self.scale_max_pow)
            scale_mult = max(1, int(10**pow_scale * self.rng.uniform(0.5, 2.0)))
            if self.rng.random() < 0.2:
                scale_mult = -scale_mult
        else:
            scale_mult = 1

        if family == "POLYNOMIAL_LINEAR":
            a = self.rng.randint(1, 10) * scale_mult
            b = self.rng.randint(-50, 100)
            if b >= 0:
                wat = f'(module (func (export "compute") (param $n i32) (result i64) local.get $n i64.extend_i32_u i64.const {a} i64.mul i64.const {b} i64.add))'
            else:
                wat = f'(module (func (export "compute") (param $n i32) (result i64) local.get $n i64.extend_i32_u i64.const {a} i64.mul i64.const {abs(b)} i64.sub))'
            return wat, {"a": a, "b": b}

        elif family == "POLYNOMIAL_QUADRATIC":
            variant = self.rng.choice(["triangular", "square", "general"])
            if variant == "triangular":
                wat = '(module (func (export "compute") (param $n i32) (result i64) local.get $n i64.extend_i32_u local.get $n i64.extend_i32_u i64.const 1 i64.add i64.mul i64.const 2 i64.div_u))'
                return wat, {"variant": "triangular"}
            elif variant == "square":
                c = self.rng.randint(0, 50)
                wat = f'(module (func (export "compute") (param $n i32) (result i64) local.get $n i64.extend_i32_u local.get $n i64.extend_i32_u i64.mul i64.const {scale_mult} i64.mul i64.const {c} i64.add))'
                return wat, {"variant": "square", "c": c, "scale": scale_mult}
            else:
                a = self.rng.randint(1, 4) * (scale_mult // max(1, abs(scale_mult)))
                b = self.rng.randint(1, 6)
                c = self.rng.randint(0, 10)
                wat = f'(module (func (export "compute") (param $n i32) (result i64) local.get $n i64.extend_i32_u local.get $n i64.extend_i32_u i64.mul i64.const {a} i64.mul local.get $n i64.extend_i32_u i64.const {b} i64.mul i64.add i64.const {c} i64.add))'
                return wat, {"variant": "general", "a": a, "b": b, "c": c}

        elif family == "POLYNOMIAL_CUBIC":
            wat = '(module (func (export "compute") (param $n i32) (result i64) local.get $n i64.extend_i32_u local.get $n i64.extend_i32_u local.get $n i64.extend_i32_u i64.mul i64.mul))'
            return wat, {"degree": 3}

        elif family == "RECURRENCE_ORDER1":
            # Geometric powers e.g. 2^n or 3^n via loop
            base = self.rng.choice([2, 3, 5])
            init_c = self.rng.choice([1, 2, 3])
            wat = f'(module (func (export "compute") (param $n i32) (result i64) (local $res i64) (local $i i32) i64.const {init_c} local.set $res i32.const 0 local.set $i (block $exit (loop $loop local.get $i local.get $n i32.ge_s br_if $exit local.get $res i64.const {base} i64.mul local.set $res local.get $i i32.const 1 i32.add local.set $i br $loop)) local.get $res))'
            return wat, {"base": base, "init": init_c}

        elif family == "RECURRENCE_FIBONACCI":
            a0 = self.rng.choice([0, 1, 2, 3])
            b0 = self.rng.choice([1, 2, 3, 4])
            c1 = self.rng.choice([1, 2])
            c2 = self.rng.choice([1, 2])
            if c1 == 1 and c2 == 1:
                next_expr = "local.get $a local.get $b i64.add"
            elif c1 == 2 and c2 == 1:
                next_expr = "local.get $a local.get $b i64.const 2 i64.mul i64.add"
            else:
                next_expr = f"local.get $a i64.const {c2} i64.mul local.get $b i64.const {c1} i64.mul i64.add"

            wat = f'(module (func (export "compute") (param $n i32) (result i64) (local $a i64) (local $b i64) (local $temp i64) (local $i i32) i64.const {a0} local.set $a i64.const {b0} local.set $b i32.const 0 local.set $i (block $exit (loop $loop local.get $i local.get $n i32.ge_s br_if $exit {next_expr} local.set $temp local.get $b local.set $a local.get $temp local.set $b local.get $i i32.const 1 i32.add local.set $i br $loop)) local.get $a))'
            return wat, {"a0": a0, "b0": b0, "c1": c1, "c2": c2}

        elif family == "HOLONOMIC_FACTORIAL":
            offset = self.rng.choice([0, 1, 2])
            a0 = 1
            wat = f'(module (func (export "compute") (param $n i32) (result i64) (local $a i64) (local $b i64) (local $temp i64) (local $i i32) i64.const {a0} local.set $a i32.const 1 local.set $i (block $exit (loop $loop local.get $i local.get $n i32.gt_s br_if $exit local.get $a local.get $i {"i32.const " + str(offset) + " i32.add " if offset != 0 else ""}i64.extend_i32_u i64.mul local.set $a local.get $i i32.const 1 i32.add local.set $i br $loop)) local.get $a))'
            return wat, {"offset": offset, "a0": a0}

        elif family == "MODULAR_PERIODIC":
            m = self.rng.choice([2, 3, 5, 7, 10])
            wat = f'(module (func (export "compute") (param $n i32) (result i64) local.get $n i64.extend_i32_u i64.const {m} i64.rem_u))'
            return wat, {"modulus": m}

        else:
            # Fallback constant sequence
            c = self.rng.randint(1, 100)
            wat = f'(module (func (export "compute") (param $n i32) (result i64) i64.const {c}))'
            return wat, {"constant": c}

    def generate_multilimb_sample(self, sample_idx: int) -> Optional[SyntheticDemonstrationPair]:
        """Generates a verified multi-limb demonstration pair with signed coefficients in [-3, 3]."""
        families = [
            "MULTILIMB_POLYNOMIAL_SIGNED",
            "MULTILIMB_ALTERNATING_SIGNED",
            "MULTILIMB_RECURRENCE_ORDER1",
            "MULTILIMB_RECURRENCE_ORDER2_SIGNED",
            "MULTILIMB_RECURRENCE_ORDER3",
            "MULTILIMB_HOLONOMIC_SIGNED",
        ]
        chosen_family = self.rng.choice(families)
        sample_id = f"SYNTH_MULTILIMB_{sample_idx:06d}"

        if chosen_family == "MULTILIMB_POLYNOMIAL_SIGNED":
            a = self.rng.choice([-3, -2, -1, 1, 2, 3])
            b = self.rng.choice([-3, -2, -1, 0, 1, 2, 3])
            c = self.rng.choice([-5, -3, 0, 1, 4, 7])
            # a*n^2 + b*n + c with dynamic sign extension
            wat = f"""(module
  (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
    (local $n64 i64) (local $val i64) (local $sign i64)
    local.get $n i64.extend_i32_u local.set $n64
    local.get $n64 local.get $n64 i64.mul i64.const {a} i64.mul
    local.get $n64 i64.const {b} i64.mul i64.add i64.const {c} i64.add
    local.set $val
    local.get $val i64.const 63 i64.shr_s local.set $sign
    local.get $val local.get $sign local.get $sign local.get $sign
  )
)"""
            metadata = {"family": chosen_family, "a": a, "b": b, "c": c}

        elif chosen_family == "MULTILIMB_ALTERNATING_SIGNED":
            # a(n) = (-1)^n * (2n + 1) with dynamic sign extension
            wat = """(module
  (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
    (local $val i64) (local $sign i64) (local $j i32)
    local.get $n i64.extend_i32_u i64.const 2 i64.mul i64.const 1 i64.add local.set $val
    local.get $n i32.const 1 i32.and local.set $j
    local.get $j
    if
      i64.const 0 local.get $val i64.sub local.set $val
    end
    local.get $val i64.const 63 i64.shr_s local.set $sign
    local.get $val local.get $sign local.get $sign local.get $sign
  )
)"""
            metadata = {"family": chosen_family, "pattern": "alternating"}

        elif chosen_family == "MULTILIMB_RECURRENCE_ORDER1":
            base = self.rng.choice([2, 3, -2])
            wat = f"""(module
  (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
    (local $a0 i64) (local $a1 i64) (local $a2 i64) (local $a3 i64)
    (local $i i32)
    i256.const 1 local.set $a3 local.set $a2 local.set $a1 local.set $a0
    (block $exit
      (loop $loop
        local.get $i local.get $n i32.ge_s br_if $exit
        local.get $a0 local.get $a1 local.get $a2 local.get $a3
        i64.const {base}
        i256.mul_scalar
        local.set $a3 local.set $a2 local.set $a1 local.set $a0
        local.get $i i32.const 1 i32.add local.set $i
        br $loop
      )
    )
    local.get $a0 local.get $a1 local.get $a2 local.get $a3
  )
)"""
            metadata = {"family": chosen_family, "base": base}

        elif chosen_family == "MULTILIMB_RECURRENCE_ORDER2_SIGNED":
            # a(n) = c1*a(n-1) + c2*a(n-2) with signed c1, c2 in [-3, 3]
            c1 = self.rng.choice([1, 1, 2, 3, -1])
            c2 = self.rng.choice([1, 1, -1, 2])
            a0 = self.rng.choice([0, 1, 2, 3])
            b0 = self.rng.choice([1, 1, 2, -1])

            term_b = "local.get $b0 local.get $b1 local.get $b2 local.get $b3"
            if c1 != 1:
                term_b += f"\n        i64.const {c1}\n        i256.mul_scalar"

            term_a = "local.get $a0 local.get $a1 local.get $a2 local.get $a3"
            if c2 != 1:
                term_a += f"\n        i64.const {c2}\n        i256.mul_scalar"

            wat = f"""(module
  (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
    (local $a0 i64) (local $a1 i64) (local $a2 i64) (local $a3 i64)
    (local $b0 i64) (local $b1 i64) (local $b2 i64) (local $b3 i64)
    (local $t0 i64) (local $t1 i64) (local $t2 i64) (local $t3 i64)
    (local $i i32)
    i256.const {a0} local.set $a3 local.set $a2 local.set $a1 local.set $a0
    i256.const {b0} local.set $b3 local.set $b2 local.set $b1 local.set $b0
    (block $exit
      (loop $loop
        local.get $i local.get $n i32.ge_s br_if $exit
        {term_b}
        {term_a}
        i256.add
        local.set $t3 local.set $t2 local.set $t1 local.set $t0
        local.get $b0 local.set $a0 local.get $b1 local.set $a1
        local.get $b2 local.set $a2 local.get $b3 local.set $a3
        local.get $t0 local.set $b0 local.get $t1 local.set $b1
        local.get $t2 local.set $b2 local.get $t3 local.set $b3
        local.get $i i32.const 1 i32.add local.set $i
        br $loop
      )
    )
    local.get $a0 local.get $a1 local.get $a2 local.get $a3
  )
)"""
            metadata = {"family": chosen_family, "c1": c1, "c2": c2, "a0": a0, "b0": b0}

        elif chosen_family == "MULTILIMB_RECURRENCE_ORDER3":
            wat = """(module
  (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
    (local $a0 i64) (local $a1 i64) (local $a2 i64) (local $a3 i64)
    (local $b0 i64) (local $b1 i64) (local $b2 i64) (local $b3 i64)
    (local $c0 i64) (local $c1 i64) (local $c2 i64) (local $c3 i64)
    (local $t0 i64) (local $t1 i64) (local $t2 i64) (local $t3 i64)
    (local $i i32)
    i256.zero local.set $a3 local.set $a2 local.set $a1 local.set $a0
    i256.zero local.set $b3 local.set $b2 local.set $b1 local.set $b0
    i256.const 1 local.set $c3 local.set $c2 local.set $c1 local.set $c0
    (block $exit
      (loop $loop
        local.get $i local.get $n i32.ge_s br_if $exit
        local.get $c0 local.get $c1 local.get $c2 local.get $c3
        local.get $b0 local.get $b1 local.get $b2 local.get $b3
        i256.add
        local.get $a0 local.get $a1 local.get $a2 local.get $a3
        i256.add
        local.set $t3 local.set $t2 local.set $t1 local.set $t0
        local.get $b0 local.set $a0 local.get $b1 local.set $a1
        local.get $b2 local.set $a2 local.get $b3 local.set $a3
        local.get $c0 local.set $b0 local.get $c1 local.set $b1
        local.get $c2 local.set $b2 local.get $c3 local.set $b3
        local.get $t0 local.set $c0 local.get $t1 local.set $c1
        local.get $t2 local.set $c2 local.get $t3 local.set $c3
        local.get $i i32.const 1 i32.add local.set $i
        br $loop
      )
    )
    local.get $a0 local.get $a1 local.get $a2 local.get $a3
  )
)"""
            metadata = {"family": chosen_family, "type": "tribonacci"}

        else:
            # HOLONOMIC
            wat = """(module
  (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
    (local $a0 i64) (local $a1 i64) (local $a2 i64) (local $a3 i64)
    (local $i i32)
    i256.const 1 local.set $a3 local.set $a2 local.set $a1 local.set $a0
    i32.const 1 local.set $i
    (block $exit
      (loop $loop
        local.get $i local.get $n i32.gt_s br_if $exit
        local.get $a0 local.get $a1 local.get $a2 local.get $a3
        local.get $i i64.extend_i32_u
        i256.mul_scalar
        local.set $a3 local.set $a2 local.set $a1 local.set $a0
        local.get $i i32.const 1 i32.add local.set $i
        br $loop
      )
    )
    local.get $a0 local.get $a1 local.get $a2 local.get $a3
  )
)"""
            metadata = {"family": chosen_family, "type": "factorial"}

        # Execute in multi-limb profile
        res = self.wasm_runner.run_single(wat, terms_to_generate=20, result_profile="i256x4_v1")
        if res.status != "SUCCESS" or len(res.output) < 20:
            return None

        terms = res.output
        _, byte_size, lz = self.mdl_verifier.compute_mdl_ratio(wat, terms)

        return SyntheticDemonstrationPair(
            sample_id=sample_id,
            family=chosen_family,
            terms=terms,
            wat_code=wat,
            byte_size=byte_size,
            lz_complexity=lz,
            metadata=metadata,
        )

    def generate_multilimb_dataset(self, num_samples: int = 20000) -> SyntheticDemonstrationDataset:
        """Generates a dataset of verified multi-limb demonstrations balanced across Stages 1-3."""
        samples: List[SyntheticDemonstrationPair] = []
        sample_idx = 0
        attempts = 0
        max_attempts = num_samples * 2

        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1
            sample = self.generate_multilimb_sample(sample_idx)
            if sample is not None:
                samples.append(sample)
                sample_idx += 1

        return SyntheticDemonstrationDataset(
            version="1.0.0",
            total_samples=len(samples),
            samples=samples,
        )

    def generate_dataset(self, num_samples: int = 5000) -> SyntheticDemonstrationDataset:
        """Generates a comprehensive dataset of synthetic demonstration pairs."""
        samples: List[SyntheticDemonstrationPair] = []
        sample_idx = 0
        attempts = 0
        max_attempts = num_samples * 3

        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1
            sample = self.generate_sample(sample_idx)
            if sample is not None:
                samples.append(sample)
                sample_idx += 1

        return SyntheticDemonstrationDataset(
            version="1.0.0",
            total_samples=len(samples),
            samples=samples,
        )

    def save_dataset(self, dataset: SyntheticDemonstrationDataset, file_path: str) -> None:
        """Saves synthetic demonstration dataset to a JSON file."""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dataset.to_dict(), f, indent=2)

    def load_dataset(self, file_path: str) -> SyntheticDemonstrationDataset:
        """Loads synthetic demonstration dataset from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SyntheticDemonstrationDataset.from_dict(data)
