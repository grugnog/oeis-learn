"""Elite Seed Demonstration Replay Buffer for Trajectory Injection and SYMPLE Vulnerability Replay."""

from __future__ import annotations

import hashlib
import logging
import random
from typing import Dict, List, Optional, Tuple
import numpy as np
from oeis_learn.data.models import EliteDemonstrationEntry, EliteReplayBufferEntry
from oeis_learn.decoder.wat_grammar import tokenize_wat

logger = logging.getLogger("oeis_learn.elite_buffer")


class EliteSeedDemonstrationBuffer:
    """Maintains an associative memory buffer of verified canonical programs for reference trajectory injection

    and SYMPLE vulnerability-weighted SFT replay.
    """

    def __init__(self, capacity_per_seq: int = 4, context_limit: int = 1024):
        if not 1 <= capacity_per_seq <= 4:
            raise ValueError("at most four training implementations per output")
        self.capacity_per_seq = capacity_per_seq
        self.context_limit = context_limit
        self.training_entries = set()
        self.entries: Dict[str, EliteReplayBufferEntry] = {}
        self.canonical_archive: Dict[str, List[EliteDemonstrationEntry]] = {}
        self.last_visited: Dict[str, int] = {}
        self.last_active_visit: Dict[str, int] = {}
        self.last_replay_visit: Dict[str, int] = {}
        self.discovery_step: Dict[str, int] = {}
        self.last_verified_step: Dict[str, int] = {}
        # Legacy canonical teachers are inspection-only and never implicitly seeded.

    def record_active_visit(self, oeis_id: str, step: int) -> None:
        """Records active task exploration step."""
        self.last_active_visit[oeis_id] = step
        self.last_visited[oeis_id] = step

    def record_replay_visit(self, oeis_id: str, step: int) -> None:
        """Records replay review step without resetting active exploration history."""
        self.last_replay_visit[oeis_id] = step

    def add_entry(self, entry: EliteReplayBufferEntry) -> None:
        """Adds or updates a verified canonical program entry."""
        self.add_canonical_entry(entry.oeis_id, entry.wat_code, entry.terms, step=0)

    def add_canonical_entry(
        self,
        oeis_id: str,
        wat_code: str,
        terms: List[int],
        fuel: int = 0,
        step: int = 0,
    ) -> None:
        """Stores a verified canonical AST program in the sequence archive, bounded by capacity and deduplicated."""
        from oeis_learn.sandbox.runner import WasmRunner
        from oeis_learn.decoder.grounding_reference import scalar_value
        from oeis_learn.discovery.program_archive import eligibility

        if not 1 <= len(terms) <= 100 or any(type(v) is not int for v in terms):
            raise ValueError("real exact conditioning/validation terms required")
        eligible, _, _ = eligibility(wat_code, self.context_limit, codec="legacy")
        # Independent straight-line scalar inspection. Other legacy forms must
        # use strict archive admission; native success alone is not evidence.
        reference = [scalar_value(wat_code, i) for i in range(len(terms))]
        result = WasmRunner().run_single(wat_code, terms_to_generate=len(terms))
        if reference != terms or result.status != "SUCCESS" or result.output != terms:
            raise ValueError("legacy admission mismatch; archive views unchanged")
        tokens = tokenize_wat(wat_code)
        fuel = result.total_fuel
        canonical_str = " ".join(tokens)
        ast_hash = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

        if oeis_id not in self.canonical_archive:
            self.canonical_archive[oeis_id] = []

        existing = self.canonical_archive[oeis_id]
        # Check deduplication
        if any(e.ast_hash == ast_hash for e in existing):
            return

        entry = EliteDemonstrationEntry(
            oeis_id=oeis_id,
            canonical_wat=wat_code,
            token_length=len(tokens),
            fuel_consumed=fuel,
            ast_hash=ast_hash,
            discovery_step=step,
            mdl_score=-float(len(tokens)),
        )
        existing.append(entry)
        # Sort by shortest token length (Occam's razor)
        existing.sort(key=lambda x: x.token_length)
        if len(existing) > self.capacity_per_seq:
            self.canonical_archive[oeis_id] = existing[: self.capacity_per_seq]

        # Update backward-compatible entry
        self.entries[oeis_id] = EliteReplayBufferEntry(
            oeis_id=oeis_id,
            terms=terms,
            wat_code=wat_code,
            byte_size=len(wat_code.encode("utf-8")),
            extrapolation_passed=len(terms) > 20,
            mdl_ratio=1.0,
            source="LEGACY_INSPECTION",
        )
        self.last_visited[oeis_id] = step

    def get_entries_for_sequence(self, oeis_id: str) -> List[EliteDemonstrationEntry]:
        """Returns all stored canonical solutions for a sequence ID."""
        return self.canonical_archive.get(oeis_id, [])

    def sample_dormancy_vulnerable_batch(
        self,
        batch_size: int = 2,
        current_step: int = 0,
    ) -> List[Tuple[str, str]]:
        """Samples sequences from EDB prioritized by elapsed dormancy (current_step - last_visited).

        Returns list of (oeis_id, shortest_canonical_wat).
        """
        valid_sids = [
            sid
            for sid, items in self.canonical_archive.items()
            if items and sid in self.training_entries
        ]
        if not valid_sids:
            return []

        if len(valid_sids) <= batch_size:
            return [(sid, self.canonical_archive[sid][0].canonical_wat) for sid in valid_sids]

        dormancies = []
        for sid in valid_sids:
            last_t = self.last_active_visit.get(sid, self.last_visited.get(sid, 0))
            d = max(1, current_step - last_t)
            dormancies.append(d)

        d_arr = np.array(dormancies, dtype=np.float64)
        p_vals = d_arr / np.sum(d_arr)

        chosen_sids = np.random.choice(valid_sids, size=batch_size, replace=False, p=p_vals)
        for sid in chosen_sids:
            self.record_replay_visit(str(sid), current_step)

        return [
            (str(sid), self.canonical_archive[str(sid)][0].canonical_wat) for sid in chosen_sids
        ]

    def get_entry(self, oeis_id: str) -> Optional[EliteReplayBufferEntry]:
        """Retrieves a reference solution for a prompt sequence ID if available."""
        return self.entries.get(oeis_id)

    def has_entry(self, oeis_id: str) -> bool:
        return oeis_id in self.entries

    def sample_demonstration(
        self,
        rng: Optional[random.Random] = None,
    ) -> Optional[EliteReplayBufferEntry]:
        """Samples a random verified demonstration from the buffer to preserve multi-modal diversity."""
        if not self.training_entries:
            return None
        chooser = rng if rng is not None else random
        return chooser.choice([self.entries[sid] for sid in sorted(self.training_entries)])

    def __len__(self) -> int:
        return len(self.entries)

    def _populate_canonical_defaults(self) -> None:
        """Pre-populates verified canonical WebAssembly programs for standard benchmark sequences."""
        # A000217: Triangular Numbers: 0, 1, 3, 6, 10, 15, 21, 28...
        self.add_entry(
            EliteReplayBufferEntry(
                oeis_id="A000217",
                terms=[n * (n + 1) // 2 for n in range(20)],
                wat_code="""(module
  (func (export "compute") (param $n i32) (result i64)
    (local $n64 i64)
    (local.set $n64 (i64.extend_i32_u (local.get $n)))
    (i64.div_u
      (i64.mul (local.get $n64) (i64.add (local.get $n64) (i64.const 1)))
      (i64.const 2)
    )
  )
)""",
                byte_size=58,
                extrapolation_passed=True,
                mdl_ratio=0.85,
                source="CANONICAL_DEFAULT",
            )
        )

        # A000027: Positive Integers: 1, 2, 3, 4, 5... (or 0, 1, 2, 3...)
        self.add_entry(
            EliteReplayBufferEntry(
                oeis_id="A000027",
                terms=[n for n in range(20)],
                wat_code="""(module
  (func (export "compute") (param $n i32) (result i64)
    (i64.extend_i32_u (local.get $n))
  )
)""",
                byte_size=32,
                extrapolation_passed=True,
                mdl_ratio=0.75,
                source="CANONICAL_DEFAULT",
            )
        )

        # A000290: The Squares: 0, 1, 4, 9, 16, 25, 36...
        self.add_entry(
            EliteReplayBufferEntry(
                oeis_id="A000290",
                terms=[n * n for n in range(20)],
                wat_code="""(module
  (func (export "compute") (param $n i32) (result i64)
    (local $n64 i64)
    (local.set $n64 (i64.extend_i32_u (local.get $n)))
    (i64.mul (local.get $n64) (local.get $n64))
  )
)""",
                byte_size=45,
                extrapolation_passed=True,
                mdl_ratio=0.80,
                source="CANONICAL_DEFAULT",
            )
        )

        # A000079: Powers of 2: 1, 2, 4, 8, 16, 32...
        self.add_entry(
            EliteReplayBufferEntry(
                oeis_id="A000079",
                terms=[2**n for n in range(20)],
                wat_code="""(module
  (func (export "compute") (param $n i32) (result i64)
    (local $res i64)
    (local $i i32)
    (local.set $res (i64.const 1))
    (local.set $i (i32.const 0))
    (block $exit
      (loop $loop
        (br_if $exit (i32.ge_s (local.get $i) (local.get $n)))
        (local.set $res (i64.mul (local.get $res) (i64.const 2)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )
    (local.get $res)
  )
)""",
                byte_size=72,
                extrapolation_passed=True,
                mdl_ratio=0.90,
                source="CANONICAL_DEFAULT",
            )
        )

        # A000045: Fibonacci Numbers: 0, 1, 1, 2, 3, 5, 8, 13...
        self.add_entry(
            EliteReplayBufferEntry(
                oeis_id="A000045",
                terms=[
                    0,
                    1,
                    1,
                    2,
                    3,
                    5,
                    8,
                    13,
                    21,
                    34,
                    55,
                    89,
                    144,
                    233,
                    377,
                    610,
                    987,
                    1597,
                    2584,
                    4181,
                ],
                wat_code="""(module
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
        (local.set $temp (i64.add (local.get $a) (local.get $b)))
        (local.set $a (local.get $b))
        (local.set $b (local.get $temp))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )
    (local.get $a)
  )
)""",
                byte_size=88,
                extrapolation_passed=True,
                mdl_ratio=0.92,
                source="CANONICAL_DEFAULT",
            )
        )

        # A000012: The All 1's sequence
        self.add_entry(
            EliteReplayBufferEntry(
                oeis_id="A000012",
                terms=[1] * 20,
                wat_code="""(module
  (func (export "compute") (param $n i32) (result i64)
    (i64.const 1)
  )
)""",
                byte_size=25,
                extrapolation_passed=True,
                mdl_ratio=0.50,
                source="CANONICAL_DEFAULT",
            )
        )

        # A000578: The Cubes: n^3
        self.add_entry(
            EliteReplayBufferEntry(
                oeis_id="A000578",
                terms=[n**3 for n in range(20)],
                wat_code="""(module
  (func (export "compute") (param $n i32) (result i64)
    (local $n64 i64)
    (local.set $n64 (i64.extend_i32_u (local.get $n)))
    (i64.mul (local.get $n64) (i64.mul (local.get $n64) (local.get $n64)))
  )
)""",
                byte_size=48,
                extrapolation_passed=True,
                mdl_ratio=0.82,
                source="CANONICAL_DEFAULT",
            )
        )

        # A005408: The Odd numbers: 2n + 1
        self.add_entry(
            EliteReplayBufferEntry(
                oeis_id="A005408",
                terms=[2 * n + 1 for n in range(20)],
                wat_code="""(module
  (func (export "compute") (param $n i32) (result i64)
    (local $n64 i64)
    (local.set $n64 (i64.extend_i32_u (local.get $n)))
    (i64.add (i64.mul (local.get $n64) (i64.const 2)) (i64.const 1))
  )
)""",
                byte_size=42,
                extrapolation_passed=True,
                mdl_ratio=0.80,
                source="CANONICAL_DEFAULT",
            )
        )

        # A000032: Lucas numbers: 2, 1, 3, 4, 7, 11, 18, 29...
        self.add_entry(
            EliteReplayBufferEntry(
                oeis_id="A000032",
                terms=[
                    2,
                    1,
                    3,
                    4,
                    7,
                    11,
                    18,
                    29,
                    47,
                    76,
                    123,
                    199,
                    322,
                    521,
                    843,
                    1364,
                    2207,
                    3571,
                    5778,
                    9349,
                ],
                wat_code="""(module
  (func (export "compute") (param $n i32) (result i64)
    (local $a i64)
    (local $b i64)
    (local $temp i64)
    (local $i i32)
    (local.set $a (i64.const 2))
    (local.set $b (i64.const 1))
    (local.set $i (i32.const 0))
    (block $exit
      (loop $loop
        (br_if $exit (i32.ge_s (local.get $i) (local.get $n)))
        (local.set $temp (i64.add (local.get $a) (local.get $b)))
        (local.set $a (local.get $b))
        (local.set $b (local.get $temp))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )
    (local.get $a)
  )
)""",
                byte_size=88,
                extrapolation_passed=True,
                mdl_ratio=0.92,
                source="CANONICAL_DEFAULT",
            )
        )

        # A000129: Pell numbers: 0, 1, 2, 5, 12, 29, 70...
        self.add_entry(
            EliteReplayBufferEntry(
                oeis_id="A000129",
                terms=[
                    0,
                    1,
                    2,
                    5,
                    12,
                    29,
                    70,
                    169,
                    408,
                    985,
                    2378,
                    5741,
                    13860,
                    33461,
                    80782,
                    195025,
                    470832,
                    1136689,
                    2744210,
                    6625109,
                ],
                wat_code="""(module
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
        (local.set $temp (i64.add (i64.mul (local.get $b) (i64.const 2)) (local.get $a)))
        (local.set $a (local.get $b))
        (local.set $b (local.get $temp))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )
    (local.get $a)
  )
)""",
                byte_size=92,
                extrapolation_passed=True,
                mdl_ratio=0.93,
                source="CANONICAL_DEFAULT",
            )
        )
        self.seed_canonical_multilimb_canaries()

    def seed_canonical_multilimb_canaries(self) -> None:
        """Seeds canonical 4-limb programs for the 6 landmark canaries."""
        canary_programs = {
            "A000217": """(module
  (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
    (local $n64 i64)
    local.get $n i64.extend_i32_u local.set $n64
    local.get $n64 local.get $n64 i64.const 1 i64.add i64.mul i64.const 2 i64.div_u
    i64.const 0 i64.const 0 i64.const 0
  )
)""",
            "A000290": """(module
  (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
    (local $n64 i64)
    local.get $n i64.extend_i32_u local.set $n64
    local.get $n64 local.get $n64 i64.mul
    i64.const 0 i64.const 0 i64.const 0
  )
)""",
            "A000079": """(module
  (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
    (local $a0 i64) (local $a1 i64) (local $a2 i64) (local $a3 i64)
    (local $s0 i64) (local $s1 i64) (local $s2 i64) (local $s3 i64)
    (local $i i32)
    i64.const 1 local.set $a0 i64.const 0 local.set $a1 i64.const 0 local.set $a2 i64.const 0 local.set $a3
    (block $exit
      (loop $loop
        local.get $i local.get $n i32.ge_s br_if $exit
        local.get $a0 i64.const 1 i64.shl local.set $s0
        local.get $a1 i64.const 1 i64.shl local.get $a0 i64.const 63 i64.shr_u i64.or local.set $s1
        local.get $a2 i64.const 1 i64.shl local.get $a1 i64.const 63 i64.shr_u i64.or local.set $s2
        local.get $a3 i64.const 1 i64.shl local.get $a2 i64.const 63 i64.shr_u i64.or local.set $s3
        local.get $s0 local.set $a0 local.get $s1 local.set $a1
        local.get $s2 local.set $a2 local.get $s3 local.set $a3
        local.get $i i32.const 1 i32.add local.set $i
        br $loop
      )
    )
    local.get $a0 local.get $a1 local.get $a2 local.get $a3
  )
)""",
            "A000045": """(module
  (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
    (local $a0 i64) (local $a1 i64) (local $a2 i64) (local $a3 i64)
    (local $b0 i64) (local $b1 i64) (local $b2 i64) (local $b3 i64)
    (local $t0 i64) (local $t1 i64) (local $t2 i64) (local $t3 i64)
    (local $i i32)
    i256.zero local.set $a3 local.set $a2 local.set $a1 local.set $a0
    i256.const 1 local.set $b3 local.set $b2 local.set $b1 local.set $b0
    (block $exit
      (loop $loop
        local.get $i local.get $n i32.ge_s br_if $exit
        local.get $b0 local.get $b1 local.get $b2 local.get $b3
        local.get $a0 local.get $a1 local.get $a2 local.get $a3
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
)""",
            "A000032": """(module
  (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
    (local $a0 i64) (local $a1 i64) (local $a2 i64) (local $a3 i64)
    (local $b0 i64) (local $b1 i64) (local $b2 i64) (local $b3 i64)
    (local $t0 i64) (local $t1 i64) (local $t2 i64) (local $t3 i64)
    (local $i i32)
    i256.const 2 local.set $a3 local.set $a2 local.set $a1 local.set $a0
    i256.const 1 local.set $b3 local.set $b2 local.set $b1 local.set $b0
    (block $exit
      (loop $loop
        local.get $i local.get $n i32.ge_s br_if $exit
        local.get $b0 local.get $b1 local.get $b2 local.get $b3
        local.get $a0 local.get $a1 local.get $a2 local.get $a3
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
)""",
            "A000129": """(module
  (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
    (local $a0 i64) (local $a1 i64) (local $a2 i64) (local $a3 i64)
    (local $b0 i64) (local $b1 i64) (local $b2 i64) (local $b3 i64)
    (local $t0 i64) (local $t1 i64) (local $t2 i64) (local $t3 i64)
    (local $i i32)
    i256.zero local.set $a3 local.set $a2 local.set $a1 local.set $a0
    i256.const 1 local.set $b3 local.set $b2 local.set $b1 local.set $b0
    (block $exit
      (loop $loop
        local.get $i local.get $n i32.ge_s br_if $exit
        local.get $b0 local.get $b1 local.get $b2 local.get $b3
        local.get $b0 local.get $b1 local.get $b2 local.get $b3
        i256.add
        local.get $a0 local.get $a1 local.get $a2 local.get $a3
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
)""",
        }
        for sid, wat in canary_programs.items():
            self.add_canonical_entry(
                oeis_id=sid,
                wat_code=wat,
                terms=[],
                fuel=10,
                step=0,
            )

    def inject_into_prompt_group(self, oeis_id, candidates, rewards):
        """Compatibility API: imported teachers never become on-policy trajectories."""
        return list(candidates), list(rewards), False

    def admit_from_archive(self, archive, key, oeis_id):
        value = archive.get(key)
        if not value["entry"]["training_eligible"]:
            raise ValueError("ineligible archive entry cannot reach legacy training view")
        # The strict complete codec is different from legacy module tokens.
        # This view is retained for inspection; qualified replay reads the same
        # immutable archive directly with its declared codec.
        self.entries[oeis_id] = EliteReplayBufferEntry(
            oeis_id,
            [int(v) for v in value["entry"]["conditioning_values"]],
            value["source"],
            len(value["source"].encode()),
            bool(value["entry"]["validation_indices"]),
            1.0,
            value["entry"]["origin"],
        )
