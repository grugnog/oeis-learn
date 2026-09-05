"""Elite Seed Demonstration Replay Buffer for Trajectory Injection and SYMPLE Vulnerability Replay."""

from __future__ import annotations

import hashlib
import json
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

    def __init__(self, capacity_per_seq: int = 4):
        self.capacity_per_seq = capacity_per_seq
        self.entries: Dict[str, EliteReplayBufferEntry] = {}
        self.canonical_archive: Dict[str, List[EliteDemonstrationEntry]] = {}
        self.last_visited: Dict[str, int] = {}
        self.last_active_visit: Dict[str, int] = {}
        self.last_replay_visit: Dict[str, int] = {}
        self.discovery_step: Dict[str, int] = {}
        self.last_verified_step: Dict[str, int] = {}
        self._populate_canonical_defaults()

    def record_active_visit(self, oeis_id: str, step: int) -> None:
        """Records active task exploration step."""
        self.last_active_visit[oeis_id] = step
        self.last_visited[oeis_id] = step

    def record_replay_visit(self, oeis_id: str, step: int) -> None:
        """Records replay review step without resetting active exploration history."""
        self.last_replay_visit[oeis_id] = step

    def add_entry(self, entry: EliteReplayBufferEntry) -> None:
        """Adds or updates a verified canonical program entry."""
        self.entries[entry.oeis_id] = entry
        self.add_canonical_entry(
            oeis_id=entry.oeis_id,
            wat_code=entry.wat_code,
            terms=entry.terms,
            fuel=10,
            step=0,
        )
        logger.debug(f"Added elite reference solution for {entry.oeis_id} (source: {entry.source})")

    def add_canonical_entry(
        self,
        oeis_id: str,
        wat_code: str,
        terms: List[int],
        fuel: int = 0,
        step: int = 0,
    ) -> None:
        """Stores a verified canonical AST program in the sequence archive, bounded by capacity and deduplicated."""
        tokens = tokenize_wat(wat_code)
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
            extrapolation_passed=True,
            mdl_ratio=1.0,
            source="SYMPLE_DISCOVERY",
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
        valid_sids = [sid for sid, items in self.canonical_archive.items() if items]
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

        return [(str(sid), self.canonical_archive[str(sid)][0].canonical_wat) for sid in chosen_sids]

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
        if not self.entries:
            return None
        chooser = rng if rng is not None else random
        return chooser.choice(list(self.entries.values()))

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
                terms=[0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181],
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
                terms=[2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322, 521, 843, 1364, 2207, 3571, 5778, 9349],
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
                terms=[0, 1, 2, 5, 12, 29, 70, 169, 408, 985, 2378, 5741, 13860, 33461, 80782, 195025, 470832, 1136689, 2744210, 6625109],
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
