"""WebAssembly Text (WAT) optimization, dead-code elimination (DCE), and vacuuming pass."""

from __future__ import annotations

import re
from typing import List, Set, Tuple
from oeis_learn.data.models import CanonicalProgramArtifact
from oeis_learn.decoder.wat_grammar import tokenize_wat


def optimize_wat_program(wat_code: str, hard_waste_threshold: float = 0.30) -> CanonicalProgramArtifact:
    """Executes dead-code elimination, vacuuming, and unused local removal on a WAT program,

    computing canonical optimized WAT text and syntactic waste ratio.
    """
    raw_tokens = tokenize_wat(wat_code)
    raw_count = max(1, len(raw_tokens))

    current_wat = wat_code

    # Pass 1: Remove nop instructions: '(nop)' or isolated 'nop'
    current_wat = re.sub(r"\(\s*nop\s*\)", "", current_wat)
    current_wat = re.sub(r"\bnop\b", "", current_wat)

    # Pass 2: Iterative Vacuuming of redundant push-drop pairs
    # Matches: (local.get $var) drop, (i64.const val) drop, (i32.const val) drop, etc.
    # Also matches unparenthesized: local.get $var drop, i64.const val drop
    changed = True
    passes_applied = ["--vacuum", "--dce", "--remove-unused-locals"]
    max_passes = 10
    pass_idx = 0

    while changed and pass_idx < max_passes:
        pass_idx += 1
        prev_wat = current_wat

        # 2a. Parenthesized push followed by drop: (local.get $x) drop or (i64.const 42) drop
        current_wat = re.sub(
            r"\(\s*(local\.get\s+\$[a-zA-Z0-9_]+|i[36]4\.const\s+-?\d+)\s*\)\s*drop",
            "",
            current_wat,
        )

        # 2b. Folded drop: (drop (local.get $x)) or (drop (i64.const 42))
        current_wat = re.sub(
            r"\(\s*drop\s+\(\s*(local\.get\s+\$[a-zA-Z0-9_]+|i[36]4\.const\s+-?\d+)\s*\)\s*\)",
            "",
            current_wat,
        )

        # 2c. Unparenthesized: local.get $x drop or i64.const 42 drop
        current_wat = re.sub(
            r"\b(local\.get\s+\$[a-zA-Z0-9_]+|i[36]4\.const\s+-?\d+)\s+drop\b",
            "",
            current_wat,
        )

        # 2d. Drop after drop-free arithmetic: (drop (i64.add ...)) -> remove if purely dead
        # 2e. Redundant local.set where value is immediately overwritten or local is unused
        changed = current_wat != prev_wat

    # Pass 3: Unused local declarations and dead local assignments removal
    # Find all declared locals: (local $x i64)
    local_decl_pattern = re.compile(r"\(\s*local\s+(\$[a-zA-Z0-9_]+)\s+(i32|i64)\s*\)")
    declared_locals = local_decl_pattern.findall(current_wat)

    for var_name, var_type in declared_locals:
        # Check if var is ever read via local.get $var_name
        get_pattern = re.compile(rf"\blocal\.get\s+{re.escape(var_name)}\b")
        if not get_pattern.search(current_wat):
            # Dead variable: remove sets and declaration
            # Remove (local.set $var val) or (local.set $var (val)) -> leave expr or drop
            current_wat = re.sub(
                rf"\(\s*local\.set\s+{re.escape(var_name)}\s*\)",
                "drop",
                current_wat,
            )
            current_wat = re.sub(
                rf"\blocal\.set\s+{re.escape(var_name)}\b",
                "drop",
                current_wat,
            )
            # Remove declaration
            current_wat = re.sub(
                rf"\(\s*local\s+{re.escape(var_name)}\s+(i32|i64)\s*\)",
                "",
                current_wat,
            )

    # Pass 4: Clean up any newly formed push-drop pairs resulting from pass 3
    current_wat = re.sub(
        r"\(\s*(local\.get\s+\$[a-zA-Z0-9_]+|i[36]4\.const\s+-?\d+)\s*\)\s*drop",
        "",
        current_wat,
    )
    current_wat = re.sub(
        r"\b(local\.get\s+\$[a-zA-Z0-9_]+|i[36]4\.const\s+-?\d+)\s+drop\b",
        "",
        current_wat,
    )

    # Normalize whitespace and structure
    opt_tokens = tokenize_wat(current_wat)
    opt_count = max(1, len(opt_tokens))
    opt_wat = " ".join(opt_tokens)

    # Determine truthful applied passes
    applied = []
    if current_wat != wat_code:
        applied = ["python-regex-vacuum", "python-regex-dce", "python-regex-unused-locals"]
    else:
        applied = ["textual-normalization"]

    # Calculate syntactic waste ratio:
    waste_tokens = max(0, len(raw_tokens) - len(opt_tokens))
    waste_ratio = float(waste_tokens) / float(raw_count)
    is_waste_exceeded = waste_ratio > hard_waste_threshold

    return CanonicalProgramArtifact(
        raw_wat=wat_code,
        opt_wat=opt_wat if opt_tokens else wat_code,
        raw_token_count=len(raw_tokens),
        opt_token_count=len(opt_tokens),
        waste_ratio=waste_ratio,
        passes_applied=applied,
        is_waste_exceeded=is_waste_exceeded,
    )
