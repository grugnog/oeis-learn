"""Foundation `wat_body_decimal_v1` program codec (T007).

The codec maps a canonical WAT body to a fixed token vocabulary using fixed
opcode/local tokens and signed digit immediates with an explicit terminator
``</int>``. It enforces the foundation body constraints:

- body tokens non-empty, EOS exactly once at the end
- no BOS / padding / unknown tokens
- at most 1,024 body tokens
- canonical source at most 64 KiB

The vocabulary exceeds 128 tokens so a fixed-width (128-wide) logit mask
cannot silently drop vocabulary members.

The legacy codec identities (``i64_scalar_v1``, ``i256x4_v1``) are preserved
by ``decoder/wat_grammar.py`` for diagnostics; this module is the distinct
``wat_body_decimal_v1`` codec and never reuses incompatible output weights.
"""

from __future__ import annotations

from typing import List, Set

from oeis_learn.experiments.profiles import (
    ALLOWED_OPERATORS,
    I32_LOCAL_NAMES,
    I64_LOCAL_NAMES,
    INPUT_LOCAL,
)

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

_SYNTAX_TOKENS = [
    "(", ")", "local", "result", "param",
    "block", "loop", "if", "then", "else", "end",
    "br", "br_if", "return", "drop", "unreachable",
    "func", "module", "export", "i32", "i64",
    '"compute"', '"generate_term"',
]

_DIGIT_TOKENS = [str(d) for d in range(10)]
_NEG = "-"
_TERM = "</int>"
_EOS = "<eos>"

_FIXED_LOCALS = [INPUT_LOCAL] + list(I64_LOCAL_NAMES) + list(I32_LOCAL_NAMES)

# Fixed opcode/local tokens plus signed digit immediates with explicit
# terminator. Ordering keeps opcodes last so their token ids exceed 128,
# making any fixed-width (128-wide) mask visibly lossy.
FOUNDATION_VOCABULARY: List[str] = []
for tok in (
    [_EOS]
    + _SYNTAX_TOKENS
    + _DIGIT_TOKENS
    + [_NEG, _TERM]
    + _FIXED_LOCALS
    + list(ALLOWED_OPERATORS)
):
    if tok not in FOUNDATION_VOCABULARY:
        FOUNDATION_VOCABULARY.append(tok)

FOUNDATION_VOCAB_SIZE = len(FOUNDATION_VOCABULARY)

TOKEN_TO_ID: dict = {tok: idx for idx, tok in enumerate(FOUNDATION_VOCABULARY)}
ID_TO_TOKEN: dict = {idx: tok for idx, tok in enumerate(FOUNDATION_VOCABULARY)}

EOS_ID = TOKEN_TO_ID[_EOS]
TERM_ID = TOKEN_TO_ID[_TERM]
NEG_ID = TOKEN_TO_ID[_NEG]
DIGIT_IDS: Set[int] = {TOKEN_TO_ID[d] for d in _DIGIT_TOKENS}

# No <unk>/<pad>/<bos> tokens exist in this codec; unknown ids are forbidden.
MAX_BODY_TOKENS = 1024  # including EOS, excluding BOS and wrapper
MAX_SOURCE_BYTES = 65536  # 64 KiB canonical source


class CodecError(ValueError):
    """Raised when a body token sequence or source violates codec constraints."""


# ---------------------------------------------------------------------------
# Canonical tokenizer
# ---------------------------------------------------------------------------


def _tokenize_source(source: str) -> List[str]:
    """Split a canonical source into tokens; numeric literals split into digits."""
    tokens: List[str] = []
    i, n = 0, len(source)
    while i < n:
        c = source[i]
        if c.isspace():
            i += 1
            continue
        if c in "()":
            tokens.append(c)
            i += 1
            continue
        if c == '"':
            j = i + 1
            while j < n and source[j] != '"':
                j += 1
            tokens.append(source[i : j + 1])
            i = j + 1
            continue
        j = i
        while j < n and not source[j].isspace() and source[j] not in "()":
            j += 1
        word = source[i:j]
        i = j
        if word == _TERM:
            tokens.append(word)
            continue
        if word.startswith("-") and word[1:].isdigit():
            tokens.append(_NEG)
            word = word[1:]
            tokens.extend(list(word))
            continue
        if word.isdigit():
            tokens.extend(list(word))
            continue
        tokens.append(word)
    return tokens


# ---------------------------------------------------------------------------
# Public codec API
# ---------------------------------------------------------------------------


def encode_body(source: str) -> List[int]:
    """Encode a canonical WAT body to token ids, appending EOS.

    Raises CodecError on unknown tokens, source over 64 KiB, or more than
    1,024 body tokens (no truncation is ever performed).
    """
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise CodecError(
            f"canonical source {len(source.encode('utf-8'))} bytes exceeds {MAX_SOURCE_BYTES}"
        )
    str_tokens = _tokenize_source(source)
    ids: List[int] = []
    for tok in str_tokens:
        if tok not in TOKEN_TO_ID:
            raise CodecError(f"unknown body token {tok!r} (no UNK token in this codec)")
        ids.append(TOKEN_TO_ID[tok])
    ids.append(EOS_ID)
    if len(ids) > MAX_BODY_TOKENS:
        raise CodecError(f"body has {len(ids)} tokens (incl. EOS), exceeds {MAX_BODY_TOKENS}")
    return ids


def decode_body(token_ids: List[int]) -> str:
    """Decode body token ids back to the exact canonical source (round trip).

    Digit runs terminated by ``</int>`` are rejoined into signed integer
    literals; the terminator is retained in the canonical serialization.
    Raises CodecError on an unterminated digit run (a partially emitted
    literal) or a mid-sequence EOS.
    """
    parts: List[str] = []
    digits: List[str] = []
    for idx, t_id in enumerate(token_ids):
        if t_id not in ID_TO_TOKEN:
            raise CodecError(f"unknown token id {t_id}")
        tok = ID_TO_TOKEN[t_id]
        if tok == _EOS:
            if idx != len(token_ids) - 1:
                raise CodecError("EOS must appear exactly once, at the end")
            break
        if tok == _TERM:
            if not digits:
                raise CodecError(f"unterminated/empty integer immediate at {tok!r}")
            num = "".join(digits)
            parts.append(num)
            parts.append(tok)
            digits = []
        elif t_id in DIGIT_IDS or t_id == NEG_ID:
            digits.append(tok)
        else:
            if digits:
                raise CodecError("partially emitted literal: digits without terminator")
            parts.append(tok)
    if digits:
        raise CodecError("partially emitted literal: unterminated digit run")
    return " ".join(parts)


def validate_body(token_ids: List[int]) -> None:
    """Enforce body token constraints: non-empty, EOS exactly once at the end,
    no BOS/padding/unknown ids, at most 1,024 tokens."""
    if not token_ids:
        raise CodecError("body tokens must be non-empty")
    if token_ids[-1] != EOS_ID:
        raise CodecError("EOS must be exactly once, at the end")
    if token_ids.count(EOS_ID) != 1:
        raise CodecError("EOS must appear exactly once")
    if len(token_ids) > MAX_BODY_TOKENS:
        raise CodecError(f"body has {len(token_ids)} tokens, exceeds {MAX_BODY_TOKENS}")
    for t_id in token_ids:
        if t_id not in ID_TO_TOKEN or t_id == EOS_ID:
            raise CodecError(f"invalid body token id {t_id}")
        tok = ID_TO_TOKEN[t_id]
        if tok in ("<unk>", "<pad>", "<bos>"):
            raise CodecError(f"forbidden special token {tok!r} in body")


# ---------------------------------------------------------------------------
# Incremental type/scope mask
# ---------------------------------------------------------------------------

_OPCODE_IDS = {TOKEN_TO_ID[op] for op in ALLOWED_OPERATORS}
_LOCAL_IDS = {TOKEN_TO_ID[l] for l in _FIXED_LOCALS}
_OPEN_PAREN = TOKEN_TO_ID["("]
_CLOSE_PAREN = TOKEN_TO_ID[")"]
_LOCAL_KW = TOKEN_TO_ID["local"]

_IMMEDIATE_OPS = {
    op: TOKEN_TO_ID[op] for op in ("i64.const", "i32.const", "i256.const", "br", "br_if")
}


def allowed_tokens_at(prefix: List[int]) -> List[int]:
    """Return the token ids allowed next for the given body token prefix.

    Implements an incremental type/scope mask: immediates must be spelled as
    signed digit tokens terminated by ``</int>``; EOS is only allowed once the
    body is complete at top level.
    """
    allowed: Set[int] = set()

    if not prefix:
        # Body starts with '(' (a local declaration) or a top-level opcode.
        allowed.add(_OPEN_PAREN)
        allowed |= _OPCODE_IDS
        return sorted(allowed)

    if prefix[-1] == EOS_ID:
        return [EOS_ID]

    # Immediate in progress: after an opcode that takes an immediate, or while
    # accumulating signed digits, only digits and the terminator are allowed.
    if prefix[-1] in DIGIT_IDS or prefix[-1] == NEG_ID:
        allowed |= DIGIT_IDS
        allowed.add(TERM_ID)
        allowed.add(NEG_ID)
        return sorted(allowed)

    # Inside a numeric immediate: the terminator is required before continuing.
    if prefix[-1] == _TERM:
        pass  # fall through to general body continuation

    last_op = prefix[-1]
    if last_op in _IMMEDIATE_OPS.values():
        allowed |= DIGIT_IDS
        allowed.add(NEG_ID)
        return sorted(allowed)

    # General body context: opcodes, fixed locals, parens, and EOS at top level.
    allowed |= _OPCODE_IDS
    allowed |= _LOCAL_IDS
    allowed.add(_OPEN_PAREN)
    allowed.add(_CLOSE_PAREN)
    allowed.add(_LOCAL_KW)

    # EOS is valid only when the body is complete: no open parens and the last
    # token was a terminator or a closing paren / opcode at top level.
    open_parens = 0
    for t_id in prefix:
        tok = ID_TO_TOKEN[t_id]
        if tok == "(":
            open_parens += 1
        elif tok == ")":
            open_parens = max(0, open_parens - 1)
    if open_parens == 0 and len(prefix) >= 1:
        allowed.add(EOS_ID)

    return sorted(allowed)
