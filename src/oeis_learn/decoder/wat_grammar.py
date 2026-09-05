"""WebAssembly Text (WAT) grammar definitions and token vocabulary."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

# Supported WAT result profiles
RESULT_PROFILES = ("i64_scalar_v1", "i256x4_v1")

# WAT Special & Reserved Tokens
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

SYNTAX_TOKENS = [
    "(", ")", "module", "func", "export", '"compute"', '"generate_term"',
    "param", "result", "local",
    "i32", "i64",
]

INSTRUCTION_TOKENS = [
    "local.get", "local.set", "local.tee",
    "i32.const", "i64.const", "i64.const_?",
    "i64.add", "i64.sub", "i64.mul", "i64.div_s", "i64.div_u", "i64.rem_s", "i64.rem_u",
    "i64.and", "i64.or", "i64.xor", "i64.shl", "i64.shr_s", "i64.shr_u",
    "i64.eqz", "i64.eq", "i64.ne", "i64.lt_s", "i64.gt_s", "i64.le_s", "i64.ge_s",
    "i32.add", "i32.sub", "i32.mul", "i32.div_s", "i32.div_u", "i32.rem_s", "i32.rem_u",
    "i32.and", "i32.or", "i32.xor", "i32.shl", "i32.shr_s", "i32.shr_u",
    "i32.ge_s", "i32.lt_s", "i32.gt_s", "i32.le_s", "i32.eq", "i32.ne", "i32.eqz",
    "i32.wrap_i64", "i64.extend_i32_s", "i64.extend_i32_u",
    "drop", "nop", "unreachable", "return", "br", "br_if",
    "block", "loop", "if", "then", "else",
]

IDENTIFIER_TOKENS = [
    "$n", "$a", "$b", "$c", "$d", "$i", "$j", "$k", "$temp", "$val", "$res", "$n64",
    "$l", "$exit", "$loop", "$block",
]

LITERAL_TOKENS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "12", "16", "32", "42", "64", "100", "-1", "-2",
]

WAT_VOCABULARY = SPECIAL_TOKENS + SYNTAX_TOKENS + INSTRUCTION_TOKENS + IDENTIFIER_TOKENS + LITERAL_TOKENS

# Opcode stack signature mappings: opcode -> (tuple of input types, tuple of output types)
OPCODE_SIGNATURES: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    # Binary i64 arithmetic
    "i64.add": (("i64", "i64"), ("i64",)),
    "i64.sub": (("i64", "i64"), ("i64",)),
    "i64.mul": (("i64", "i64"), ("i64",)),
    "i64.div_s": (("i64", "i64"), ("i64",)),
    "i64.div_u": (("i64", "i64"), ("i64",)),
    "i64.rem_s": (("i64", "i64"), ("i64",)),
    "i64.rem_u": (("i64", "i64"), ("i64",)),
    "i64.and": (("i64", "i64"), ("i64",)),
    "i64.or": (("i64", "i64"), ("i64",)),
    "i64.xor": (("i64", "i64"), ("i64",)),
    "i64.shl": (("i64", "i64"), ("i64",)),
    "i64.shr_s": (("i64", "i64"), ("i64",)),
    "i64.shr_u": (("i64", "i64"), ("i64",)),
    # Binary i64 comparisons -> i32
    "i64.eq": (("i64", "i64"), ("i32",)),
    "i64.ne": (("i64", "i64"), ("i32",)),
    "i64.lt_s": (("i64", "i64"), ("i32",)),
    "i64.gt_s": (("i64", "i64"), ("i32",)),
    "i64.le_s": (("i64", "i64"), ("i32",)),
    "i64.ge_s": (("i64", "i64"), ("i32",)),
    "i64.eqz": (("i64",), ("i32",)),
    # Binary i32 arithmetic
    "i32.add": (("i32", "i32"), ("i32",)),
    "i32.sub": (("i32", "i32"), ("i32",)),
    "i32.mul": (("i32", "i32"), ("i32",)),
    "i32.div_s": (("i32", "i32"), ("i32",)),
    "i32.div_u": (("i32", "i32"), ("i32",)),
    "i32.rem_s": (("i32", "i32"), ("i32",)),
    "i32.rem_u": (("i32", "i32"), ("i32",)),
    "i32.and": (("i32", "i32"), ("i32",)),
    "i32.or": (("i32", "i32"), ("i32",)),
    "i32.xor": (("i32", "i32"), ("i32",)),
    "i32.shl": (("i32", "i32"), ("i32",)),
    "i32.shr_s": (("i32", "i32"), ("i32",)),
    "i32.shr_u": (("i32", "i32"), ("i32",)),
    # Binary i32 comparisons
    "i32.eq": (("i32", "i32"), ("i32",)),
    "i32.ne": (("i32", "i32"), ("i32",)),
    "i32.lt_s": (("i32", "i32"), ("i32",)),
    "i32.gt_s": (("i32", "i32"), ("i32",)),
    "i32.le_s": (("i32", "i32"), ("i32",)),
    "i32.ge_s": (("i32", "i32"), ("i32",)),
    "i32.eqz": (("i32",), ("i32",)),
    # Conversions
    "i32.wrap_i64": (("i64",), ("i32",)),
    "i64.extend_i32_s": (("i32",), ("i64",)),
    "i64.extend_i32_u": (("i32",), ("i64",)),
    # Constants
    "i32.const": ((), ("i32",)),
    "i64.const": ((), ("i64",)),
    "i64.const_?": ((), ("i64",)),
    # Control / stack
    "drop": (("any",), ()),
    "br_if": (("i32",), ()),
    "nop": ((), ()),
    "unreachable": ((), ()),
    "return": ((), ()),
}

# Pre-indexed signature groups
SIG_I64_I64_TO_I64 = [op for op, (inp, out) in OPCODE_SIGNATURES.items() if inp == ("i64", "i64") and out == ("i64",)]
SIG_I32_I32_TO_I32 = [op for op, (inp, out) in OPCODE_SIGNATURES.items() if inp == ("i32", "i32") and out == ("i32",)]
SIG_I64_TO_I32 = [op for op, (inp, out) in OPCODE_SIGNATURES.items() if inp == ("i64",) and out == ("i32",)]
SIG_I32_TO_I64 = [op for op, (inp, out) in OPCODE_SIGNATURES.items() if inp == ("i32",) and out == ("i64",)]

TOKEN_TO_ID: Dict[str, int] = {tok: idx for idx, tok in enumerate(WAT_VOCABULARY)}
ID_TO_TOKEN: Dict[int, str] = {idx: tok for idx, tok in enumerate(WAT_VOCABULARY)}

PAD_ID = TOKEN_TO_ID["<pad>"]
BOS_ID = TOKEN_TO_ID["<bos>"]
EOS_ID = TOKEN_TO_ID["<eos>"]
UNK_ID = TOKEN_TO_ID["<unk>"]
VOCAB_SIZE = len(WAT_VOCABULARY)


def tokenize_wat(wat_str: str) -> List[str]:
    """Tokenize WAT code string into token list."""
    tokens = []
    i = 0
    n = len(wat_str)
    while i < n:
        c = wat_str[i]
        if c.isspace():
            i += 1
            continue
        if c in "()":
            tokens.append(c)
            i += 1
            continue
        if c == '"':
            # string literal
            j = i + 1
            while j < n and wat_str[j] != '"':
                j += 1
            tokens.append(wat_str[i : j + 1])
            i = j + 1
            continue
        # regular token
        j = i
        while j < n and not wat_str[j].isspace() and wat_str[j] not in "()":
            j += 1
        tokens.append(wat_str[i:j])
        i = j
    return tokens


def encode_wat(wat_str: str) -> List[int]:
    """Encode WAT string to list of token IDs."""
    tokens = tokenize_wat(wat_str)
    return [TOKEN_TO_ID.get(tok, UNK_ID) for tok in tokens]


def decode_wat_tokens(token_ids: List[int]) -> str:
    """Decode token IDs back to a formatted WAT string."""
    tokens = [ID_TO_TOKEN.get(t_id, "<unk>") for t_id in token_ids if t_id not in (PAD_ID, BOS_ID, EOS_ID)]
    return " ".join(tokens)
