"""Frozen bigint-codec/v1 vocabulary; BOS is outside the 1024-token body cap."""

import re
from .bigint_grammar import BodyState, CodecError, EnvironmentTracker as EnvironmentTracker
from oeis_learn.synthesis.contracts import identity

VOCABULARY = (
    "<pad>",
    "<bos>",
    "<eos>",
    "(",
    ")",
    "<int>",
    "</int>",
    "-",
    *tuple("0123456789"),
    "program",
    "local",
    "bigint",
    "control",
    "bool",
    "int",
    "param.get",
    "get",
    "add",
    "sub",
    "mul",
    "eq",
    "lt",
    "le",
    "to_control",
    "to_bigint",
    "increment",
    "control.lt",
    "true",
    "false",
    "select",
    "set",
    "if",
    "then",
    "else",
    "loop",
    "return",
    "$index",
    *(f"$b{n}" for n in range(16)),
    *(f"$c{n}" for n in range(8)),
    *(f"$q{n}" for n in range(8)),
    *(f"p{n}" for n in range(8)),
)
FOUNDATION_VOCAB_SIZE = len(VOCABULARY)  # generic sampler compatibility interface
TOKEN_TO_ID = {s: i for i, s in enumerate(VOCABULARY)}
ID_TO_TOKEN = dict(enumerate(VOCABULARY))
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2
INT_ID, TERM_ID, NEG_ID = (TOKEN_TO_ID[s] for s in ("<int>", "</int>", "-"))
DIGIT_IDS = frozenset(TOKEN_TO_ID[s] for s in "0123456789")
MAX_BODY_TOKENS = 1024
MAX_SOURCE_BYTES = 65536


def codec_identity():
    return dict(
        id="bigint-codec/v1",
        vocabulary=list(VOCABULARY),
        bos_id=BOS_ID,
        pad_id=PAD_ID,
        eos_id=EOS_ID,
        max_body_tokens=MAX_BODY_TOKENS,
        max_source_bytes=MAX_SOURCE_BYTES,
        parameters="first-occurrence/v1",
        language="bigint-body/v1",
    )


def codec_digest():
    return identity(codec_identity())


def encode_body(source):
    if type(source) is not str or not source.isascii() or len(source.encode()) > MAX_SOURCE_BYTES:
        raise CodecError("source limit")
    parts = re.findall(r"\(|\)|[^\s()]+", source, flags=re.ASCII)
    parameters = {}
    ids = []
    for word in parts:
        if re.fullmatch(r"p[0-7]", word):
            if word not in parameters:
                parameters[word] = f"p{len(parameters)}"
            word = parameters[word]
        if re.fullmatch(r"0|-?[1-9][0-9]*", word, flags=re.ASCII):
            ids.extend([INT_ID, *(TOKEN_TO_ID[c] for c in word), TERM_ID])
        elif word in TOKEN_TO_ID and not word.startswith("<"):
            ids.append(TOKEN_TO_ID[word])
        else:
            raise CodecError(f"unknown/noncanonical source atom {word!r}")
        if len(ids) >= MAX_BODY_TOKENS:
            raise CodecError("token cap")
    ids.append(EOS_ID)
    validate_body(ids)
    return ids


def validate_body(ids):
    if not ids or len(ids) > MAX_BODY_TOKENS:
        raise CodecError("token cap")
    # Complete targets need one grammar parse. Prefix replay remains necessary
    # for generation, but would make corpus validation quadratic in token count.
    from .bigint_grammar import Reader, Need
    if any(type(token) is not int or token not in ID_TO_TOKEN for token in ids):
        raise CodecError("invalid token")
    try:
        Reader([ID_TO_TOKEN[token] for token in ids]).program()
    except Need as exc:
        raise CodecError("incomplete source or missing EOS") from exc



def decode_body(ids):
    validate_body(ids)
    words = []
    digits = []
    for index in ids[:-1]:
        token = ID_TO_TOKEN[index]
        if index == INT_ID:
            digits = []
        elif index == TERM_ID:
            words.append("".join(digits))
        elif index in DIGIT_IDS or index == NEG_ID:
            digits.append(token)
        else:
            words.append(token)
    return " ".join(words)


def allowed_tokens_at(prefix):
    state = BodyState()
    for token in prefix:
        state.consume(token)
    return state.allowed()


def fitting_policy(source):
    """Classify parsed codec atoms, independent of source whitespace."""
    return "none" if TOKEN_TO_ID["param.get"] in encode_body(source) else "eligible-literals/v1"
