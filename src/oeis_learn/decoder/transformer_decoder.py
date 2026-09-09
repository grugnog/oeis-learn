"""Alias module exposing WatTransformerDecoder as TransformerDecoder."""

from __future__ import annotations

from oeis_learn.decoder.wat_decoder import PositionalEncodingDecoder, WatTransformerDecoder

TransformerDecoder = WatTransformerDecoder
__all__ = ["WatTransformerDecoder", "TransformerDecoder", "PositionalEncodingDecoder"]
