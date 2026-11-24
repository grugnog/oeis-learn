"""Lossless transport of original comparison evidence within the public ZIP budget."""
import base64
import hashlib
import zlib

from .contracts import load, seal, verify_refs
from .report import MAX_BYTES, MAX_PART


def read_original(root, ref):
    raw, = verify_refs(root, [ref], max_bytes=MAX_BYTES)
    if len(raw) <= MAX_PART:
        return raw.decode("utf-8")
    return {"zlib_base64": base64.b64encode(zlib.compress(raw)).decode("ascii")}


def original_bytes(value, ref):
    size = ref["bytes"]
    if type(size) is not int or not 0 <= size <= MAX_BYTES:
        raise ValueError("comparison producer exceeds the original projection budget")
    if isinstance(value, str):
        raw = value.encode("utf-8")
    elif isinstance(value, dict) and set(value) == {"zlib_base64"}:
        decoder = zlib.decompressobj()
        try:
            packed = base64.b64decode(value["zlib_base64"], validate=True)
            raw = decoder.decompress(packed, size + 1)
        except (ValueError, TypeError, zlib.error) as exc:
            raise ValueError("invalid compressed comparison producer") from exc
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError("comparison producer stream or expanded length differs")
    else:
        raise ValueError("unknown comparison producer encoding")
    if len(raw) != size or "sha256:" + hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError("comparison original producer bytes differ")
    return raw


def export_chain(root):
    """Store changed map entries once. Every original transition seal is preserved."""
    previous = {}
    chain = []
    for path in sorted(root.glob("transition-*.json")):
        transition = load(path)
        state = transition["state"]
        fields, updates = {}, {}
        for key, value in state.items():
            before = previous.get(key, {})
            if isinstance(value, dict) and isinstance(before, dict) and before.keys() <= value.keys():
                updates[key] = {k:v for k,v in value.items() if k not in before or v != before[k]}
            else:
                fields[key] = value
        chain.append(dict(identity=transition["identity"], previous=transition["previous"], fields=fields, updates=updates))
        previous = state
    return chain


def transitions(chain, campaign_id):
    if not isinstance(chain, list) or len(chain) > 4096:
        raise ValueError("bounded comparison ownership chain required")
    tip, state = campaign_id, {}
    for entry in chain:
        if set(entry) == {"previous", "identity", "fields", "updates"}:
            fields, updates = entry["fields"], entry["updates"]
            if not isinstance(fields, dict) or not isinstance(updates, dict) or fields.keys() & updates.keys():
                raise ValueError("invalid comparison ownership changes")
            merged = dict(fields)
            for key, changes in updates.items():
                if not isinstance(changes, dict) or not isinstance(state.get(key, {}), dict):
                    raise ValueError("invalid comparison ownership map")
                merged[key] = {**state.get(key, {}), **changes}
            entry = dict(previous=entry["previous"], identity=entry["identity"], state=merged)
        if entry["previous"] != tip or seal(entry) != entry:
            raise ValueError("comparison ownership chain differs")
        state, tip = entry["state"], entry["identity"]
        yield entry
