"""One Lamport SHA-256 publication key per frozen observation boundary.

Each private key signs one immutable scientific statement. The independent pre-score
archive pins the hash of its 512 public values. Each signature discloses one preimage
and the opposite public value per bit, keeping the reader standard-library only.
"""

from hashlib import sha256
import hmac

from oeis_learn.generalization.progressive.contracts import digest, load, seal
from .preparation_recovery import retain


def _pairs(secret, grant, slot):
    seed = hmac.digest(secret, f"round029-publication/v1/{grant}/{slot}".encode(), "sha256")
    return [(hmac.digest(seed, n.to_bytes(2, "big"), "sha256"),
             hmac.digest(seed, (n+1).to_bytes(2, "big"), "sha256")) for n in range(0, 512, 2)]


def _bits(statement):
    raw = bytes.fromhex(digest(statement)[7:])
    return [(byte >> shift) & 1 for byte in raw for shift in range(7, -1, -1)]


def anchors(secret, grant, count):
    return [sha256(b"".join(sha256(x).digest() for pair in _pairs(secret, grant, slot)
                            for x in pair)).hexdigest() for slot in range(count + 1)]


def sign(area, secret, grant, slot, statement, *, readonly):
    # Installing the message first prevents reuse of a signing key after process loss.
    path = area / "publication-commitments" / f"{slot}.json"
    intent = seal(dict(grant=grant, slot=slot, statement=digest(statement)))
    if not readonly:
        retain(path, intent)
    if load(path) != intent:
        raise ValueError("this observation boundary already committed different science")
    proof = bytearray()
    for pair, bit in zip(_pairs(secret, grant, slot), _bits(statement), strict=True):
        proof.extend(pair[bit])
        proof.extend(sha256(pair[1-bit]).digest())
    return dict(slot=slot, proof=proof.hex())


def verify(keys, slot, statement, signature):
    if (set(signature) != {"slot", "proof"} or type(slot) is not int or not 0 <= slot < len(keys)
            or signature["slot"] != slot or not isinstance(signature["proof"], str)
            or len(signature["proof"]) != 256 * 128):
        raise ValueError("invalid publication proof or boundary")
    proof = bytes.fromhex(signature["proof"])
    if len(proof) != 256 * 64:
        raise ValueError("invalid publication proof length")
    public = sha256()
    for i, bit in enumerate(_bits(statement)):
        piece = proof[64*i:64*(i+1)]
        revealed, opposite = sha256(piece[:32]).digest(), piece[32:]
        public.update(opposite + revealed if bit else revealed + opposite)
    if not hmac.compare_digest(public.hexdigest(), keys[slot]):
        raise ValueError("scientific publication differs from its independently committed key")
