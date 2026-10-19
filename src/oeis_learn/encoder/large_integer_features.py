"""Finite mathematical features with exact authoritative integers.

log10(n+1) uses the leading 53 bits and a binary exponent. Relative mantissa
truncation is <2**-52, so absolute log error is <1e-12 over the accepted sizes.
No entire large integer is converted to float or decimal.
"""

import math

PRIMES = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59)
PROFILE = "hybrid-chunks/v1"


def exact(value, bits=4096):
    if type(value) is not int or abs(value).bit_length() > bits:
        raise ValueError("input must be a bounded exact integer")
    return value


def signed_log(value):
    exact(value, 131074)
    if value == 0:
        return 0.0
    n = abs(value) + 1
    shift = max(0, n.bit_length() - 53)
    result = 1 + math.log10(n >> shift) + shift * math.log10(2)
    return result if value > 0 else -result


def valuation(value, prime, cap=32):
    if value == 0:
        return cap
    n = abs(value)
    k = 0
    while k < cap and n % prime == 0:
        n //= prime
        k += 1
    return k


def features(sequence, *, bits=4096):
    xs = [exact(x, bits) for x in sequence]
    differences = ([0] + [b - a for a, b in zip(xs, xs[1:])]) if xs else []
    second = ([0] + [b - a for a, b in zip(differences, differences[1:])]) if xs else []
    # Historical stream defines both first two second differences as zero.
    if len(second) > 1:
        second[1] = 0
    return {
        "magnitudes": [signed_log(x) for x in xs],
        "differences": differences,
        "difference_logs": [signed_log(x) for x in differences],
        "second_difference_logs": [signed_log(x) for x in second],
        "residues": [[x % p for p in PRIMES] for x in xs],
        "valuations": [[valuation(x, p) for p in PRIMES] for x in xs],
    }
