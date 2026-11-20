"""Operational headroom, distinct from frozen scientific work entitlements.

Measured feasibility uses 4x observed work: a healthy 3x slowdown still leaves
slack. Unknown cold cases get a substantial watchdog, never an equal share of
their siblings' time. Every allowance remains inside its enclosing reservation.
"""
import math

TIMING_MULTIPLIER = 4
CASE_SECONDS = 3600
HEALTH_SECONDS = 120
CLEANUP_SECONDS = 120
IMAGE_SECONDS = 7200
REPORT_SECONDS = 7200
CPU_ACCEPTANCE_SECONDS = 3300
DEVICE_ACCEPTANCE_SECONDS = 3600
ACCEPTANCE_SECONDS = 7200


def padded_us(measured):
    if type(measured) is not int or measured <= 0:
        raise ValueError('positive original measured microseconds required')
    return TIMING_MULTIPLIER * measured


def case_allowance(remaining, *, maximum=CASE_SECONDS):
    if type(remaining) not in (int, float) or not math.isfinite(remaining):
        raise ValueError('finite remaining case allowance required')
    if type(maximum) not in (int, float) or not math.isfinite(maximum) or maximum <= 0:
        raise ValueError('finite positive case watchdog required')
    if remaining <= 0:
        raise TimeoutError('enclosing batch allowance exhausted')
    # Keep bounded publication/cleanup time. Short diagnostic batches retain the
    # same rule proportionally; their allowance cannot authorize research.
    reserve = min(CLEANUP_SECONDS, remaining / 10)
    return min(maximum, remaining - reserve)
