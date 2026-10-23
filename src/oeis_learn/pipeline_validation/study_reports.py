"""Reuse registered full-denominator matched reports and descriptive family intervals."""
from oeis_learn.fresh.comparison import compare_runs, family_bootstrap, promotion


def matched_runs(roots,output):
    # The existing implementation retains missing/interrupted arms, the exact
    # 128+128 denominators, all controls and fixed 2,000 family resamples.
    return compare_runs(roots,output)
