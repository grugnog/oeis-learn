"""Outcome-blind assembly adapters; no hidden model scoring or guessed inputs."""


def census(*args, **kwargs):
    from oeis_learn.generalization.progressive.cohorts import census as count

    return count(*args, **kwargs)


def publish_cohort(*args, **kwargs):
    from oeis_learn.generalization.progressive.cohorts import publish

    return publish(*args, **kwargs)
