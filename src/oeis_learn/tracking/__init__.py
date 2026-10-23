"""Tracking utilities, including stdlib-only host telemetry.

Load the legacy YAML-backed run manager only when its public classes are used.
Importing a telemetry submodule must not require container-only dependencies.
"""

__all__ = ["RunManager", "RunContext"]


def __getattr__(name):
    if name in __all__:
        from oeis_learn.tracking.run_manager import RunContext, RunManager

        globals().update(RunManager=RunManager, RunContext=RunContext)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
