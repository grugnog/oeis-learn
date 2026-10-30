"""Prospective routing only; not registered with the portable host CLI until Phase C."""
from types import MappingProxyType

from oeis_learn.generalization.interfaces import (
    Command, CommandRequest, CommandResult, ScientificStatus, SoftwareStatus,
)


def _unimplemented(request: CommandRequest) -> CommandResult:
    return CommandResult(
        request.operation, SoftwareStatus.BLOCKED, ScientificStatus.UNMEASURED,
        "generalization_phase_c_not_implemented",
    )


HANDLERS = MappingProxyType({command: _unimplemented for command in Command})


def dispatch(request: CommandRequest) -> CommandResult:
    """Always blocked in Phase A; unknown operations fail before any side effects."""
    return HANDLERS[Command(request.operation)](request)
