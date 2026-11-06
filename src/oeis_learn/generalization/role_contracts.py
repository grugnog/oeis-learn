"""Declarative role mounts. Actual confinement/enforcement is Phase B/C work.

Channels denote separate payload roots, never keys in a shared truth document.
All input mounts are read-only; every role has its own output root.
"""
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class Role(StrEnum):
    PREPARATION = "preparation"
    LEARNER = "learner"
    GENERATOR = "generator"
    FITTER = "fitter"
    SELECTION = "selection"
    ASSESSMENT = "assessment"
    REPORT = "report"


@dataclass(frozen=True, slots=True)
class MountDeclaration:
    channel: str
    destination: str
    read_only: bool = True


ROLE_INPUTS = MappingProxyType({
    Role.PREPARATION: (
        MountDeclaration("mechanical_sources", "/inputs/mechanical"),
        MountDeclaration("isolation_truth", "/inputs/custodian-truth"),
    ),
    Role.LEARNER: (
        MountDeclaration("training_targets", "/inputs/training"),
        MountDeclaration("verified_training_replay", "/inputs/replay"),
    ),
    Role.GENERATOR: (
        MountDeclaration("visible_prompts", "/inputs/prompts"),
        MountDeclaration("frozen_weights", "/inputs/weights"),
    ),
    Role.FITTER: (
        MountDeclaration("visible_prompts", "/inputs/prompts"),
        MountDeclaration("generated_candidates", "/inputs/candidates"),
    ),
    Role.SELECTION: (
        MountDeclaration("selection_supervision", "/inputs/selection-truth"),
        MountDeclaration("selection_candidates", "/inputs/selection-candidates"),
    ),
    Role.ASSESSMENT: (
        MountDeclaration("assessment_truth", "/inputs/assessment-truth"),
        MountDeclaration("frozen_assessment_authorization", "/inputs/authorization"),
        MountDeclaration("frozen_assessment_candidates", "/inputs/frozen-candidates"),
    ),
    Role.REPORT: (MountDeclaration("public_evidence", "/inputs/evidence"),),
})
ROLE_OUTPUTS = MappingProxyType({
    role: MountDeclaration("role_output", f"/output/{role.value}", False) for role in Role
})


def pac_training_refs(prepared,arm):
    """Only training payloads cross into the learner; pair proofs stay custodian-side."""
    fields={'version','role','literal','parameterized','probes','inventory','pairs','identity'}
    if (set(prepared)!=fields or prepared['version']!='pac-prepared/v1'
        or prepared['role']!='training' or arm not in {'A','B','C','D'}):
        raise ValueError('closed training-only PAC projection required')
    return [*prepared['literal' if arm=='A' else 'parameterized'],prepared['probes'],prepared['inventory']]
