"""Immutable boundary types, with no compatibility or transition algorithms."""
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol


class SoftwareStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    PARTIAL = "PARTIAL"
    INCONCLUSIVE = "INCONCLUSIVE"
    INTERRUPTED = "INTERRUPTED"


class ScientificStatus(StrEnum):
    UNMEASURED = "UNMEASURED"
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    INCONCLUSIVE = "INCONCLUSIVE"


EXIT_CODES = MappingProxyType({
    SoftwareStatus.PASS: 0, SoftwareStatus.FAIL: 1, SoftwareStatus.BLOCKED: 3,
    SoftwareStatus.PARTIAL: 3, SoftwareStatus.INCONCLUSIVE: 3,
    SoftwareStatus.INTERRUPTED: 130,
})


class Command(StrEnum):
    PREFLIGHT = "preflight"
    QUALIFY = "qualify"
    PREPARE = "prepare"
    EVALUATE = "evaluate"
    TRAIN = "train"
    REPLAY = "replay"
    REPORT = "report"
    VERIFY_BUNDLE = "verify-bundle"
    ACCEPTANCE = "acceptance"


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    path: str
    sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class QualificationProducer:
    """Original PASS evidence reference; compatibility is not established by this type."""
    contract: ArtifactRef
    receipt: ArtifactRef
    artifacts: tuple[ArtifactRef, ...]


@dataclass(frozen=True, slots=True)
class WeightsOnlyCheckpoint:
    weights: ArtifactRef
    initialization: str


@dataclass(frozen=True, slots=True)
class FullStateCheckpoint:
    weights: ArtifactRef
    state: ArtifactRef
    contract_identity: str
    tokens: int
    updates: int
    pins: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CommandRequest:
    operation: Command
    output: str
    manifest: ArtifactRef | None = None
    qualification: QualificationProducer | None = None
    checkpoints: tuple[str, ...] = ()
    arms: tuple[str, ...] = ()
    seeds: tuple[int, ...] = ()
    inputs: tuple[ArtifactRef, ...] = ()
    archive: ArtifactRef | None = None
    resume: FullStateCheckpoint | None = None
    proposal_bank: ArtifactRef | None = None
    case: str | None = None


@dataclass(frozen=True, slots=True)
class CommandResult:
    operation: Command
    software_status: SoftwareStatus
    scientific_status: ScientificStatus
    reason: str
    artifacts: tuple[ArtifactRef, ...] = ()


@dataclass(frozen=True, slots=True)
class PreparationResult:
    software_status: SoftwareStatus
    study: ArtifactRef | None
    manifests: tuple[ArtifactRef, ...]
    shortfalls: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SelectionResult:
    checkpoint_identity: str | None
    eligible: tuple[str, ...]
    reason: str


class PreparationPolicy(Protocol):
    def prepare(self, request: CommandRequest) -> PreparationResult: ...


class SelectionPolicy(Protocol):
    def select(self, request: CommandRequest) -> SelectionResult: ...


class CommandHandler(Protocol):
    def __call__(self, request: CommandRequest) -> CommandResult: ...
