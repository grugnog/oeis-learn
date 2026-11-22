"""Explicit host-only migration of round009's original calibration evidence.

This is not qualification reuse. Current-source qualification is mandatory. The
old source/image stay on every measured receipt; only this reviewed migration
allows them to supply unchanged complete-iteration bounds.
"""
import ast
import hashlib
import inspect
from pathlib import Path

from oeis_learn.generalization.progressive.contracts import digest

ORIGINAL_SOURCE = 'sha256:3e9c50d0ef29797d77448e44f0aa0d4daf29e2b73e3d6789bddbf7d1608df797'
PREFIX = 'src/oeis_learn/'
HOST_SCRIPTS = {'scripts/discovery_repair_reservation.py'}
# None of these modules executes learner updates, tensor persistence, native
# evaluation or device calibration. Their new behavior is qualified afresh.
HOST_ONLY = {
    # Outcome-blind panel grouping/diagnosis; exact prepared learner inputs must
    # still match individually before old calibration bounds may be reused.
    'pipeline_validation/progressive_panel_audit.py',
    'pipeline_validation/progressive_panel_revision.py',
    'pipeline_validation/progressive_panel_campaign.py',
    'pipeline_validation/progressive_research_census.py',
    'pipeline_validation/progressive_research_materials.py',
    'pipeline_validation/discovery_cleanup.py',
    'pipeline_validation/progressive_preparation_retirement.py',
    # Prospective roster/scheduling and ownership replay; exact input compatibility
    # still checks every original package and all calibrated execution stays pinned.
    # Host storage admission recognizes the same exact v3/v4 policies as
    # preparation. No update, checkpoint serialization or measurement changes;
    # new-source qualification is still required before consuming these bounds.
    'generalization/progressive/freeze.py',
    'generalization/progressive/research_roster.py',
    'generalization/progressive/runtime_reservation.py',
    'pipeline_validation/progressive_research_design.py',
    'pipeline_validation/progressive_research_bindings.py',
    'pipeline_validation/progressive_study.py',
    'pipeline_validation/progressive_reused_preparation_recovery.py',
    'pipeline_validation/progressive_preparation_preflight.py',
    # Only fresh qualification scheduling changes; old HIP authority is never reused.
    'pipeline_validation/generalization_qualification_io.py',
    'pipeline_validation/progressive_research_qualification.py',
    'pipeline_validation/progressive_trajectory.py',
    'pipeline_validation/progressive_calibration_equivalence.py',
    'pipeline_validation/progressive_calibration_reuse.py',
    'pipeline_validation/progressive_calibration_replay.py',
    'pipeline_validation/progressive_calibration_reuse_workstation.py',
    'pipeline_validation/progressive_completed_calibration_recovery.py',
    'pipeline_validation/progressive_calibration_recovery.py',
    'pipeline_validation/progressive_process_result.py',
    'pipeline_validation/progressive_research_resources.py',
    'pipeline_validation/progressive_capability.py',
    'pipeline_validation/progressive_preparation_owner.py',
    'pipeline_validation/progressive_preparation_reuse.py',
    'pipeline_validation/progressive_freeze_recovery.py',
    'pipeline_validation/progressive_round019_successor.py',
    'pipeline_validation/progressive_round020_successor.py',
    'pipeline_validation/progressive_round022_successor.py',
    'pipeline_validation/progressive_round023_successor.py',
    'pipeline_validation/progressive_round024_successor.py',
    'pipeline_validation/progressive_acceptance_retry.py',
    'pipeline_validation/progressive_research_launch.py',
    'pipeline_validation/progressive_research_workstation.py',
    'pipeline_validation/progressive_workstation.py',
    'pipeline_validation/progressive_workstation_state.py',
    'pipeline_validation/progressive_cli.py',
}
# Runtime task validation and cycle dispatch changed; accept only the exact
# reviewed bytes with the original round009 source and fresh device qualification.
REVIEWED_FILES = {
    'generalization/progressive/contracts.py': 'sha256:223436c6dad6c462e26feb2a3b766e0cb10aa26af825f4510ab354868344e1a9',
    'pipeline_validation/progressive_outputs.py': 'sha256:835e5314aaaaa09af8935162fc902aef2495ddadac134d63b674d0eccff7f7ca',
    'pipeline_validation/progressive_roles.py': 'sha256:d5b8b34435f662b5587d423ffe6146ebbb3630850a0321044d550aeaf1d3e28e',
    'pipeline_validation/progressive_quarantine_control.py': 'sha256:237cf5d807fa24078043fb80cb392a66f7ba03e91f68db30c379f2b561b53573',
    'generalization/progressive/projections.py': 'sha256:f2bd084df568763f06d5d4e7b08876735f20e0c86f6e0f07978bd0e6d7f2c378',
    'generalization/progressive/campaign.py': 'sha256:177707688b3e222d2c33c506daf736c363c44712de59e5d07ccc80056199668c',
    'pipeline_validation/progressive_oeis.py': 'sha256:9ae730e18c6ff3eb32fed6ebeb89a5096a26a870eac684086eead6ffe20c07be',
    'pipeline_validation/progressive_commands.py': 'sha256:88a66f8efe9ebbfd6b972fc1bbca89308ee73d6fca55749c74560e64b2542f06',
    'generalization/progressive/freeze_progress.py': 'sha256:d19809fdb5cabe978cc96c93431d91419f9247e370c405531cad05aa2cef7232',
    'pipeline_validation/qualification_retirement.py': 'sha256:d07bb9b6f11901825c96de6aa0ef3a9740264e2b33a8ed01f79b6670e3423b07',
}
# Historical proofs replay the exact source hashes that were reviewed for round023.
PREVIOUS_REVIEWED = {
    'generalization/progressive/projections.py': 'sha256:b1c310f74aab5e02c9689d459a80b8b09e20cc0f2d876e34a59c0f8026742ac0',
    'pipeline_validation/progressive_commands.py': 'sha256:bb9406f67f2a44cb33435c422569a7c6fdf2307cb0a337f80ee1d9dc5f889e09',
}
PROTECTED = {
    'pipeline_validation/progressive_calibration.py': (('validate_prepared', 'verify_result', 'verify_case'), '9fa7b840a3c615c20be50e23c29763a208f72e6dafd1ab8125bf41a3be234d37'),
    'generalization/progressive/iteration.py': ('validate_producer', '890aa87c7473dcfb00e63ca630f6aa0be48c54344ace9cd1dcbdab31dab30cc6'),
    'pipeline_validation/progressive_runtime_timing.py': (None, '75a28a62544615a65e52b26da341562c6c414df984fed2a47c581211f4f9d8a5'),
}


def validate(original, current, *, package_root=None):
    if digest(original) != ORIGINAL_SOURCE:
        raise ValueError('only the reviewed original round009 source is eligible')
    package_root = Path(package_root or Path(__file__).resolve().parents[1])
    changed = sorted(k for k in set(original) | set(current) if original.get(k) != current.get(k))
    # The reviewed change only handles a late native call during generation.
    # No learner timing, calibration or checkpoint algorithm changed.
    round025 = {
        'generalization/progressive/budgets.py': 'sha256:64819084108419e7ec6885199210cc06deb2252a95263f4421596c69363f7b0b',
        'pipeline_validation/progressive_commands.py': 'sha256:9c10bd4d899ba6a32c336697067f61752d0e32a8f50493c99a4b68e7c978c987',
    }
    for name in changed:
        relative = name.removeprefix(PREFIX)
        if name.startswith(PREFIX) and relative in round025 and current.get(name) == round025[relative]:
            continue
        if name.startswith('tests/'):
            continue
        if name in HOST_SCRIPTS and name in current:
            continue
        if (name.startswith(PREFIX) and name in current and relative in REVIEWED_FILES
                and current[name] in (REVIEWED_FILES[relative], PREVIOUS_REVIEWED.get(relative))):
            continue
        if not name.startswith(PREFIX) or relative not in HOST_ONLY | PROTECTED.keys():
            raise ValueError('calibrated execution source changed: '+name)
        if name not in current:
            raise ValueError('calibrated source removed: '+name)
        if relative in PROTECTED:
            excluded, expected = PROTECTED[relative]
            tree = ast.parse((package_root/relative).read_bytes())
            if excluded:
                tree.body = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) or n.name not in (excluded if isinstance(excluded,tuple) else (excluded,))]
            else:
                tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in {'case_package', 'run_case'}]
            # The reviewed pins were computed on Python 3.12. Python 3.13
            # omits empty fields by default; explicitly retain them where the
            # new option exists. Do not re-pin or relax protected code checks.
            options = {'show_empty': True} if 'show_empty' in inspect.signature(ast.dump).parameters else {}
            actual = hashlib.sha256(ast.dump(tree, include_attributes=False, **options).encode()).hexdigest()
            if actual != expected:
                raise ValueError('protected calibration execution changed: '+name)
    return changed
