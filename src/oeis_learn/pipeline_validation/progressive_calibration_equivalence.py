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
    'pipeline_validation/progressive_research_launch.py',
    'pipeline_validation/progressive_research_workstation.py',
    'pipeline_validation/progressive_workstation.py',
    'pipeline_validation/progressive_workstation_state.py',
    'pipeline_validation/progressive_cli.py',
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
    for name in changed:
        relative = name.removeprefix(PREFIX)
        if name.startswith('tests/'):
            continue
        if name in HOST_SCRIPTS and name in current:
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
