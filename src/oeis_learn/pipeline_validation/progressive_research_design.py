"""Prospective fixed-reference research design, never a guessed execution freeze."""
from pathlib import Path

from .identity import seal
from .progressive_workstation_state import write
from .progressive_research_inputs import _json, checked_file
from oeis_learn.generalization.progressive.contracts import reference, record


def construct(root, output, manifest, registration):
    """Resolve original model choices before scores, retain finite unmet inputs.

    No larger tuning domain is implicitly selected. The publication is a design
    proposal; original measured calibration and fresh membership are still needed
    by the existing canonical prepare/freeze operations.
    """
    from oeis_learn.generalization.progressive.models import resolve_profile
    from oeis_learn.generalization.progressive.budgets import KEYS
    from oeis_learn.generalization.progressive.sensitivity import default_design, validate_design
    root, output = Path(root), Path(output)
    refs = []
    def put(name, value, *, kind='policy'):
        path = output / name
        write(path, value)
        ref = reference(root, path)
        refs.append(dict(name=name.removesuffix('.json'), role='custodian', kind=kind, artifact=ref))
        return ref
    seeds = [20261016, 20261017]
    recipe = _json(checked_file(root, manifest['retained']['recipe']))
    profiles = [resolve_profile(recipe, seed=seed, shape='reference', dropout=recipe['model']['dropout']) for seed in seeds]
    put('selected-profiles.json', seal(dict(version='progressive-selected-research-profiles/v1',
        profiles=profiles, precision='fp32', tuning_selected=False)))
    design = default_design(seed=20261110, repeats=8)
    design['scenarios'] = [s for s in design['scenarios'] if s['claim'] == 'capability']
    validate_design(design)
    put('sensitivity-design.json', design)
    protocol = seal(dict(version='progressive-research-design/v1', executable=False,
        selected_stages=['capability', 'diversity'], seeds=seeds,
        retained_D=manifest['retained']['checkpoints']['D']['descriptor']['identity'],
        controls=dict(B='retained provenance/control, no new replication', D='capability reference'),
        initialization=dict(capability='retained_D_weights', diversity='paired_fresh_random'),
        diversity=dict(arms=['static', 'refreshed'], charged_token_target=25000000,
            task_seconds=43200, tasks=4, allocation_block_updates=8,
            proposed_refresh_updates=8, refresh_feasibility='must pass exact target-token/shard supply before freeze',
            familiar_ppm=200000, frontier_ppm=100000, adaptive_ppm=700000),
        panels=dict(strata=['parameter', 'composition', 'unseen_structure', 'oeis'],
            required_per_stratum=128, capability_per_stratum=128, minimum_groups=16, visible=20, total=100,
            roles=['development', 'confirmation-diversity', 'assessment'],
            prior_exposures='all registered originals including fresh-002 excluded',
            deficit_policy='publish availability; no post-outcome reseeding or reduced quota'),
        evaluation=dict(attempts=8, portfolio_seconds=60, fit_seconds=4, token_cap=1024,
            schedule='drain', methods=['retained_D', 'random', 'shuffled', 'absent',
                                       'templates', 'polynomial', 'linear_recurrence']),
        reservations_seconds=dict(training=48*3600, preparation=24*3600,
            qualification_acceptance=12*3600, evaluation=96*3600, packaging=4*3600),
        evaluation_feasibility=dict(capability_portfolio_seconds=128*4*7*60,
            diversity_portfolio_seconds=128*3*2*2*60,
            scope='prospective portfolio ceilings only; startup/scoring/storage still require measurement',
            screening_trials_selected=False),
        hardware=dict(memory_gib=64, cpus=8, host_free_memory_gib=24, gpu_owners=1),
        sustained=dict(status='PENDING_ELIGIBLE_TWO_TASK_PROTOCOL',
            no_automatic_extra_soak=True, no_task_extension=True),
        not_selected=['tuning', 'scaling', 'oeis', 'components', 'source_pilot'],
        source_selection='original exact indexed OEIS snapshot required for fresh transfer panels'))
    put('research-design.json', protocol)
    caps = dict.fromkeys(KEYS, 0)
    caps.update(tokens=100_100_000, draws=4_000_000, rejections=4_000_000,
                attempts=4_000_000, native_work=10_000_000_000,
                elapsed_us=sum(protocol['reservations_seconds'].values())*1_000_000,
                disk_bytes=256 << 30, archive_bytes=512 << 20)
    unresolved = ['fresh_indexed_oeis_snapshot', 'fresh_cohort_and_training_support',
                  'measured_exact_input_calibration', 'current_research_model_qualification',
                  'finite_evaluation_and_refresh_supply']
    blueprint = record('blueprint', dict(executable=False, selected_stages=['capability', 'diversity'],
        references=refs, unresolved=unresolved, proposed_caps=caps,
        reference_campaign=str(output.relative_to(root))), created_at='2026-11-10T00:00:00Z',
        purpose='research', track='synthetic_strict', parents=[registration['identity']])
    write(output / 'blueprint.json', blueprint)
    return blueprint
