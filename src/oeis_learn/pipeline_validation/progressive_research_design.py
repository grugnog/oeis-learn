"""Prospective fixed-reference research design, never a guessed execution freeze."""
from pathlib import Path

from .identity import seal
from .progressive_workstation_state import write
from .progressive_research_inputs import _json, checked_file
from oeis_learn.generalization.progressive.contracts import reference, record


def scientific_round(data):
    """Prospective extension of the registered fixed-reference design.

    The requested optimization round retains the full registered search space.
    Feasibility may stop it, but may not silently reduce candidates, looks,
    independent panels, or the protected reserve to manufacture readiness.
    """
    from oeis_learn.generalization.progressive.tuning import SPACE,BASELINE,CONTROL,SHAPES
    if seal(data)!=data or data.get('version')!='progressive-registered-data/v1':
        raise ValueError('original registered-data receipt required')
    from oeis_learn.generalization.progressive.research_budget import ROLE_SECONDS,SETUP_SECONDS,LIMITS_SECONDS
    revised='census_reuse' in data
    return seal(dict(version='progressive-research-round-design/v2' if revised else 'progressive-research-round-design/v1',executable=False,
        original_design=data['design']['identity'],registration=data['identity'],
        selected_stages=['capability','diversity','tuning'],
        seeds=data['design']['seeds'],precision='float32',gpu_owners=1,
        panels=dict(required_per_stratum=128,minimum_groups=16,
            tuning_confirmation=4,separate_retention=['development','diversity','tuning'],
            assessment='explicit locked assessment command only'),
        support=dict(token_horizon=25000000,maximum_shards=64,
            cadence='prospective support-projection-policy/v2 per effective batch' if revised else 'prospective support-projection-policy/v1 before census',
            fresh_initialization='paired random; original D remains an evaluation reference'),
        tuning=dict(learning_trials=12,sizing_trials=4,space=SPACE,shapes=list(SHAPES),
            unchanged_baseline=BASELINE,unpruned_learning_control=CONTROL,
            screen_tokens=4000000,screen_seconds=7200,
            screening_looks=[1000000,2000000,4000000],screening_tasks_per_stratum=32,
            screening_groups_per_stratum=16,screening_tasks_per_group=2,
            **({'screening_group_policy':'balanced-at-most-two/v2'} if revised else {}),
            protected_finalization_seconds=1800,confirmation_tokens=25000000,
            confirmation_seconds=43200,confirmation_schedule_horizon=400000000,
            confirmation_cosine_warmup=1250000,maximum_nonbaseline_finalists=3),
        operational_reservation=dict(role_seconds=ROLE_SECONDS,setup_seconds=SETUP_SECONDS,ancillary_limits_seconds=LIMITS_SECONDS,
            selected_training_seconds=176*3600,selected_total_seconds=312*3600),
        qualification=dict(maximum_profiles=12,maximum_cases=48,
            exact_calibration_maximum_combinations=1296,
            historical_rehearsal='retained original evidence; no transfer of current model authority'),
        feasibility='measure exact selected training and evaluation work; stop a design that cannot fit before scoring',
        later_branches={
            'scaling':'requires confirmed compatible finalist and a separately reserved 100M/400M support horizon',
            'oeis':'requires eligible scaling initializer and independently reserved training/panel support',
            'components':'assess independent eligibility even if OEIS outcome is negative',
            'source_pilot':'assess independent eligibility after capability; no primary scaling claim',
            'assessment':'never part of preparation or adaptive run'},
        later_reservations='no automatic borrowing or unbounded continuation; explicit finite freeze required'))


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
