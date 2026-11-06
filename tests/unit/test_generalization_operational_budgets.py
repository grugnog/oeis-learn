"""Independent retained-model-size accounting without allocating model tensors."""
import pytest
from oeis_learn.generalization.resources import checkpoint_reservation
from oeis_learn.generalization.budgets import reserve_storage


def test_acceptance_peak_counts_resume_and_single_projection():
    # Four retained runs with initial/latest/one milestone, one resume projection
    # and one atomic generation: 4*3 + 3 + 1 = 16 full states. Ten weight copies
    # cover origins plus live initializer, eval and proof views.
    result=checkpoint_reservation(snapshot_bytes=1393781168,weights_bytes=421525760,
        milestones=1,arms=2,seeds=1,acceptance_replay=True,selection=True)
    expected=16*1393781168+10*421525760
    assert result['required_checkpoint_bytes']==expected
    assert result['learner_runs']==4
    assert result['phase_generations']=={'training':16,'proof':14}
    assert expected<26<<30
    assert reserve_storage(expected,1,expected)==expected
    with pytest.raises(ValueError):reserve_storage(expected,1,expected-1)


def test_research_selection_and_more_seeds_reserve_their_live_copies():
    result=checkpoint_reservation(snapshot_bytes=100,weights_bytes=10,milestones=2,arms=2,seeds=3,selection=True)
    assert result['phase_generations']=={'training':29,'proof':26,'selection':34}
    assert result['required_checkpoint_bytes']==34*100+12*10


def test_pac_work_and_ancillary_limits_keep_floors_and_prior_longer_limits():
    from oeis_learn.generalization.budgets import pac_work_allowance,pac_ancillary_allowance
    assert pac_work_allowance('learner',forecast=100,startup=20,publication=30,existing=0)['seconds']==43200
    assert pac_work_allowance('preparation',forecast=40000,startup=100,publication=200,existing=0)['seconds']==127500
    assert pac_work_allowance('learner',forecast=100,startup=20,publication=30,existing=90000)['seconds']==90000
    assert pac_work_allowance('generation',forecast=100,startup=20,publication=30,existing=30000)['seconds']==30000
    missing=pac_work_allowance('learner',forecast=None,startup=20,publication=30,existing=50000)
    assert missing['seconds']==50000 and missing['calibration_gaps']==['forecast']
    for cost in ('cold_start','load','save','hash','copy','exit','resumed_proof'):
        got=pac_ancillary_allowance(cost,observations=[1,100,3],overhead=40,existing=60)
        assert got['seconds']==340
        assert pac_ancillary_allowance(cost,observations=[],overhead=40,existing=500)['seconds']==500
        with pytest.raises(ValueError):pac_ancillary_allowance(cost,observations=[],overhead=40,existing=0)
        with pytest.raises(ValueError):pac_ancillary_allowance(cost,observations=[0],overhead=40,existing=500)


def pac_node(seconds,*,children=(),mode='serial',work=0,cleanup=0,finalization=0,overhead=0,contention=0,schema_max=None):
    return dict(requested=seconds,effective=seconds,work=work,children=list(children),mode=mode,
                cleanup=cleanup,cleanup_floor=cleanup,finalization=finalization,overhead=overhead,contention=contention,schema_max=schema_max)


def test_pac_nested_graph_exact_boundary_parallel_and_short_inner_cap():
    from oeis_learn.generalization.budgets import pac_deadline_graph
    nodes={'load':pac_node(340,work=340),'work':pac_node(43200,work=43200),
           'save':pac_node(600,work=600),'proof':pac_node(500,work=500),
           'learner':pac_node(44700,children=['load','work','save','proof'],cleanup=60),
           'root':pac_node(48360,children=['learner'],cleanup=60,finalization=3600)}
    assert pac_deadline_graph(nodes,root='root')['seconds']==48360
    import copy
    for edit in ('parent','schema','effective','cleanup','missing_finalization'):
        bad=copy.deepcopy(nodes)
        if edit=='parent':bad['root']['requested']-=1;bad['root']['effective']-=1
        if edit=='schema':bad['work']['schema_max']=21600;bad['root']['requested']=999999;bad['root']['effective']=999999
        if edit=='effective':bad['save']['effective']=599
        if edit=='cleanup':bad['learner']['cleanup']=59
        if edit=='missing_finalization':bad['root']['finalization']=3599
        with pytest.raises(ValueError):pac_deadline_graph(bad,root='root')
    parallel={'a':pac_node(100,work=100),'b':pac_node(200,work=200),
              'root':pac_node(3840,children=['a','b'],mode='parallel',contention=40,finalization=3600)}
    assert pac_deadline_graph(parallel,root='root')['seconds']==3840
    parallel['root']['mode']='serial'
    with pytest.raises(ValueError):pac_deadline_graph(parallel,root='root')


def test_pac_optional_arms_add_every_reservation_and_complete_observer_costs():
    from oeis_learn.generalization.resources import pac_campaign_reservation,pac_diagnostic_reservation
    view=dict(preparation=10,training=43200,diagnostics=100,checkpoints=30,generation=23400,export=20)
    base=dict(preparation=86400,qualification=100,acceptance=200,cleanup=60,finalization=3600,existing=0)
    two=pac_campaign_reservation(['B','C'],{'B':view,'C':view},**base)
    four=pac_campaign_reservation(['A','B','C','D'],{a:view for a in 'ABCD'},**base)
    assert four['seconds']-two['seconds']==2*sum(view.values())
    with pytest.raises(ValueError):pac_campaign_reservation(['B','C','D'],{'B':view,'C':view},**base)
    costs=dict(reward_probe=2,grade=3,common_loss_probe=1,semantic_attempt=5,mode_rng_restore=1,publication=10,cleanup=20)
    c=pac_diagnostic_reservation(32,arm='C',costs=costs,update_seconds=100,initial_final_overhead=50,existing=0)
    # 12 probes*2 + 8 grades*3 + 512 losses + 64 semantic*5 +
    # 18 restore events + 2 publications*10 + cleanup20 = 938; 3x + 50.
    assert c['forecast_seconds']==938 and c['seconds']==2864
    assert c['seconds']>c['default_seconds']==70
    for key in costs:
        incomplete=dict(costs);incomplete[key]=None
        with pytest.raises(ValueError):pac_diagnostic_reservation(32,arm='C',costs=incomplete,update_seconds=100,initial_final_overhead=50,existing=0)
