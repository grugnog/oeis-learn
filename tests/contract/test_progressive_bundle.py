"""Public bundle authority is independently reduced, never a supplied PASS."""

import json
import zipfile
import pytest
from oeis_learn.generalization.progressive.contracts import atomic_write, digest, seal, reference
from oeis_learn.generalization.progressive.report import build
from oeis_learn.generalization.progressive.verify import verify


def test_report_campaign_reference_and_embedded_record_follow_their_contracts():
    from oeis_learn.generalization.progressive.contracts import schema

    properties = schema("report")["properties"]
    assert properties["campaign"] == {"$ref": "progressive-common.schema.json#/$defs/hash"}
    assert properties["campaign_record"]["oneOf"] == [
        {"type": "null"}, schema("campaign")
    ]


def fixture(root):
    roster = [dict(task=digest("task"), group=digest("group"), stratum="composition")]
    rows = [
        dict(roster[0], seed=s, arm=a, outcome="failure")
        for s in [1, 2]
        for a in ["baseline", "candidate"]
    ]
    docs = dict(roster=roster, outcomes=rows, accounting=[], ancestry=[], qualifications=[])
    tables = {}
    for k, v in docs.items():
        atomic_write(root / (k + ".json"), v)
        tables[k] = [reference(root, k + ".json")]
    request = seal(
        dict(
            version="progressive-public-report-input/v1",
            campaign_record=None,
            scope_record=None,
            terminal=None,
            campaign=digest("campaign"),
            source=digest("source"),
            seeds=[1, 2],
            design=dict(claim="capability", bootstrap_seed=13, comparisons=1, links=[], costs=None),
            scope=dict(required=list(docs), qualification_scopes=[], tasks=[]),
            tables=tables,
            decisions=[],
            producer_error=None,
            collector_errors=[],
        )
    )
    return bind_scope(root, request, roster)


def test_bounded_closed_export_and_independent_incomplete_verdict(tmp_path):
    request = fixture(tmp_path)
    report = build(tmp_path, tmp_path / "report.zip", request)
    assert report["status"] == "COMPLETE"
    result = verify(tmp_path / "report.zip")
    assert result["status"] == "VERIFIED"
    assert result["decision"] == "INCOMPLETE"
    with zipfile.ZipFile(tmp_path / "report.zip") as z:
        assert all("private" not in p and not p.endswith(".pt") for p in z.namelist())


def rewrite(path, edit):
    with zipfile.ZipFile(path) as z:
        docs = {n: z.read(n) for n in z.namelist()}
    edit(docs)
    with zipfile.ZipFile(path, "w") as z:
        for n, b in docs.items():
            z.writestr(n, b)


@pytest.mark.parametrize("change", ["counts", "missing", "gate", "extra", "zip_path"])
def test_forged_or_missing_evidence_never_verifies(tmp_path, change):
    build(tmp_path, tmp_path / "report.zip", fixture(tmp_path))

    def edit(docs):
        manifest = json.loads(docs["manifest.json"])
        if change == "counts":
            manifest["summary"]["outcomes"] = 99
        if change == "gate":
            manifest["decisions"] = [dict(verdict="CONFIRMED_GAIN", action="ADVANCE_REGISTERED")]
        if change == "missing":
            del docs["tables/roster-0000.json"]
        if change == "extra":
            docs["private.json"] = b"{}"
        if change == "zip_path":
            docs["../escape.json"] = b"{}"
        docs["manifest.json"] = json.dumps(seal(manifest)).encode()

    rewrite(tmp_path / "report.zip", edit)
    with pytest.raises(ValueError):
        verify(tmp_path / "report.zip")


def test_export_bounds_and_missing_qualification_are_explicit(tmp_path):
    request = fixture(tmp_path)
    request = seal(dict(request, scope=dict(request["scope"], qualification_scopes=["hip-short"])))
    request = bind_scope(tmp_path, request, json.loads((tmp_path / "roster.json").read_text()))
    build(tmp_path, tmp_path / "report.zip", request)
    assert verify(tmp_path / "report.zip")["status"] == "INCOMPLETE"
    with pytest.raises(ValueError):
        build(tmp_path, tmp_path / "tiny.zip", request, max_bytes=20)


def test_independent_reducer_matches_gold_paired_negative_and_positive():
    from tests.unit.test_progressive_decisions import fixture as panel
    from oeis_learn.generalization.progressive.verify import reduce_outcomes

    data, roster, design, _ = panel()
    metrics, verdict = reduce_outcomes(data, roster, [1, 2], design)
    assert verdict == "CONFIRMED_GAIN" and metrics["mean_gain"] > 0
    for r in data:
        if r["arm"] == "candidate" and r["stratum"] != "retention":
            r["outcome"] = "failure"
    metrics, verdict = reduce_outcomes(data, roster, [1, 2], design)
    assert verdict == "NO_MATERIAL_GAIN" and metrics["upper"] == 0
    for r in data:
        if r["arm"] == "baseline" and r["stratum"] != "retention":
            r["outcome"] = "success"
    metrics, verdict = reduce_outcomes(data, roster, [1, 2], design)
    assert verdict == "REGRESSION" and metrics["upper"] == -1


def test_original_qualification_producer_is_required_and_replayed(tmp_path):
    request = fixture(tmp_path)
    producer = seal(
        dict(
            version="progressive-hip-case/v1",
            source=request["source"],
            scope="short",
            device="actual_HIP",
            status="PASS",
            requested=[
                "finite_update",
                "parameter_change",
                "full_restore",
                "deterministic_continuation",
                "generation",
                "native_reference_score",
                "cleanup",
            ],
            executed=[
                "finite_update",
                "parameter_change",
                "full_restore",
                "deterministic_continuation",
                "generation",
                "native_reference_score",
                "cleanup",
            ],
            measurements=dict(
                finite_update=True,
                parameter_change=True,
                full_restore=True,
                deterministic_continuation=True,
                native_reference_score=True,
                generation=True,
                cleanup=True,
            ),
        )
    )
    atomic_write(tmp_path / "hip.json", producer)
    row = {
        k: producer[k]
        for k in ("identity", "source", "scope", "device", "status", "requested", "executed")
    }
    row["receipt"] = reference(tmp_path, "hip.json")
    (tmp_path / "qualifications.json").unlink()
    atomic_write(tmp_path / "qualifications.json", [row])
    request = seal(
        dict(
            request,
            tables=dict(
                request["tables"], qualifications=[reference(tmp_path, "qualifications.json")]
            ),
            scope=dict(request["scope"], qualification_scopes=["short"]),
        )
    )
    request = bind_scope(tmp_path, request, json.loads((tmp_path / "roster.json").read_text()))
    build(tmp_path, tmp_path / "report.zip", request)
    assert verify(tmp_path / "report.zip")["status"] == "VERIFIED"

    def edit(docs):
        name = next(n for n in docs if n.startswith("producers/"))
        del docs[name]

    rewrite(tmp_path / "report.zip", edit)
    with pytest.raises(ValueError):
        verify(tmp_path / "report.zip")


def test_failed_terminal_export_keeps_first_and_collector_errors(tmp_path):
    from oeis_learn.generalization.progressive.report import terminal_request

    atomic_write(
        tmp_path / "result.json",
        seal(
            dict(
                version="discovery-result/v1",
                campaign=None,
                producer_error=dict(
                    type="RuntimeError", code="FIRST", producer="learner", local_traceback=None
                ),
                collector_errors=[
                    dict(type="OSError", code="SECOND", producer="collector", local_traceback=None)
                ],
            )
        ),
    )
    request = terminal_request(tmp_path, tmp_path)
    build(tmp_path, tmp_path / "failed.zip", request)
    result = verify(tmp_path / "failed.zip")
    assert result["status"] == "INCOMPLETE"
    assert result["producer_error"]["code"] == "FIRST"
    assert result["collector_errors"][0]["code"] == "SECOND"


def bind_scope(root, request, roster):
    from tests.unit.test_progressive_freeze import campaign

    qrefs = request["tables"].get("qualifications", [])
    qualifiers = []
    for ref in qrefs:
        qualifiers.extend(json.loads((root / ref["path"]).read_text()))
    scope = seal(
        dict(
            version="progressive-public-scope/v1",
            source=request["source"],
            scope=request["scope"],
            seeds=request["seeds"],
            design=request["design"],
            roster=digest(roster),
            qualification_producers={q["scope"]: q["identity"] for q in qualifiers},
        )
    )
    name = "scope-" + scope["identity"].split(":")[1] + ".json"
    if not (root / name).exists():
        atomic_write(root / name, scope)
    ref = reference(root, name)
    c = campaign()
    c["payload"]["references"].append(
        dict(name="public-scope", role="verifier", kind="policy", artifact=ref)
    )
    c = seal(c)
    return seal(dict(request, campaign=c["identity"], campaign_record=c, scope_record=ref))


def test_resealing_cannot_weaken_prospectively_frozen_scope(tmp_path):
    request = fixture(tmp_path)
    build(tmp_path, tmp_path / "report.zip", request)

    def edit(docs):
        m = json.loads(docs["manifest.json"])
        m["scope"]["required"] = []
        m["seeds"] = [1]
        docs["manifest.json"] = json.dumps(seal(m)).encode()

    rewrite(tmp_path / "report.zip", edit)
    with pytest.raises(ValueError, match="frozen authority"):
        verify(tmp_path / "report.zip")


def test_efficiency_uses_measured_costs_instead_of_supplied_summary():
    from tests.unit.test_progressive_decisions import fixture as panel
    from oeis_learn.generalization.progressive.verify import reduce_outcomes

    data, roster, design, _ = panel()
    for row in data:
        row["outcome"] = "success"
    design = dict(
        design,
        claim="efficiency",
        costs=dict(
            metric="total_active_us", scope="preparation_search_training", amortization="none"
        ),
    )
    assert reduce_outcomes(data, roster, [1, 2], design)[1] == "INCOMPLETE"
    assert (
        reduce_outcomes(data, roster, [1, 2], design, costs=dict(baseline=100, candidate=80))[1]
        == "CONFIRMED_EFFICIENCY"
    )
    assert (
        reduce_outcomes(data, roster, [1, 2], design, costs=dict(baseline=100, candidate=90))[1]
        == "NO_MATERIAL_GAIN"
    )

@pytest.mark.parametrize('change,expected', [('none','CONFIRMED_GAIN'),('censored','INCOMPLETE'),('regression','REGRESSION'),('seed','INCOMPLETE'),('retention','INCONCLUSIVE')])
def test_oeis_endpoint_has_independent_paired_verdict(change, expected):
    from oeis_learn.generalization.progressive.decisions import recompute
    from oeis_learn.generalization.progressive.verify import reduce_outcomes
    roster = [dict(task=digest(['oeis',g]),group=digest(['group',g]),stratum='oeis') for g in range(16)]
    roster += [dict(task=digest('retention'),group=digest('retention-group'),stratum='retention')]
    seeds = [1] if change == 'seed' else [1,2]
    rows=[]
    for seed in seeds:
        for arm in ['baseline','candidate']:
            for member in roster:
                outcome='success' if member['stratum']=='retention' or arm=='candidate' else 'failure'
                if change=='censored' and arm=='candidate': outcome='unresolved'
                if change=='regression' and member['stratum']=='oeis':outcome='success' if arm=='baseline' else 'failure'
                if change=='retention' and member['stratum']=='retention' and arm=='candidate':outcome='failure'
                rows.append(dict(member,seed=seed,arm=arm,outcome=outcome))
    design=dict(claim='capability',bootstrap_seed=13,comparisons=2,links=[],costs=None,primary=['oeis'])
    assert recompute(rows,roster,seeds,design)[1]==expected
    assert reduce_outcomes(rows,roster,seeds,design)[1]==expected


def test_oeis_endpoint_roundtrips_bound_public_bundle(tmp_path):
    request = fixture(tmp_path)
    roster = [dict(task=digest(['oeis', g]), group=digest(['group', g]), stratum='oeis') for g in range(16)]
    roster += [dict(task=digest('retention'),group=digest('retention-group'),stratum='retention')]
    rows = [dict(r,seed=s,arm=a,outcome='success' if a=='candidate' or r['stratum']=='retention' else 'failure')
            for r in roster for s in [1,2] for a in ['baseline','candidate']]
    for name, data in [('roster',roster),('outcomes',rows)]:
        (tmp_path/(name+'.json')).write_text(json.dumps(data))
        request['tables'][name] = [reference(tmp_path,name+'.json')]
    request['design']['primary'] = ['oeis']
    request = bind_scope(tmp_path,seal(request),roster)
    build(tmp_path,tmp_path/'oeis.zip',request)
    assert verify(tmp_path/'oeis.zip')['decision']=='CONFIRMED_GAIN'


def test_subsidiary_comparison_cannot_verify_without_original_binding(tmp_path):
    request = fixture(tmp_path)
    comparison = dict(id=digest('unbound'), rows=json.loads((tmp_path/'outcomes.json').read_text()), roster=json.loads((tmp_path/'roster.json').read_text()), seeds=[1,2],
        design=request['design'], verdict='INCOMPLETE', action='EXPLORE_RESERVED')
    atomic_write(tmp_path/'comparisons.json',[comparison])
    request = seal(dict(request,tables=dict(request['tables'],comparisons=[reference(tmp_path,'comparisons.json')])))
    build(tmp_path,tmp_path/'report.zip',request)
    result = verify(tmp_path/'report.zip')
    assert result['status'] == 'INCOMPLETE'
    assert 'comparison_binding:'+comparison['id'] in result['missing']


def test_exact_public_comparison_rejects_filtered_frozen_panel():
    from oeis_learn.generalization.progressive.public_comparisons import verify_authority
    roster = [dict(task=digest('a'),group=digest('g'),stratum='composition')]
    panel = seal(dict(version='discovery-cohort/v1',purpose='acceptance',track='synthetic_strict',
        payload=dict(role='confirmation',rows=[dict(id=r['task'],groups=[r['group']],stratum=r['stratum']) for r in roster])))
    design = dict(claim='capability',bootstrap_seed=0,comparisons=1,links=[],costs=None)
    registration = seal(dict(version='progressive-gate-registration/v1',source_slot='a',target_slot='b',
        panels=[panel['identity']],seeds=[1,2],design=digest(design),producers=[]))
    comparison = dict(id=digest('comparison'),roster=[],seeds=[1,2],design=design,
        binding=dict(registration=registration['identity'],source_slot='a',target_slot='b',producers=[]))
    with pytest.raises(ValueError,match='exact original panel'):
        verify_authority(comparison,[panel,registration],dict(purpose='acceptance',track='synthetic_strict'))


def test_comparison_efficiency_requires_original_task_costs():
    from oeis_learn.generalization.progressive.public_comparisons import measured_costs
    campaign = dict(payload=dict(graph=dict(nodes=[dict(id='scope',dependencies=[])]),tasks=[
        dict(id='base',slot='scope',seed=1,arm='static'),dict(id='candidate',slot='scope',seed=1,arm='refreshed')]))
    comparison = dict(design=dict(claim='efficiency'),binding=dict(source_slot='scope'),seeds=[1],
        costs=dict(baseline=[digest('base')],candidate=[digest('candidate')]))
    accounting = [dict(task=digest('base'),elapsed_us=100),dict(task=digest('candidate'),elapsed_us=80)]
    assert measured_costs(comparison,campaign,accounting) == dict(baseline=100,candidate=80)
    assert measured_costs(comparison,campaign,accounting[:1]) is None
    comparison['costs']['candidate'].append(digest('foreign'))
    with pytest.raises(ValueError,match='task scope'):
        measured_costs(comparison,campaign,accounting)


@pytest.mark.parametrize('claim',['capability','efficiency'])
def test_original_static_comparison_producers_roundtrip_and_reject_tampering(tmp_path,claim):
    from tests.unit.test_progressive_decisions import fixture as panel
    from tests.unit.test_progressive_cohorts import fixture as cohort_fixture
    request = fixture(tmp_path)
    rows, roster, design, _ = panel()
    if claim=='efficiency':
        for row in rows:row['outcome']='success'
        design=dict(design,claim='efficiency',costs=dict(metric='total_active_us',scope='preparation_search_training',amortization='none'))
    cohort, _ = cohort_fixture(tmp_path/'cohort')
    template = cohort['payload']['rows'][0]
    cohort['payload']['rows'] = [dict(template,id=r['task'],groups=[r['group']],stratum=r['stratum']) for r in roster]
    cohort = seal(cohort)
    atomic_write(tmp_path/'comparison-panel.json',cohort)
    atomic_write(tmp_path/'comparison-design.json',design)
    registered, originals = [], []
    for seed in (1,2):
        for arm in ('baseline','candidate'):
            prefix = arm+str(seed)
            generated = seal(dict(version='discovery-candidates/v1',checkpoint=digest(prefix),model=digest('model'),records={r['task']:{} for r in roster}))
            atomic_write(tmp_path/(prefix+'-candidates.json'),generated)
            candidate_ref = reference(tmp_path,prefix+'-candidates.json')
            outcomes={r['task']:dict(equal_budget=dict(success=r['outcome']=='success',complete=True,outcome=r['outcome'])) for r in rows if r['seed']==seed and r['arm']==arm}
            scored = seal(dict(version='discovery-scoring/v1',candidates=candidate_ref['sha256'],denominator=len(roster),outcomes=outcomes))
            atomic_write(tmp_path/(prefix+'-scores.json'),scored)
            registered.append(dict(arm=arm,seed=seed,checkpoint=dict(path=prefix+'.pt',bytes=1,sha256=digest(prefix)),model=digest('model'),candidates=candidate_ref,scores=reference(tmp_path,prefix+'-scores.json')))
            originals.append(dict(candidates=(tmp_path/(prefix+'-candidates.json')).read_text(),scores=(tmp_path/(prefix+'-scores.json')).read_text()))
    registration = seal(dict(version='progressive-gate-registration/v1',source_slot='diversity',target_slot='capability',panels=[cohort['identity']],seeds=[1,2],design=digest(design),producers=registered))
    atomic_write(tmp_path/'registration.json',registration)
    authorities=[];campaign=request['campaign_record']
    for name in ('comparison-panel','comparison-design','registration'):
        ref=reference(tmp_path,name+'.json')
        campaign['payload']['references'].append(dict(name=name,role='verifier',kind='groups' if name=='comparison-panel' else 'policy',artifact=ref))
        authorities.append(dict(artifact=ref,raw=(tmp_path/(name+'.json')).read_text()))
    if claim=='efficiency':
        import copy
        template=campaign['payload']['tasks'][0]
        plans=[];accounting=[]
        for seed in (1,2):
            for arm in ('static','refreshed'):
                task=dict(copy.deepcopy(template),id=arm+str(seed),arm=arm,seed=seed,output=arm+str(seed))
                plans.append(task)
                accounting.append(dict(task=digest(task['id']),arm='baseline' if arm=='static' else 'candidate',tokens=0,useful_tokens=0,discarded_tokens=0,elapsed_us=50 if arm=='static' else 40,token_cap=task['caps']['tokens'],elapsed_cap_us=task['caps']['elapsed_us']))
        campaign['payload']['tasks']=plans
        next(n for n in campaign['payload']['graph']['nodes'] if n['id']=='diversity')['caps']={k:sum(t['caps'][k] for t in plans) for k in template['caps']}
        atomic_write(tmp_path/'measured-accounting.json',accounting)
        request['tables']['accounting']=[reference(tmp_path,'measured-accounting.json')]
        request['scope']['tasks']=[digest(t['id']) for t in plans]
        request=bind_scope(tmp_path,seal(request),json.loads((tmp_path/'roster.json').read_text()))
        # Keep the augmented comparison authority campaign and update only the
        # prospectively frozen scope reference returned by bind_scope.
        campaign['payload']['references']=[r for r in campaign['payload']['references'] if r['name']!='public-scope']+[r for r in request['campaign_record']['payload']['references'] if r['name']=='public-scope']
    campaign=seal(campaign)
    comparison=dict(id=digest('original'),rows=rows,roster=roster,seeds=[1,2],design=design,verdict='CONFIRMED_GAIN' if claim=='capability' else 'CONFIRMED_EFFICIENCY',action='STOP_AFFECTED',binding=dict(registration=registration['identity'],source_slot='diversity',target_slot='capability',producers=originals))
    if claim=='efficiency':
        from oeis_learn.generalization.progressive.public_comparisons import cost_tasks
        comparison['costs']=cost_tasks(campaign,'diversity',[1,2])
    atomic_write(tmp_path/'original-comparison.json',[comparison]);atomic_write(tmp_path/'authorities.json',authorities)
    request=seal(dict(request,campaign=campaign['identity'],campaign_record=campaign,tables=dict(request['tables'],comparisons=[reference(tmp_path,'original-comparison.json')],authorities=[reference(tmp_path,'authorities.json')])))
    build(tmp_path,tmp_path/'original.zip',request)
    assert verify(tmp_path/'original.zip')['status']=='VERIFIED'
    comparison['binding']['producers'][0]['scores']+=' '
    atomic_write(tmp_path/'tampered-comparison.json',[comparison])
    request=seal(dict(request,tables=dict(request['tables'],comparisons=[reference(tmp_path,'tampered-comparison.json')])))
    build(tmp_path,tmp_path/'tampered.zip',request)
    with pytest.raises(ValueError,match='producer bytes'):
        verify(tmp_path/'tampered.zip')


def test_reserved_public_exploration_replays_trigger_and_original_budget(tmp_path):
    from oeis_learn.generalization.progressive.public_comparisons import verify_exploration
    from oeis_learn.generalization.progressive.targets import defaults
    from tests.helpers.progressive_fixture import caps
    design=dict(claim='capability',bootstrap_seed=0,comparisons=1,links=[],costs=None)
    policy=defaults();policy['payload']['selected']=True
    target=next(t for t in policy['payload']['targets'] if t['id']=='gain')
    target.update(allowed_actions=['EXPLORE_RESERVED'],budget_ref='diagnostic')
    panel=seal(dict(version='discovery-cohort/v1',payload=dict(role='confirmation')))
    branch=dict(id='diagnostic',failure_class='weak-gain',trigger='heldout-negative',signal='heldout_yield',
        design=digest(design),panel=panel['identity'],task_slots=['diagnostic'],reservation=caps(),maximum_visits=1,terminal_action='RETAIN_BASELINE',assessment=False)
    policy['payload']['diagnostics']=[branch];policy=seal(policy)
    campaign=dict(identity=digest('campaign'),payload=dict(target_policy=policy['identity'],caps=caps(),
        graph=dict(nodes=[dict(id='diagnostic',kind='diagnostic')],edges=[dict(source='source',target='diagnostic',condition='NO_MATERIAL_GAIN',action='EXPLORE_RESERVED')],diagnostic_pool=caps()),
        tasks=[dict(id='diagnostic-task',slot='diagnostic',caps=caps())]))
    decision=seal(dict(payload=dict(next_action='EXPLORE_RESERVED')))
    bundle=seal(dict(source_slot='source',target_slot='diagnostic',decision=decision,rows=[],roster=[],seeds=[1,2],design=design,kwargs=dict(target_id='gain',action='EXPLORE_RESERVED')))
    audit=seal(dict(bundle=bundle['identity'],kwargs=bundle['kwargs'],prior_decisions=[],budget_events=[]))
    transition=seal(dict(previous=campaign['identity'],state=dict(decisions={'edge':dict(bundle=bundle['identity'])})))
    comparison=dict(id=decision['identity'],action='EXPLORE_RESERVED',verdict='NO_MATERIAL_GAIN',rows=[],roster=[],seeds=[1,2],design=design,
        binding=dict(source_slot='source',target_slot='diagnostic',bundle=bundle,authorization=audit,chain=[transition]))
    metrics=dict(mean_gain=0,lower=0,seed_effects={'1':0,'2':0},groups=16)
    assert verify_exploration(comparison,[panel,policy],campaign,metrics)==[]
    with pytest.raises(ValueError,match='trigger unsupported'):
        verify_exploration(comparison,[panel,policy],campaign,dict(metrics,mean_gain=.1,lower=.05,seed_effects={'1':.1,'2':.1}))
    campaign['payload']['graph']['diagnostic_pool']['elapsed_us']=0
    with pytest.raises(ValueError,match='reservation exhausted'):
        verify_exploration(comparison,[panel,policy],campaign,metrics)


def test_large_comparison_parts_preserve_original_bytes_and_bound_omissions():
    from oeis_learn.generalization.progressive.public_comparisons import partition,assemble
    comparison=dict(id=digest('large-comparison'),rows=[],roster=[],seeds=[1,2],design={},verdict='INCOMPLETE',action='STOP_AFFECTED',binding={'original_bytes':'x'*(5<<20)})
    observations=[]
    stubs=partition([comparison],observations,maximum_bytes=16<<20)
    assert set(stubs[0])=={'id','parts'}
    missing=[]
    assert assemble(stubs,observations,missing)==[comparison] and not missing
    reversed_parts=dict(stubs[0],parts=list(reversed(stubs[0]['parts'])))
    with pytest.raises(ValueError,match='reordered'):
        assemble([reversed_parts],observations,[])
    omitted=partition([comparison],[],maximum_bytes=4<<20)
    missing=[]
    assert assemble(omitted,[],missing)==[]
    assert missing==['comparison_omitted:'+comparison['id']]


def test_large_finite_campaign_manifest_roundtrips_without_relaxing_table_limits(tmp_path):
    from oeis_learn.generalization.progressive.report import MAX_PART,MAX_MANIFEST
    request=fixture(tmp_path)
    campaign=request['campaign_record']
    # Finite domains add many legitimate references to a containing campaign.
    # Individual public table parts retain their independent small bound.
    template=campaign['payload']['references'][0]
    campaign['payload']['references'].extend(dict(template,name='finite-'+str(i)+'x'*900) for i in range(5000))
    campaign=seal(campaign)
    request=seal(dict(request,campaign=campaign['identity'],campaign_record=campaign))
    build(tmp_path,tmp_path/'large.zip',request)
    with zipfile.ZipFile(tmp_path/'large.zip') as archive:
        assert MAX_PART<archive.getinfo('manifest.json').file_size<MAX_MANIFEST
        assert all(i.file_size<=MAX_PART for i in archive.infolist() if i.filename.startswith('tables/'))
    assert verify(tmp_path/'large.zip')['status']=='VERIFIED'
