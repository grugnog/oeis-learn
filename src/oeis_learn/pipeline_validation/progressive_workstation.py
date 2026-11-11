"""Fixed workstation phases over canonical foundation/discovery operations.

No arbitrary command manifest, extra training authority, host ML dependencies,
or invented retained inputs. Research consumes an original concrete blueprint.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from .identity import confined, seal
from .progressive_workstation_state import Session, Blocked, write, read, telemetry

PHASES=('register-inputs','rehearse','prepare-research','run-research','assess','collect')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',choices=('podman','docker'),required=True)
    p.add_argument('--session',required=True)
    p.add_argument('--memory-gib',type=int,default=64)
    p.add_argument('--cpus',type=int,default=8)
    p.add_argument('--render-node')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('phase',choices=PHASES)
    p.add_argument('--reference-campaign')
    p.add_argument('--blueprint',help='original concrete research blueprint; never a template with guessed inputs')
    p.add_argument('--selection',help='original locked assessment selection')
    p.add_argument('--input-manifest', help='reviewed research input-resolution manifest for register-inputs')
    p.add_argument('--reuse-exposure-session', help='reuse reviewed complete original exposure data only')
    p.add_argument('--oeis-snapshot', help='explicit existing indexed-source/v1 manifest for register-inputs')
    return p


def command(root,args,operation,**arguments):
    argv=[sys.executable,'-S',str(Path(root)/'scripts/foundation_run.py'),'--engine',args.engine,
        '--memory-gib',str(args.memory_gib),'--cpus',str(args.cpus),'--host-free-memory-gib','24']
    if args.render_node: argv+=['--render-node',args.render_node]
    argv+=['discovery',operation]
    for key,value in arguments.items(): argv+=['--'+key.replace('_','-'),str(value)]
    return argv


class Workflow:
    def __init__(self,session,args):
        self.s=session; self.root=session.root; self.args=args
        self.base=session.folder/args.phase

    def run_command(self,name,operation,*,dependencies=(),seconds=3600,**arguments):
        from .supervision import launch
        def action():
            argv=command(self.root,self.args,operation,**arguments)
            logs=self.base/'logs'/name
            launch(argv,logs,seconds=seconds,stream_cap=4<<20,cleanup_seconds=45,
                   grace_seconds=40,allowed_returncodes=(0,1,2,3))
            process=json.loads((logs/'process.json').read_bytes())
            code=process['returncode']
            if code:
                exception=Blocked if code==2 else RuntimeError
                raise exception(f'{operation} exited {code}; original logs: {logs}')
            # Commands expose original result records, not wrapper PASS bits.
            target=arguments.get('output')
            result=confined(self.root,target,exists=False)/'result.json' if target else None
            files=[logs/'process.json']
            if result is not None and result.is_file(): files.append(result)
            return dict(operation=operation,output=target,exit_code=code),files
        return self.s.step(name,action,dependencies=dependencies)

    def host(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('discovery_foundation_host',self.root/'scripts/foundation_local.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        from types import SimpleNamespace
        return module.Host(SimpleNamespace(engine=self.args.engine,memory_gib=self.args.memory_gib,
            cpus=self.args.cpus,render_node=self.args.render_node,free_disk_gib=10,
            host_free_memory_gib=24,image=None,gpu_base_image=None))

    def images(self):
        from .supervision import launch
        for target in ('qualification','gpu'):
            def build(target=target):
                output=self.base/('image-'+target); output.mkdir(parents=True,exist_ok=True)
                code=('import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);'
                      'from oeis_learn.container_runtime import Engine;'
                      'from oeis_learn.pipeline_validation.progressive_workstation_state import write;'
                      'write(sys.argv[4],dict(image=Engine(sys.argv[2]).build(sys.argv[3],target=sys.argv[5],tag="oeis-learn:discovery-"+sys.argv[5])))')
                launch([sys.executable,'-S','-c',code,str(self.root/'src'),self.args.engine,str(self.root),
                        str(output/'image.json'),target],output/'logs',seconds=3600,stream_cap=4<<20)
                return json.loads((output/'image.json').read_bytes())['image'],[output/'image.json']
            self.s.step('image-'+target,build,dependencies=('rehearse/environment',))
        entries=self.s.journal['steps']
        return {t:entries.get('rehearse/image-'+t,{}).get('value')
                if entries.get('rehearse/image-'+t,{}).get('status')=='COMPLETE' else None
                for t in ('qualification','gpu')}

    def installed(self,name,image,body,*,gpu=False,seconds=600,dependencies=(),outputs=()):
        from .progressive_handoff import stage
        from .supervision import diagnostic_program
        def action():
            output=self.base/name
            stage(self.host(),self.root,image,output,['python','-c',diagnostic_program(body,name)],gpu=gpu,seconds=seconds)
            return str(output.relative_to(self.root)),[output/p for p in outputs]
        return self.s.step(name,action,dependencies=dependencies)

    def rehearse(self):
        ref=self.s.binding['reference_campaign']
        self.run_command('inventory','init',reference_campaign=ref,
            output=str((self.base/'inventory').relative_to(self.root)),seconds=120)
        def environment():
            host=self.host()
            value=dict(engine=host.engine.facts(),resources=host.policy,
                       measurements=telemetry(self.root),scope='DIAGNOSTIC_ONLY')
            write(self.base/'environment.json',value)
            return value,[self.base/'environment.json']
        self.s.step('environment',environment)
        built=self.images()
        # CPU contracts and fixture construction are independent of GPU build/Q.
        # Retain these results even when the workstation cannot start HIP.
        if built['qualification'] is not None:
            self.installed('assessment-contracts',built['qualification'],
                "import pytest; raise SystemExit(pytest.main(['-vv','-ra','--tb=long','-o','addopts=',"
                "'--basetemp','/work/output/private','--junitxml','/work/output/cases.xml',"
                "'tests/contract/test_progressive_assessment.py']))",
                seconds=300,outputs=('cases.xml',),dependencies=('rehearse/image-qualification',))
            self.installed('assets',built['qualification'],
                "from oeis_learn.pipeline_validation.progressive_workstation_fixture import assets; assets('/work/output')",
                outputs=('profile.json','recipe.json','initial.pt'),dependencies=('rehearse/image-qualification',))
        if any(v is None for v in built.values()):
            self.s.step('base-qualification',lambda:None,dependencies=('rehearse/image-qualification','rehearse/image-gpu'))
            return
        images=[built[t] for t in ('qualification','gpu')]
        def foundation():
            from .generalization_qualification_io import produce
            result=produce(self.host(),self.root,self.base/'base-qualification',images=images,
                           contract=self.s.binding['source'])
            if result['status']!='PASS': raise RuntimeError('base qualification failed; inspect original pipeline receipts')
            return str((self.base/'base-qualification').relative_to(self.root)),[self.base/'base-qualification/qualification.json']
        self.s.step('base-qualification',foundation,dependencies=('rehearse/image-qualification','rehearse/image-gpu'))
        body=("import os;from oeis_learn.pipeline_validation.progressive_qualification import matrix,hardware_facts;"
              "from oeis_learn.pipeline_validation.progressive_workstation_fixture import profile;"
              "from oeis_learn.experiments.source_snapshot import source_snapshot;"
              "from oeis_learn.generalization.progressive.contracts import atomic_write,digest;"
              "atomic_write('/work/output/matrix.json',matrix([profile()],source=digest(source_snapshot()),"
              "image=os.environ['OEIS_QUALIFICATION_IMAGE'],engine=os.environ['OEIS_QUALIFICATION_ENGINE'],hardware=hardware_facts()))")
        self.installed('matrix',images[1],body,gpu=True,seconds=120,outputs=('matrix.json',),
                       dependencies=('rehearse/base-qualification','rehearse/assets'))
        def qualify_plan():
            from .progressive_workstation_fixture import qualification_inputs
            path=qualification_inputs(self.root,self.base/'plans',base=self.base/'base-qualification',matrix=self.base/'matrix/matrix.json')
            return str(path.relative_to(self.root)),[path]
        b=self.s.step('qualification-plan',qualify_plan,dependencies=('rehearse/matrix',))
        self.run_command('qualify','qualify',blueprint=b or 'missing',output=str((self.base/'qualification').relative_to(self.root)),
            dependencies=('rehearse/qualification-plan',),seconds=1900)
        def prepare_plan():
            from .progressive_workstation_fixture import preparation_inputs
            path=preparation_inputs(self.root,self.base/'plans',qualification=self.base/'qualification',assets_root=self.base/'assets')
            return str(path.relative_to(self.root)),[path]
        b=self.s.step('preparation-plan',prepare_plan,dependencies=('rehearse/qualify','rehearse/assets'))
        self.run_command('prepare','prepare',blueprint=b or 'missing',qualification=str((self.base/'qualification').relative_to(self.root)),
            output=str((self.base/'prepared').relative_to(self.root)),dependencies=('rehearse/preparation-plan',),seconds=300)
        self.run_command('freeze','freeze',blueprint=b or 'missing',qualification=str((self.base/'qualification').relative_to(self.root)),
            prepared=str((self.base/'prepared').relative_to(self.root)),output=str((self.base/'frozen').relative_to(self.root)),
            dependencies=('rehearse/prepare',),seconds=300)
        self.run_command('acceptance','acceptance',campaign=str((self.base/'frozen/campaign.json').relative_to(self.root)),
            output=str((self.base/'acceptance').relative_to(self.root)),dependencies=('rehearse/freeze',),seconds=3700)

    def prepare_research(self):
        """Missing workstation inputs block only research, never fixture acceptance."""
        from oeis_learn.generalization.progressive.contracts import load,reference,verify_refs,validate
        def select():
            path=self.args.blueprint
            if path is None:
                inventory_path=self.s.folder/'rehearse/inventory/inputs.json'
                inventory=load(inventory_path) if inventory_path.is_file() else {'references':[]}
                refs=[r['artifact'] for r in inventory['references'] if r['name']=='research_blueprint']
                if len(refs)==1:
                    verify_refs(self.root,refs); path=refs[0]['path']
            if path is None:
                deficits=dict(required=['original research_blueprint with prospective preparation, qualification and calibration plans',
                    'retained recipe/checkpoint and full exposure inventory','real cohort/source/support inputs and finite task reservations'],
                    reason='RESEARCH_INPUTS_REQUIRED',hardware_results=str(self.s.folder/'share/rehearse.zip'))
                write(self.base/'deficits.json',deficits)
                raise Blocked('Research blueprint is not supplied by retained producers; see deficits.json. No research settings or truth were invented.')
            b=validate('blueprint',load(confined(self.root,path)))
            if b['purpose']!='research': raise ValueError('research purpose required')
            write(self.base/'selection.json',seal(dict(blueprint=reference(self.root,path))))
            return path,[confined(self.root,path),self.base/'selection.json']
        b=self.s.step('select-inputs',select,dependencies=('rehearse/acceptance',))
        q=str((self.base/'qualification').relative_to(self.root))
        self.run_command('qualify','qualify',blueprint=b or 'missing',output=q,dependencies=('prepare-research/select-inputs',),seconds=3700)
        self.run_command('prepare','prepare',blueprint=b or 'missing',qualification=q,
            output=str((self.base/'prepared').relative_to(self.root)),dependencies=('prepare-research/qualify',),seconds=86500)
        self.run_command('freeze','freeze',blueprint=b or 'missing',qualification=q,
            prepared=str((self.base/'prepared').relative_to(self.root)),output=str((self.base/'frozen').relative_to(self.root)),
            dependencies=('prepare-research/prepare',),seconds=600)
        self.run_command('acceptance','acceptance',campaign=str((self.base/'frozen/campaign.json').relative_to(self.root)),
            output=str((self.base/'acceptance').relative_to(self.root)),dependencies=('prepare-research/freeze',),seconds=3700)

    def register_research(self):
        """Local original-data gate before GPU qualification or scientific work.

        The manifest is an audit input, not an execution blueprint. Keep this
        distinct from the existing explicit-blueprint preparation route.
        """
        from .progressive_research_inputs import audit, validate_manifest, checked_file, source_candidates
        from oeis_learn.generalization.progressive.contracts import load, reference
        from .progressive_research_design import construct
        manifest = validate_manifest(load(confined(self.root, self.args.input_manifest)))
        def inventory():
            value = source_candidates(self.root)
            write(self.base / 'source-candidates.json', value)
            return value, [self.base / 'source-candidates.json']
        candidates = self.s.step('source-inventory', inventory)
        previous = self.s.journal['steps'].get('register-inputs/project-source', {})
        if previous.get('status') == 'COMPLETE':
            selected = getattr(self.args, 'oeis_snapshot', None)
            if selected and reference(self.root, selected) != previous['value']['snapshot']:
                raise ValueError('selected original source changed; retain prior projection and use a new session')
        def source():
            from .progressive_research_source import project
            path = getattr(self.args, 'oeis_snapshot', None)
            if not path:
                write(self.base / 'deficits.json', dict(reason='ORIGINAL_SOURCE_SELECTION_REQUIRED',
                    required=['explicit immutable indexed-source/v1 source manifest for fresh OEIS transfer panels'],
                    candidates=candidates, research_ready=False, training_started=False))
                raise Blocked('Select an original source snapshot from source-candidates.json; no source was downloaded or inferred')
            result = project(self.root, path, self.base / 'source')
            if result['status'] != 'COMPLETE':
                raise Blocked('Original source projection has integrity deficits; see source-projection.json')
            originals = {ref['path']: confined(self.root, ref['path'])
                         for ref in [result['snapshot'], *result['originals']]}
            return result, [self.base / 'source/source-projection.json',
                            self.base / 'source/private/oeis-candidates.json', *originals.values()]
        self.s.step('project-source', source)
        registration_root = self.base / 'registration'
        def register():
            value = audit(self.root, manifest, registration_root)
            if value['status'] != 'COMPLETE':
                raise Blocked('Original research input verification has deficits; see registration.json')
            originals = {row['workspace_artifact']['path']: confined(self.root, row['workspace_artifact']['path'])
                         for row in value['checks'] if row.get('status') == 'PASS'}
            return value, [registration_root / 'registration.json',
                           registration_root / 'retained-inputs.json', *originals.values()]
        registration = self.s.step('register-originals', register)
        if registration is not None and registration['manifest'] != manifest['identity']:
            raise ValueError('registered original input manifest changed')
        def proposal():
            value = construct(self.root, self.base / 'design', manifest, registration)
            return str((self.base / 'design/blueprint.json').relative_to(self.root)), sorted((self.base / 'design').glob('*.json'))
        self.s.step('prospective-design', proposal, dependencies=('register-inputs/register-originals',))
        # CPU exposure assembly is independent of HIP and of fresh-data deficits.
        def image():
            from .supervision import launch
            output = self.base / 'image-qualification'
            code = ('import sys;sys.path.insert(0,sys.argv[1]);'
                    'from oeis_learn.container_runtime import Engine;'
                    'from oeis_learn.pipeline_validation.progressive_workstation_state import write;'
                    'write(sys.argv[4],dict(image=Engine(sys.argv[2]).build(sys.argv[3],target="qualification",tag="oeis-learn:discovery-qualification")))')
            launch([sys.executable, '-S', '-c', code, str(self.root / 'src'), self.args.engine,
                    str(self.root), str(output / 'image.json')], output / 'logs',
                   seconds=3650, stream_cap=4 << 20)
            value = json.loads((output / 'image.json').read_bytes())['image']
            return value, [output / 'image.json']
        reuse = getattr(self.args, 'reuse_exposure_session', None)
        selected_image = None if reuse else self.s.step('research-custodian-image', image,
            dependencies=('register-inputs/register-originals',))
        def exposure():
            if reuse:
                from .progressive_research_inputs import reuse_exposure
                return reuse_exposure(self.root, reuse, manifest, registration, self.base / 'exposure-reuse')
            import runpy
            from .supervision import diagnostic_program
            runner = runpy.run_path(str(self.root / 'scripts/qualify_bigint_phase5.py'))
            output = self.base / 'exposure'
            write(output / 'registration.json', registration)
            write(output / 'input-resolution.json', manifest)
            names = {'original.' + item['name'] for item in manifest['prior_inputs']['originals']}
            names.update({'prior.probes', 'prior.observed', 'fresh-002.targets', 'fresh-002.roster'})
            refs = {row['workspace_artifact']['path']: row['workspace_artifact']
                    for row in registration['checks'] if row.get('status') == 'PASS' and row['name'] in names}
            mounts = [(checked_file(self.root, ref), '/work/inputs/' + ref['path']) for ref in refs.values()]
            body = ("from oeis_learn.pipeline_validation.progressive_research_inputs import import_exposure;"
                    "from oeis_learn.generalization.progressive.contracts import load;"
                    "r=import_exposure('/work/inputs',load('/work/output/input-resolution.json'),"
                    "load('/work/output/registration.json'),'/work/output',seconds=7100);"
                    "raise SystemExit(0 if r['status']=='COMPLETE' else 2)")
            runner['container_stage'](self.host(), selected_image, output,
                ['python', '-c', diagnostic_program(body, 'research-exposure')], seconds=7200,
                readonly_inputs=mounts)
            return str(output.relative_to(self.root)), [output / 'exposure-import.json', output / 'private/exposure.sqlite']
        self.s.step('import-exposure', exposure,
            dependencies=('register-inputs/register-originals',) + (() if reuse else ('register-inputs/research-custodian-image',)))
        def report():
            value = seal(dict(version='progressive-research-registration-status/v1',
                status='INPUTS_REGISTERED', research_ready=False, executable=False,
                next_required=['fresh panel/support census and measured sensitivity',
                               'concrete task graph and exact-input calibration',
                               'current-source HIP qualification, freeze and campaign acceptance'],
                scope='Original input registration only; no research preparation or execution authority'))
            write(self.base / 'registration-status.json', value)
            return value, [self.base / 'registration-status.json']
        self.s.step('registration-status', report,
            dependencies=('register-inputs/register-originals', 'register-inputs/project-source',
                          'register-inputs/import-exposure', 'register-inputs/prospective-design'))

    def research(self):
        prepared=self.s.folder/'prepare-research'
        c=prepared/'frozen/campaign.json'
        deps=('prepare-research/acceptance',)
        from oeis_learn.generalization.progressive.contracts import load
        # Canonical run/resume owns all task caps and journal reconciliation.
        import re
        executions=sorted(p for p in (self.s.folder/'run-research').glob('execution-*')
                          if p.is_dir() and re.fullmatch(r'execution-\d{4}',p.name))
        if self.args.phase=='assess':
            executions=sorted(p for p in self.base.glob('execution-*')
                              if p.is_dir() and re.fullmatch(r'execution-\d{4}',p.name))
        if executions and (executions[-1]/'result.json').is_file():
            last=load(executions[-1]/'result.json')
            if last.get('mechanical_status')=='COMPLETE':
                print('Original campaign invocation is already complete: '+str(executions[-1]),flush=True)
                return
        output=self.base/('execution-'+str(len(executions)).zfill(4))
        def execute():
            if self.args.phase=='assess' and not self.args.selection:
                raise Blocked('--selection must identify the original locked final selection')
            operation='assess' if self.args.phase=='assess' else 'resume' if executions else 'run'
            args=dict(output=str(output.relative_to(self.root)))
            if operation=='resume': args['run']=str(executions[0].relative_to(self.root))
            else: args.update(campaign=str(c.relative_to(self.root)),acceptance=str((prepared/'acceptance').relative_to(self.root)))
            if operation=='assess': args['selection']=self.args.selection
            cap=load(c,maximum=32<<20)['payload']['caps']['elapsed_us']/1000000
            from .supervision import launch
            logs=self.base/'logs'/output.name
            launch(command(self.root,self.args,operation,**args),logs,seconds=cap+120,
                stream_cap=4<<20,cleanup_seconds=45,grace_seconds=40,allowed_returncodes=(0,1,2,3))
            result=load(output/'result.json')
            if result['mechanical_status'] not in {'COMPLETE','PARTIAL','PAUSED','STOPPED','BLOCKED'}:
                raise RuntimeError('research execution failed; original result retained')
            return dict(run=str(output.relative_to(self.root)),result=result),[output/'result.json']
        key='invocation-'+str(len(executions)).zfill(4)
        self.s.step(key,execute,dependencies=deps)
        if output.exists():
            archive=self.base/(output.name+'.zip')
            self.run_command(key+'-report','report',run=str(output.relative_to(self.root)),output=str(archive.relative_to(self.root)),seconds=600)
            if archive.is_file():
                self.run_command(key+'-verify','verify-bundle',archive=str(archive.relative_to(self.root)),
                    output=str((self.base/(output.name+'-verified')).relative_to(self.root)),seconds=600)


def main(root,argv=None):
    args=parser().parse_args(argv); root=Path(root).resolve()
    if args.phase == 'register-inputs' and not args.input_manifest:
        parser().error('register-inputs requires --input-manifest')
    if (args.input_manifest or args.oeis_snapshot or args.reuse_exposure_session) and args.phase != 'register-inputs':
        parser().error('input manifest, source snapshot and exposure reuse belong to register-inputs')
    if args.memory_gib<=0 or args.cpus<=0: raise ValueError('positive finite resource limits required')
    if args.dry_run:
        print(json.dumps(dict(phase=args.phase,engine=args.engine,session=args.session,
            source_policy='exact current closure; changes require requalification',
            sequence=dict(**{'register-inputs':['source-inventory','project-explicit-source','verify-originals','prospective-design','CPU-image','private-exposure-import','registration-status']},rehearse=['inventory','images','base-qualification','tiny-Q','prepare','freeze','CPU+HIP-acceptance'],
            **{'prepare-research':['original-inputs','selected-Q','exact-calibration','freeze','acceptance'],
               'run-research':['run-or-resume','report','verify'],'assess':['locked-assess','report','verify'],'collect':['package-existing']})[args.phase],
            launches=False,host_dependencies='Python standard library only'),indent=2))
        return 0
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from oeis_learn.generalization.progressive.contracts import digest
    folder=confined(root,args.session,exists=False)
    old=read(folder/'session.json')['binding'] if (folder/'session.json').is_file() else {}
    ref=args.reference_campaign or old.get('reference_campaign')
    if not ref and args.phase!='collect':
        print('--reference-campaign is required for the first invocation',file=sys.stderr); return 2
    binding=dict(source=digest(collect_snapshot(root)),engine=args.engine,memory_gib=args.memory_gib,
        cpus=args.cpus,render_node=args.render_node,reference_campaign=ref)
    if args.phase == 'register-inputs':
        from .progressive_workstation_state import sha
        binding['input_manifest'] = dict(path=args.input_manifest, sha256=sha(confined(root,args.input_manifest)))
        if args.reuse_exposure_session:
            binding['reuse_exposure_session'] = args.reuse_exposure_session
    if args.phase=='collect' and old: binding=old # Recovery collects old evidence; never executes it.
    s=Session(root,args.session,binding,args.phase)
    stop=threading.Event()
    def observations():
        while not stop.wait(30):
            try:
                value=telemetry(root)
                path=s.folder/'telemetry.jsonl'
                if not path.exists() or path.stat().st_size < 16<<20:
                    with path.open('a') as f: f.write(json.dumps(value,sort_keys=True)+'\n'); f.flush()
                print('Workstation heartbeat; original stage progress and logs: '+str(s.folder),flush=True)
            except OSError: pass
    def interrupted(*_): raise KeyboardInterrupt()
    previous={sig:signal.signal(sig,interrupted) for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    try:
        with s.own():
            write(s.folder/'environment.json',telemetry(root))
            thread=threading.Thread(target=observations,daemon=True); thread.start()
            w=Workflow(s,args)
            if args.phase=='register-inputs': w.register_research()
            elif args.phase=='rehearse': w.rehearse()
            elif args.phase=='prepare-research': w.prepare_research()
            elif args.phase in {'run-research','assess'}: w.research()
            else: return 0
            statuses={k:v['status'] for k,v in s.journal['steps'].items()}
            write(s.folder/'hardware-results.json',dict(
                source=binding['source'],engine=args.engine,scope='selected original receipts only',
                tiny_qualification=statuses.get('rehearse/qualify','PENDING'),
                tiny_acceptance=statuses.get('rehearse/acceptance','PENDING'),
                research_acceptance=statuses.get('prepare-research/acceptance','PENDING'),
                sustained_status='PENDING_ACTUAL_PROTOCOL_RECONCILIATION',
                scientific_status='consult original campaign report and independent verification',
                tasks=dict(T082='pending until original selected HIP producers pass',
                           T088='tiny scope first; research scopes separately qualified',
                           T089='research freeze requires original retained inputs and calibration',
                           T090='consult campaign-bound acceptance',T091='actual sustained protocol evidence required',
                           T111='phase dispositions retained; no unexecuted research marked complete')))
            return s.code()
    except KeyboardInterrupt: return 130
    except (Blocked,FileNotFoundError) as exc: print(str(exc),file=sys.stderr); return 2
    except Exception as exc: print(str(exc),file=sys.stderr); return 1
    finally:
        stop.set()
        for sig,handler in previous.items(): signal.signal(sig,handler)
