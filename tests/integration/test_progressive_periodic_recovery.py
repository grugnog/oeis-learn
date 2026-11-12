"""Real optimizer tests for the shared hourly policy and progressive accounting."""
import copy
import pytest
import torch
from oeis_learn.generalization.progressive.contracts import digest
from oeis_learn.generalization.progressive.periodic_recovery import PeriodicRecovery
from oeis_learn.generalization.progressive.budgets import BudgetJournal
from oeis_learn.pipeline_validation.progressive_training import RecoverableUpdate
from tests.helpers.progressive_fixture import caps, task
from tests.unit.test_progressive_training_policy import batch, policy, profile


def setup(root):
    p = profile(0.2); cfg = policy(p, microbatch_size=7)
    t = task(caps=caps(tokens=10000, elapsed_us=60000000))
    j = BudgetJournal(root, t['payload']['campaign'], t['payload']['caps'])
    context = {k:{} for k in ('scheduler','data','replay','history','pins','selection')}
    context['data'] = dict(cursor=0, pending_batch=None)
    cuts = {k:digest(None) for k in ('stream','allocation','archive','library','cost_journal','selection','milestones','consumed')}
    return p,cfg,t,j,context,cuts


def advance(worker, x, tokens, cuts, update):
    return worker.step(x,tokens,entries=['target-'+str(update)]*16,update=update,
        context=worker.context,cuts=cuts,boot='boot',monotonic_us=worker.recovery.clock['start']+update)


def tensors(worker):
    return [v.detach().clone() for v in [*worker.runner.encoder.parameters(),*worker.runner.decoder.parameters()]]


@pytest.mark.parametrize('failure', ['none','discard','publication','optimizer'])
def test_hourly_window_preserves_exact_state_and_never_refunds(tmp_path, monkeypatch, failure):
    p,cfg,t,j,context,cuts = setup(tmp_path)
    x,tokens,pad = batch(); charge = int(tokens[:,1:].ne(pad).sum())
    with PeriodicRecovery(j,t,digest([p,cfg])) as r:
        w=RecoverableUpdate(r,p,cfg,device='cpu'); w.initialize(context,cuts)
        initial=tensors(w)
        r.begin(boot='boot',monotonic_us=0); r.activate()
        # Ordinary updates must never capture Adam/model tensors or read a checkpoint.
        with monkeypatch.context() as m:
            m.setattr(r.store,'load',lambda *_:pytest.fail('hot-path full read'))
            m.setattr(w,'capture',lambda *_:pytest.fail('hot-path full capture'))
            advance(w,x,tokens,cuts,1); advance(w,x,tokens,cuts,2)
        assert r.current()['ledger'].updates==2
        assert r.current()['ledger'].consumed==charge*2
        live=tensors(w); live_rng=torch.get_rng_state()
        assert any(not torch.equal(a,b) for a,b in zip(initial,live))
        if failure=='publication':
            from oeis_learn.generalization.progressive.recovery import Recovery
            original=Recovery.append
            def failed(self, **kw):
                if 'checkpoint' in kw:raise InterruptedError('publication lost')
                return original(self,**kw)
            with monkeypatch.context() as m:
                m.setattr(Recovery,'append',failed)
                with pytest.raises(InterruptedError):r.flush(lambda:w.capture(w.context))
            r.finish_window(None,failed=True)
        elif failure=='optimizer':
            with monkeypatch.context() as m:
                def failed(*args,**kw):
                    with torch.no_grad():next(w.runner.encoder.parameters()).add_(123)
                    raise RuntimeError('partial optimizer')
                m.setattr(w.runner,'update',failed)
                with pytest.raises(RuntimeError):advance(w,x,tokens,cuts,3)
            assert w.failed
            r.finish_window(None,failed=True)
        else:
            r.finish_window(lambda:w.capture(w.context),failed=failure=='discard')
        r.close(boot='boot',monotonic_us=100)
    with PeriodicRecovery(j,t,digest([p,cfg])) as r:
        r.begin(boot='boot',monotonic_us=1000)
        restored=r.restore(); w=RecoverableUpdate(r,p,cfg,device='cpu')
        saved = failure=='none'
        assert restored['ledger'].consumed == charge*(3 if failure=='optimizer' else 2)
        assert restored['ledger'].updates == (2 if saved else 0)
        assert restored['discarded_tokens'] == (0 if saved else restored['ledger'].consumed)
        assert all(torch.equal(a,b) for a,b in zip(tensors(w),live if saved else initial))
        if saved:
            assert torch.equal(torch.get_rng_state(),live_rng)
            assert w.context['history']['consumed_counts']=={'target-1':16,'target-2':16}
        r.activate()
        advance(w,x,tokens,cuts,3 if saved else 1)
        r.finish_window(lambda:w.capture(w.context)); r.close(boot='boot',monotonic_us=1100)
        assert r.restore()['ledger'].consumed==charge*(4 if failure=='optimizer' else 3)


def test_hour_boundary_publishes_union_of_window_targets(tmp_path,monkeypatch):
    from oeis_learn.generalization.progressive.quarantine import Journal
    p,cfg,t,j,context,cuts=setup(tmp_path); x,tokens,_=batch()
    q=Journal(j.root)
    q.observe([dict(id='target-'+str(i),kind='target',parents=[]) for i in (1,2)])
    with PeriodicRecovery(j,t,digest([p,cfg])) as r:
        w=RecoverableUpdate(r,p,cfg,device='cpu'); w.initialize(context,cuts)
        r.begin(boot='boot',monotonic_us=0); r.activate()
        first=r.state['checkpoint']
        advance(w,x,tokens,cuts,1)
        assert r.state['checkpoint']==first
        r.last_full-=3601
        advance(w,x,tokens,cuts,2)
        assert r.state['checkpoint']!=first
        graph=q.read()[0]; node=next(n for n in graph if set(n['parents'])=={'target-1','target-2'})
        assert node['kind']=='consumed_update'
        r.finish_window(lambda:w.capture(w.context)); r.close(boot='boot',monotonic_us=100)
        assert r.restore()['ledger'].updates==2
