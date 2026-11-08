"""Pure pin and update-commit decisions; no I/O or premature latest publication."""
from .budgets import reserve_storage


def pin_plan(history,*,milestones,best,latest,selected,through=None,milestone_pins=None):
    if not history or latest not in history or best not in history or selected is not None and selected not in history:
        raise ValueError('missing named checkpoint')
    if list(milestones)!=sorted(set(milestones)): raise ValueError('milestones must increase')
    through=max(history.values()) if through is None else through
    pins={best,latest}
    if selected is not None: pins.add(selected)
    for milestone in milestones:
        if milestone>through: continue
        # Milestones are committed at the last complete update at/below exposure;
        # the adapter supplies its frozen milestone mapping when a batch straddles.
        if milestone_pins and milestone in milestone_pins:
            key = milestone_pins[milestone]
            if key not in history or history[key] > milestone:
                raise ValueError('invalid actual-exposure milestone pin')
            pins.add(key)
            continue
        matches=[key for key,tokens in history.items() if tokens==milestone]
        if not matches: raise ValueError('required milestone missing')
        pins.add(min(matches))
    return tuple(sorted(pins))


def milestone_decision(consumed,next_tokens,milestones):
    """Signal snapshot before crossing a cap; record actual exposure, never fabricate it."""
    if any(type(v) is not int or v<0 for v in (consumed,next_tokens)): raise ValueError('token boundary')
    return tuple(m for m in milestones if consumed<=m<consumed+next_tokens)


def commit_plan(previous,new,*,complete_update,verified_payloads,available_bytes,snapshot_bytes,required_pins):
    if not complete_update or not verified_payloads or new in previous: raise ValueError('checkpoint is not atomically publishable')
    reserve_storage(snapshot_bytes,required_pins,available_bytes)
    return dict(write_generation=new,verify_before_pointer=True,publish_latest=new,prune_after_pointer=True)


def prune_plan(history,pins):
    if not set(pins)<=set(history): raise ValueError('missing required pin')
    return tuple(sorted(set(history)-set(pins)))


PAC_COMPONENTS=frozenset({'weights','optimizer','scheduler','rng','data','adaptive','review','probes','transaction','ledger'})


def pac_generation(components,*,root,generation,model_update,contract,inventory,source,environment):
    """Build an atomic publication unit only after every component verifies.

    Installed C writes/synchronizes these payloads before publishing this manifest
    and the latest pointer. This pure boundary never publishes a partial generation.
    """
    from .contracts import pac_record,verify_artifact
    if set(components)!=PAC_COMPONENTS:raise ValueError('complete generation components required; weights-only is not resumable')
    if len({r['path'] for r in components.values()})!=len(components):raise ValueError('distinct generation component paths required')
    for ref in components.values():verify_artifact(root,ref)
    return pac_record('checkpoint',dict(version='pac-checkpoint/v1',kind='full_state',contract=contract,inventory=inventory,
                      policy='pac-exp3s-leitner/v1',source=source,environment=environment,generation=generation,
                      model_update=model_update,components=components),sealed=False).record()


def verify_pac_generation(record,*,root,contract,inventory,source,environment):
    from .contracts import pac_record
    record=pac_record('checkpoint',record).record()
    if any(record[k]!=v for k,v in dict(contract=contract,inventory=inventory,source=source,environment=environment).items()):
        raise ValueError('complete generation identity mismatch')
    actual=pac_generation(record['components'],root=root,generation=record['generation'],model_update=record['model_update'],
                          contract=contract,inventory=inventory,source=source,environment=environment)
    if actual!=record:raise ValueError('generation manifest mismatch')
    return actual
