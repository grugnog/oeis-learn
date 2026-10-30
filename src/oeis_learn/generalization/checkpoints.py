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
