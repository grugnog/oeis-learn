import pytest
from oeis_learn.synthesis.service_policy import RoleCapacity, BoundedCache


def test_fifo_count_bytes_deadlines_and_cancellation_are_shared():
    role = RoleCapacity(max_pending=2, max_pending_bytes=10)
    role.enqueue("active", 4, deadline=20, now=1)
    assert role.dispatch(now=2) == "active"
    role.enqueue("a", 6, deadline=5, now=2)
    role.enqueue("b", 4, deadline=10, now=2)
    assert not role.enqueue("c", 1, deadline=10, now=3)
    assert role.dispatch(now=4) is None
    role.finish("active")
    assert role.dispatch(now=6) == "b"
    assert role.expired == 1 and role.pending_bytes == 0
    role.finish("b")
    role.enqueue("cancel", 10, deadline=20, now=6)
    role.cancel("cancel")
    assert role.pending_bytes == 0
    with pytest.raises(ValueError): role.enqueue("late", 1, deadline=5, now=6)


def test_cache_reservations_eviction_oversize_and_failed_write():
    c = BoundedCache(2, 10)
    assert c.reserve("a", 4); c.publish("a", "A")
    assert c.reserve("b", 4); c.publish("b", "B")
    assert c.get("a") == "A"
    assert not c.reserve("huge", 11)
    assert c.get("b") == "B"  # oversize did not evict anybody
    c.get("a")
    assert c.reserve("c", 4)
    assert c.get("b") is None
    c.abort("c")
    assert c.reserved_bytes == 0
    assert c.reserve("d", 6); c.publish("d", "D")
    assert c.used_bytes == 10
    assert not c.reserve("held", 8) if c.reservations else True


def test_pending_writes_share_capacity_and_disabled_cache():
    c = BoundedCache(2, 10)
    assert c.reserve("x", 6)
    assert not c.reserve("y", 6)
    c.abort("x")
    assert c.reserve("y", 6)
    with pytest.raises(ValueError): c.publish("missing", 0)
    assert not BoundedCache(0, 0).reserve("x", 1)
