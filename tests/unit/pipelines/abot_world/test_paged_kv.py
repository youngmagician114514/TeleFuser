"""Physical KV accounting, logical windows, reuse and ownership handoff."""

import pytest
import torch

from telefuser.pipelines.abot_world.paged_kv import KVPagePool, PagedSessionKV


def _state(pool):
    return PagedSessionKV(pool, torch.empty(0, 18, 2, 4), layers=2, frame_tokens=4)


def test_logical_window_growth_and_shrink_only_allocates_visible_physical_pages():
    page_bytes = 2 * 2 * 4 * 2 * 4 * 4
    pool = KVPagePool(max_cached_bytes=page_bytes)
    state = _state(pool)
    for chunk in range(12):
        window = 18 if chunk < 8 else 6
        state.truncate(window, window // 3)
        state.prepare(current_end=(chunk + 1) * 12, window=window, sink=window // 3, frames=3, context_length=4)
        for layer in range(2):
            value = torch.full((1, 12, 2, 4), float(chunk * 10 + layer))
            state.write_layer(layer, value, value + 1)
        assert len(state.table) == min((chunk + 1) * 3, window)
        assert len(set(state.table)) == len(state.table)
        # One text page, one physical page per retained frame, <= one cached page.
        assert pool.snapshot()["live_bytes"] == (len(state.table) + 1) * page_bytes
        assert pool.allocated_bytes <= (len(state.table) + 2) * page_bytes
    assert len(state.table) == 6
    # The sink remains the earliest frames; the new tail contains current KV.
    sink = pool.page(state.table[0])[0, 0]
    assert sink.eq(0).all()
    assert pool.page(state.table[-1])[0, 0].eq(110).all()
    state.close()
    pool.trim_free()
    assert pool.snapshot()["allocated_bytes"] == 0


def test_received_physical_pages_install_without_another_history_allocation():
    source_pool, target_pool = KVPagePool(), KVPagePool()
    source, target = _state(source_pool), _state(target_pool)
    source.prepare(current_end=12, window=18, sink=6, frames=3, context_length=4)
    for layer in range(2):
        source.write_layer(layer, torch.full((1, 12, 2, 4), float(layer)), torch.ones(1, 12, 2, 4))
    payload = source.export()
    payload["cursor"] = (12, 12)
    received_ids = []

    def receive(value):
        page_id = target_pool.acquire(tuple(value.shape), dtype=value.dtype, device=value.device)
        received_ids.append(page_id)
        target_pool.page(page_id).copy_(value)
        return target_pool.page(page_id)

    payload["paged_self"] = [receive(value) for value in payload["paged_self"]]
    payload["paged_cross"] = receive(payload["paged_cross"])
    before = target_pool.allocated_bytes
    target.install(payload, copy=False)
    assert target.table == received_ids[:-1]
    assert target.cross_page == received_ids[-1]
    assert target_pool.allocated_bytes == before
    torch.testing.assert_close(target.clean_delta(3), source.clean_delta(3))
    expected = target.clean_delta(3).clone()
    source.close()
    source.prepare(current_end=24, window=18, sink=6, frames=3, context_length=4)
    for page_id in source.table:
        source_pool.page(page_id).fill_(99)
    torch.testing.assert_close(target.clean_delta(3), expected)


def test_budget_includes_incoming_pages_and_reuses_free_pages_before_allocating():
    pool = KVPagePool(max_bytes=128, max_cached_bytes=128)
    first = pool.acquire((16,), dtype=torch.float32, device="cpu")
    second = pool.acquire((16,), dtype=torch.float32, device="cpu")
    with pytest.raises(RuntimeError, match="Physical KV page budget"):
        pool.acquire((1,), dtype=torch.float32, device="cpu")
    pool.release(first)
    assert pool.acquire((16,), dtype=torch.float32, device="cpu") == first
    assert pool.snapshot()["live_bytes"] == 128
    pool.release(first)
    pool.release(second)
    pool.trim_free()
    assert pool.allocated_bytes == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("cache_limit", (0, 1 << 30))
def test_recycled_page_orders_a_new_peer_stream_after_its_last_consumer(cache_limit):
    pool = KVPagePool(max_cached_bytes=cache_limit)
    reader, writer = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(reader):
        page_id = pool.acquire((4096,), dtype=torch.float32, device="cuda")
        pool.page(page_id).fill_(3)
        torch.cuda._sleep(100_000_000)
        observed = pool.page(page_id).sum()
        pool.release(page_id)
    if not reader.query():
        assert pool.allocated_bytes == 4096 * 4
    reusable = page_id in pool._pages
    with torch.cuda.stream(writer):
        reused = pool.acquire((4096,), dtype=torch.float32, device="cuda")
        assert (reused == page_id) == reusable
        pool.page(reused).fill_(99)
    reader.synchronize()
    writer.synchronize()
    assert observed.item() == 3 * 4096
