# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Background-thread LMCache lookups (OFFLOAD_LOOKUP_THREAD).

The fake client models the two properties the wrapper must respect: a lookup
takes time (latency injection through a gate), and every answered lookup pins
chunks on the workers under its ID until that ID reaches a metadata's unpin
list. A second lookup under an ID whose pin was not released would orphan the
first pin (LMCache keeps one pin record per ID), which the fake reports.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.async_lookup import (
    UNCONSUMED_HOLD_FRACTION,
    ThreadedLookupClient,
)
from atom.kv_transfer.offload.dense.connector import DenseOffloadScheduler
from atom.model_engine.sequence import SequenceStatus


class _FakeLookupClient:
    """LMCacheLookupClient's surface, with latency and worker-side pins."""

    def __init__(self, hit=16, *, gated=False, error=None):
        self.hit = hit
        self.error = error
        self.calls: list[str] = []
        self.threads: list[str] = []
        self.pins: set[str] = set()
        self.double_pins: list[str] = []
        self.cleared: list[str] = []
        self.gate = threading.Event()
        if not gated:
            self.gate.set()
        self.started = threading.Event()
        self.closed = False

    def lookup(self, token_ids, lookup_id, request_configs=None):
        self.started.set()
        assert self.gate.wait(5.0), "test gate never opened"
        self.calls.append(lookup_id)
        self.threads.append(threading.current_thread().name)
        if lookup_id in self.pins:
            self.double_pins.append(lookup_id)
        self.pins.add(lookup_id)
        if self.error is not None:
            raise self.error
        return self.hit

    def clear_lookup_status(self, lookup_id):
        self.cleared.append(lookup_id)

    def unpin(self, ids):
        for sid in ids:
            self.pins.discard(sid)

    def close(self):
        self.closed = True


def _config():
    return SimpleNamespace(
        kv_transfer_config={"kv_role": "offload"},
        kv_cache_block_size=4,
        decode_context_parallel_size=2,
        tensor_parallel_size=1,
    )


def _scheduler(monkeypatch, fake, *, flag=True, pin_timeout_sec=300):
    """A dense offload scheduler built the way the engine builds it."""
    from lmcache.v1.lookup_client import factory

    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _config=None: SimpleNamespace(
            chunk_size=8, pin_timeout_sec=pin_timeout_sec
        ),
    )
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_metadata",
        lambda *_a: SimpleNamespace(engine_id="test"),
    )
    monkeypatch.setattr(
        factory.LookupClientFactory,
        "create_lookup_client",
        staticmethod(lambda *_a, **_k: fake),
    )
    if flag:
        monkeypatch.setenv("OFFLOAD_LOOKUP_THREAD", "1")
    else:
        monkeypatch.delenv("OFFLOAD_LOOKUP_THREAD", raising=False)
    sched = DenseOffloadScheduler(_config())
    sched._min_load_tokens = 0
    return sched


def _seq(req_id, n=24, *, cached=0):
    return SimpleNamespace(
        id=req_id,
        num_cached_tokens=cached,
        num_prompt_tokens=n,
        num_tokens=n,
        token_ids=list(range(n)),
        block_table=[],
        status=SequenceStatus.WAITING,
    )


def _wait_done(client, sid, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not client._jobs[sid].future.done():
        assert time.monotonic() < deadline
        time.sleep(0.001)


def _dispatch(sched, fake):
    meta = sched.build_connector_meta()
    fake.unpin(meta.lookup_requests_in_step)
    return meta


@pytest.fixture
def closing():
    clients = []
    yield clients.append
    for client in clients:
        client.close()


def test_flag_off_keeps_the_synchronous_client(monkeypatch):
    fake = _FakeLookupClient()
    sched = _scheduler(monkeypatch, fake, flag=False)
    assert sched._lookup_client is fake
    seq = _seq(1)

    sched.prefetch_lookups([seq])  # the plain client has no `submit`
    assert fake.calls == []
    assert sched.get_num_new_matched_tokens(seq) == (16, True)
    assert fake.calls == ["1"]
    assert fake.threads == [threading.current_thread().name]
    assert not sched._lookup_client_has_orphans()


def test_prefetched_answer_is_consumed_without_a_second_round_trip(
    monkeypatch, closing
):
    fake = _FakeLookupClient()
    sched = _scheduler(monkeypatch, fake)
    client = sched._lookup_client
    closing(client)
    assert isinstance(client, ThreadedLookupClient)
    seq = _seq(2)

    sched.prefetch_lookups([seq])
    _wait_done(client, "2")
    sched.prefetch_lookups([seq])  # already in flight: not sent again
    assert sched.get_num_new_matched_tokens(seq) == (16, True)
    # Asked again on the next step (allocation failed): the memo answers.
    assert sched.get_num_new_matched_tokens(seq) == (16, True)

    assert fake.calls == ["2"]
    assert fake.threads == ["atom-lmcache-lookup"]
    assert "2" not in client.pending_ids()
    # The consumed lookup's pin is the scheduler's, exactly as a sync one's.
    assert sched._lookup_results["2"] == (seq, 16)
    assert "2" in sched._lookup_in_step
    counters = client.pop_counters()
    assert counters["prefetch_ready"] == 1
    assert counters.get("sync_lookup", 0) == 0


def test_unready_prefetch_is_waited_for_not_sent_again(monkeypatch, closing):
    fake = _FakeLookupClient(gated=True)
    sched = _scheduler(monkeypatch, fake)
    client = sched._lookup_client
    closing(client)
    seq = _seq(3)

    sched.prefetch_lookups([seq])
    assert fake.started.wait(5.0)
    # No pending state: admission waits for the in-flight answer.
    assert not sched.lookup_pending(seq)
    threading.Timer(0.05, fake.gate.set).start()
    t0 = time.perf_counter()
    assert sched.get_num_new_matched_tokens(seq) == (16, True)
    assert time.perf_counter() - t0 >= 0.04
    assert fake.calls == ["3"]
    assert client.pop_counters()["prefetch_waited"] == 1


def test_no_prefetch_falls_back_to_a_lookup_ahead_of_queued_prefetches(
    monkeypatch, closing
):
    fake = _FakeLookupClient(gated=True)
    sched = _scheduler(monkeypatch, fake)
    client = sched._lookup_client
    closing(client)
    queued = [_seq(10), _seq(11), _seq(12)]
    sched.prefetch_lookups(queued)
    assert fake.started.wait(5.0)  # "10" is running, "11"/"12" queued
    admitted = _seq(13)
    threading.Timer(0.02, fake.gate.set).start()

    assert sched.get_num_new_matched_tokens(admitted) == (16, True)
    assert fake.calls[:2] == ["10", "13"]
    assert client.pop_counters()["sync_lookup"] == 1
    for sid in ("11", "12"):
        _wait_done(client, sid)
    assert fake.calls == ["10", "13", "11", "12"]


def test_request_aborted_while_its_lookup_runs_releases_the_pin_after_it_answers(
    monkeypatch, closing
):
    fake = _FakeLookupClient(gated=True)
    sched = _scheduler(monkeypatch, fake)
    client = sched._lookup_client
    closing(client)
    seq = _seq(4)
    sched.prefetch_lookups([seq])
    assert fake.started.wait(5.0)

    sched.request_finished(seq)
    # Not answered yet: the servers may not have pinned, so no unpin is sent,
    # but the engine keeps stepping until it can be.
    assert sched.has_pending_work()
    assert _dispatch(sched, fake).lookup_requests_in_step == []

    fake.gate.set()
    _wait_done(client, "4")
    assert fake.pins == {"4"}
    assert _dispatch(sched, fake).lookup_requests_in_step == ["4"]
    assert fake.pins == set()
    assert not sched.has_pending_work()
    assert _dispatch(sched, fake).lookup_requests_in_step == []
    assert "4" not in client.pending_ids()


def test_request_aborted_while_its_lookup_is_queued_is_released_too(
    monkeypatch, closing
):
    fake = _FakeLookupClient(gated=True)
    sched = _scheduler(monkeypatch, fake)
    client = sched._lookup_client
    closing(client)
    running, queued = _seq(5), _seq(6)
    sched.prefetch_lookups([running, queued])
    assert fake.started.wait(5.0)

    sched.request_finished(queued)
    fake.gate.set()
    for sid in ("5", "6"):
        _wait_done(client, sid)
    # "6" still ran (the client cannot tell an abort from a detach) and its
    # pin goes back; "5" is still waiting for admission and keeps its own.
    assert fake.calls == ["5", "6"]
    assert _dispatch(sched, fake).lookup_requests_in_step == ["6"]
    assert fake.pins == {"5"}
    assert "6" not in client.pending_ids()
    assert "5" in client.pending_ids()


def test_aborted_waiting_request_releases_a_prefetched_pin(
    monkeypatch, closing, scheduler, seq_factory
):
    """Through the engine scheduler: prefetch at arrival, then abort."""

    fake = _FakeLookupClient()
    connector = _scheduler(monkeypatch, fake)
    closing(connector._lookup_client)
    scheduler.kv_connector = connector
    seq = seq_factory(list(range(24)))
    scheduler.add(seq)
    monkeypatch.setattr(scheduler.block_manager, "can_allocate", lambda *_a, **_k: -1)
    sid = str(seq.id)

    scheduler.schedule()
    # The prefetch was consumed by admission (blocked by allocation).
    assert fake.calls == [sid]
    assert fake.threads == ["atom-lmcache-lookup"]
    assert sid in connector._load_specs

    seq.status = SequenceStatus.ABORTED
    batch, _ = scheduler.schedule()
    meta = batch.connector_meta_output
    assert meta.lookup_requests_in_step == [sid]
    fake.unpin(meta.lookup_requests_in_step)
    assert fake.pins == set()
    assert fake.double_pins == []


def test_hbm_frontier_moved_after_the_prefetch_is_read_at_admission(
    monkeypatch, closing
):
    fake = _FakeLookupClient(hit=16)
    sched = _scheduler(monkeypatch, fake)
    closing(sched._lookup_client)
    seq = _seq(7)
    sched.prefetch_lookups([seq])
    _wait_done(sched._lookup_client, "7")
    # The HBM prefix cache now covers what the tier answered.
    seq.num_cached_tokens = 16
    assert sched.get_num_new_matched_tokens(seq) == (0, False)
    assert "7" not in sched._load_specs
    assert _dispatch(sched, fake).lookup_requests_in_step == ["7"]
    assert fake.pins == set()


def test_tier_growth_after_the_prefetch_only_shortens_the_load(monkeypatch, closing):
    fake = _FakeLookupClient(hit=8)
    sched = _scheduler(monkeypatch, fake)
    closing(sched._lookup_client)
    seq = _seq(8)
    sched.prefetch_lookups([seq])
    _wait_done(sched._lookup_client, "8")
    fake.hit = 16  # stored after the prefetch answered (and pinned) 8 tokens
    assert sched.get_num_new_matched_tokens(seq) == (8, True)
    assert sched._load_specs["8"].lmcache_cached_tokens == 8
    assert fake.calls == ["8"]


def test_changed_prompt_never_takes_a_second_pin(closing):
    fake = _FakeLookupClient()
    client = ThreadedLookupClient(fake)
    closing(client)
    client.submit(list(range(24)), "9")
    _wait_done(client, "9")

    assert client.lookup(list(range(32)), "9") is None
    assert client.pop_unpin_ids() == ["9"]
    fake.unpin(["9"])
    assert client.lookup(list(range(32)), "9") == 16
    assert fake.calls == ["9", "9"]
    assert fake.double_pins == []


def test_failed_background_lookup_is_unpinned_and_retried_later(monkeypatch, closing):
    fake = _FakeLookupClient(error=RuntimeError("rank 3 timed out"))
    sched = _scheduler(monkeypatch, fake)
    closing(sched._lookup_client)
    seq = _seq(14)
    sched.prefetch_lookups([seq])
    _wait_done(sched._lookup_client, "14")

    assert sched.get_num_new_matched_tokens(seq) == (0, False)
    # As on the synchronous path: a half-run lookup is unpinned.
    assert _dispatch(sched, fake).lookup_requests_in_step == ["14"]
    assert fake.pins == set()


def test_detached_lookup_reports_its_hit_as_a_save_floor(monkeypatch, closing):
    fake = _FakeLookupClient(hit=8, gated=True)
    sched = _scheduler(monkeypatch, fake)
    client = sched._lookup_client
    closing(client)
    sched._min_load_tokens = 16
    seq = _seq(15, cached=16)
    sched.update_state_after_alloc(seq)

    assert sched.can_detach_lookup(seq, 16)
    assert sched.detach_lookup(seq, 16)
    assert sched._hit_save_floors["15"] == 16
    fake.gate.set()
    _wait_done(client, "15")
    meta = _dispatch(sched, fake)
    assert meta.lookup_requests_in_step == ["15"]
    # The tier held less than HBM: the floor drops to what it holds.
    assert sched._hit_save_floors["15"] == 8
    assert fake.pins == set()


def test_hold_follows_lmcache_pin_timeout(monkeypatch, closing):
    sched = _scheduler(monkeypatch, _FakeLookupClient(), pin_timeout_sec=60)
    closing(sched._lookup_client)
    assert sched._lookup_client._max_hold_s == 60 * UNCONSUMED_HOLD_FRACTION


def _age_answers(client, seconds):
    """Backdate every answered job as if it had waited `seconds` longer."""
    for job in client._jobs.values():
        if job.answered is not None:
            job.answered -= seconds


def test_answer_left_unconsumed_past_the_hold_is_released_then_reasked(
    monkeypatch, closing
):
    fake = _FakeLookupClient()
    sched = _scheduler(monkeypatch, fake, pin_timeout_sec=100)
    client = sched._lookup_client
    closing(client)
    seq = _seq(20)

    sched.prefetch_lookups([seq])
    _wait_done(client, "20")
    assert fake.pins == {"20"}
    # Still inside the hold: nothing is released.
    _dispatch(sched, fake)
    assert fake.pins == {"20"}
    assert "20" in client.pending_ids()

    _age_answers(client, 100 * UNCONSUMED_HOLD_FRACTION)
    meta = _dispatch(sched, fake)
    assert meta.lookup_requests_in_step == ["20"]
    assert fake.pins == set()
    # The request is still waiting: its hit is not a detached save floor.
    assert client.pop_discarded_hits() == {}
    assert sched._perf_counters["async_lookup_stale_released"] == 1

    # The release is out, so the next prefetch asks again under the same ID
    # without orphaning a pin, and admission consumes the fresh answer.
    sched.prefetch_lookups([seq])
    _wait_done(client, "20")
    assert sched.get_num_new_matched_tokens(seq) == (16, True)
    assert fake.calls == ["20", "20"]
    assert fake.double_pins == []


def test_stale_answer_admitted_before_its_release_is_adopted(monkeypatch, closing):
    fake = _FakeLookupClient()
    sched = _scheduler(monkeypatch, fake, pin_timeout_sec=100)
    client = sched._lookup_client
    closing(client)
    seq = _seq(21)

    sched.prefetch_lookups([seq])
    _wait_done(client, "21")
    _age_answers(client, 100 * UNCONSUMED_HOLD_FRACTION)
    client.pump()  # released, not yet in a metadata
    assert sched._lookup_client_has_orphans()

    assert sched.get_num_new_matched_tokens(seq) == (16, True)
    # Adopted: no second round trip, and the pin is the scheduler's again.
    assert fake.calls == ["21"]
    assert client.pop_unpin_ids() == []
    assert sched._lookup_results["21"] == (seq, 16)
    assert not sched._lookup_client_has_orphans()


def test_non_positive_hold_keeps_answers_until_consumed(closing):
    fake = _FakeLookupClient()
    client = ThreadedLookupClient(fake, max_hold_s=0)
    closing(client)
    assert client.submit([1, 2, 3], "22")
    _wait_done(client, "22")
    _age_answers(client, 1e6)
    client.pump()
    assert "22" in client.pending_ids()
    assert client.pop_unpin_ids() == []


def test_close_stops_the_thread_and_closes_the_inner_client():
    fake = _FakeLookupClient()
    client = ThreadedLookupClient(fake)
    client.close()
    assert fake.closed
    assert not client._thread.is_alive()
    assert not client.submit([1, 2, 3], "x")
