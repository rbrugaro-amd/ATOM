# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Background-thread lookups for LMCache's synchronous ZMQ lookup client.

LMCache's in-process ``LMCacheLookupClient.lookup`` hashes the prompt and
then waits for every rank's lookup server (``send_and_recv_all``). Run on the
scheduler thread, that wait is a stall in which no step is issued. This
wrapper moves the call onto one background thread and gives it the
asynchronous surface the offload scheduler already drives for LMCache MP
(``submit`` / ``pump`` / ``pending_ids`` / ``discard`` /
``pop_discarded_hits``, see ``ChunkedOffloadSchedulerBase.prefetch_lookups``):

* a request's lookup is submitted while it waits, ahead of admission;
* admission consumes the answer with the ordinary synchronous ``lookup``,
  which returns at once when the answer is in, waits only for the remainder
  of an in-flight one, and otherwise runs the lookup ahead of any queued
  prefetch. There is no "pending" state: admission never passes a request
  over (no ``is_pending``), so an unready lookup costs what it costs today.

Pin invariant. Every lookup server answers with ``pin=True``: the chunks it
found stay pinned under the lookup ID until the worker receives that ID in
``lookup_requests_in_step`` and calls ``lookup_unpin``. The worker keeps one
pin set per ID (a second lookup overwrites the first's record, whose pins
would then never be released), so the scheduler must never send a second
lookup for an ID whose pin has not been released. Hence:

* one job per ID, from submission until it is consumed or released;
* a consumed answer becomes the scheduler's own pin (``_lookup_results`` /
  ``_lookup_in_step``) exactly as a synchronous lookup's does;
* a lookup nobody will consume (``discard``) is released only once it has
  answered -- before that the servers may not have pinned anything yet -- by
  handing its ID to the scheduler (``pop_unpin_ids``), which puts it in the
  next metadata's unpin list;
* only one thread ever touches the inner client, whose ZMQ REQ sockets are
  not thread-safe.

Because the answer's chunks stay pinned until they are released, a
prefetched hit cannot be invalidated by eviction while it waits: the tier can
only have gained a longer prefix, which costs a shorter load, never a wrong
one. The HBM frontier is read when the answer is consumed, not when it was
fetched.

That holds only while the pin does. LMCache's pin monitor force-unpins a
chunk pinned for longer than ``pin_timeout_sec``, so an answer left
unconsumed for ``UNCONSUMED_HOLD_FRACTION`` of that timeout is released like
a discarded one (``max_hold_s``). Its request is still waiting: admission
adopts the answer if the release has not been sent yet (its pin is then still
well inside the timeout) and looks it up again otherwise, and the next
prefetch re-asks once the release is sent.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from concurrent.futures import Future

logger = logging.getLogger("atom")

_QUEUED, _RUNNING, _DONE, _CANCELLED = "queued", "running", "done", "cancelled"

# Fraction of the tier's pin (or read-lock) lifetime an answered lookup may
# wait for its request's admission before it is released. Well below 1: an
# answer that outlives the pin hands the load chunks the tier may already have
# evicted. Shared with lmcache_mp's lookup driver (`mp/lookup.py`).
UNCONSUMED_HOLD_FRACTION = 0.4


class _Job:
    __slots__ = ("answered", "future", "sid", "state", "submitted", "token_ids")

    def __init__(self, sid: str, token_ids: list[int]) -> None:
        self.sid = sid
        self.token_ids = token_ids
        self.future: Future = Future()
        self.state = _QUEUED
        self.submitted = time.perf_counter()
        # `time.monotonic()` when the lookup answered (pinned); None before.
        self.answered: float | None = None


class ThreadedLookupClient:
    """Run a synchronous LMCache lookup client on one background thread.

    Every method except the worker loop runs on the scheduler thread; the
    job table (`_jobs`) and the release lists are scheduler-thread state. The
    queue and job states are shared with the worker under `_cv`.

    `max_hold_s` bounds how long an answered lookup may wait unconsumed
    before its pin is released (see the module docstring); non-positive
    keeps it until consumed or discarded.
    """

    def __init__(self, inner, max_hold_s: float = 0.0) -> None:
        self._inner = inner
        self._max_hold_s = float(max_hold_s)
        self.token_database = getattr(inner, "token_database", None)
        self._cv = threading.Condition()
        self._queue: collections.deque[_Job] = collections.deque()
        # sid -> job, from submission until consumed or released.
        self._jobs: dict[str, _Job] = {}
        # Discarded while in flight: released once answered.
        self._orphans: set[str] = set()
        # Answered, never to be consumed: sid -> job. Drained into the
        # metadata's unpin list by `pop_unpin_ids`.
        self._unpins: dict[str, _Job] = {}
        self._discarded_hits: dict[str, int] = {}
        self._counters: collections.Counter = collections.Counter()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="atom-lmcache-lookup", daemon=True
        )
        self._thread.start()

    # -- worker ------------------------------------------------------------
    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._queue and not self._closed:
                    self._cv.wait()
                if self._closed:
                    return
                job = self._queue.popleft()
                job.state = _RUNNING
            t0 = time.perf_counter()
            try:
                result = self._inner.lookup(job.token_ids, lookup_id=job.sid)
            except BaseException as exc:  # noqa: BLE001  # handed to the consumer
                with self._cv:
                    job.state = _DONE
                    self._counters["bg_lookup_errors"] += 1
                job.future.set_exception(exc)
                continue
            us = int((time.perf_counter() - t0) * 1e6)
            with self._cv:
                job.state = _DONE
                job.answered = time.monotonic()
                self._counters["bg_lookup_n"] += 1
                self._counters["bg_lookup_us"] += us
                self._counters["bg_lookup_max_us"] = max(
                    self._counters["bg_lookup_max_us"], us
                )
            job.future.set_result(result)

    def _enqueue(self, job: _Job, *, front: bool) -> None:
        with self._cv:
            if self._closed:
                raise RuntimeError("LMCache lookup thread is closed")
            if front:
                self._queue.appendleft(job)
            else:
                self._queue.append(job)
            self._cv.notify()

    def _promote(self, job: _Job) -> None:
        """Move a queued job ahead of the other prefetches: someone waits on it."""
        with self._cv:
            if job.state == _QUEUED:
                self._queue.remove(job)
                self._queue.appendleft(job)

    # -- the asynchronous surface the offload scheduler drives -------------
    def supports_async(self) -> bool:
        return True

    def submit(self, token_ids: list[int], lookup_id: str) -> bool:
        """Queue this request's lookup without waiting. False if not queued."""
        if self._closed or lookup_id in self._jobs or lookup_id in self._unpins:
            return False
        job = _Job(lookup_id, token_ids)
        self._jobs[lookup_id] = job
        self._enqueue(job, front=False)
        self._counters["submitted"] += 1
        return True

    def pending_ids(self):
        """Lookups submitted and not yet consumed or released (do not mutate)."""
        return self._jobs.keys()

    def pump(self) -> None:
        """Release discarded lookups that have answered since the last call,
        and answers left unconsumed past `max_hold_s`."""
        for sid in [s for s in self._orphans if self._jobs[s].future.done()]:
            self._orphans.discard(sid)
            self._release(self._jobs.pop(sid))
        if self._max_hold_s <= 0:
            return
        now = time.monotonic()
        stale = [
            sid
            for sid, job in self._jobs.items()
            if sid not in self._orphans
            and job.answered is not None
            and now - job.answered >= self._max_hold_s
        ]
        for sid in stale:
            # A waiting request's answer, not a detached one's: no save floor.
            self._release(self._jobs.pop(sid), record_hit=False)
            self._counters["stale_released"] += 1

    @staticmethod
    def _answer(job: _Job) -> int | None:
        """A finished job's hit; None if it failed or did not answer."""
        if job.future.cancelled() or job.future.exception() is not None:
            return None
        raw = job.future.result()
        return None if raw is None else int(raw)

    def _release(self, job: _Job, *, record_hit: bool = True) -> None:
        """Queue an answered, unconsumed lookup's pin for release."""
        hit = self._answer(job) if record_hit else None
        if hit is not None:
            self._discarded_hits[job.sid] = hit
        self._unpins[job.sid] = job
        self._counters["released_unconsumed"] += 1

    def discard(self, lookup_id: str) -> None:
        """Forget a lookup its request will never consume."""
        job = self._jobs.get(lookup_id)
        if job is None:
            return
        # A queued job is not withdrawn: `detach_lookup` discards a lookup it
        # still wants sent (LRU refresh, save floor), and for an aborted
        # request one wasted background round trip is cheaper than telling
        # the two apart.
        if job.future.done():
            del self._jobs[lookup_id]
            self._orphans.discard(lookup_id)
            self._release(job)
        else:
            self._orphans.add(lookup_id)

    def has_orphans(self) -> bool:
        """Whether a discarded lookup still has a pin to hand back."""
        return bool(self._orphans or self._unpins)

    def pop_unpin_ids(self) -> list[str]:
        """IDs of answered lookups nobody will consume; their pins go back."""
        self.pump()
        if not self._unpins:
            return []
        ids = list(self._unpins)
        self._unpins.clear()
        return ids

    def pop_discarded_hits(self) -> dict[str, int]:
        hits, self._discarded_hits = self._discarded_hits, {}
        return hits

    def pop_counters(self) -> dict[str, int]:
        with self._cv:
            counters = dict(self._counters)
            self._counters.clear()
        return counters

    # -- the synchronous surface -------------------------------------------
    def lookup(self, token_ids, lookup_id: str, request_configs=None):
        """The hit for this prompt; an answered prefetch returns at once."""
        released = self._unpins.get(lookup_id)
        if released is not None:
            # Discarded and answered, its release not yet dispatched. A second
            # lookup under this ID would orphan its pin, so either adopt it
            # (same prompt) or answer "unknown" until the release is sent.
            if not self._same_prompt(released.token_ids, token_ids):
                self._counters["prompt_mismatch"] += 1
                return None
            del self._unpins[lookup_id]
            self._discarded_hits.pop(lookup_id, None)
            self._counters["adopted_released"] += 1
            return self._answer(released)
        self._orphans.discard(lookup_id)
        job = self._jobs.get(lookup_id)
        if job is not None and not self._same_prompt(job.token_ids, token_ids):
            # Never expected (a prompt is fixed). The old job may hold a pin
            # under this ID, so no new lookup may be sent until it is
            # released: answer "unknown" and let the memo retry later.
            self.discard(lookup_id)
            self.pump()
            self._counters["prompt_mismatch"] += 1
            return None
        if job is None:
            job = _Job(lookup_id, token_ids)
            self._jobs[lookup_id] = job
            self._enqueue(job, front=True)
            self._counters["sync_lookup"] += 1
        elif job.future.done():
            self._counters["prefetch_ready"] += 1
        else:
            self._promote(job)
            self._counters["prefetch_waited"] += 1
        try:
            return job.future.result()
        finally:
            # Consumed (or failed): the scheduler now owns the pin.
            if self._jobs.get(lookup_id) is job:
                del self._jobs[lookup_id]

    @staticmethod
    def _same_prompt(submitted, asked) -> bool:
        if submitted is asked:
            return True
        if len(submitted) != len(asked):
            return False
        if isinstance(submitted, list) and isinstance(asked, list):
            return submitted == asked
        return list(submitted) == list(asked)

    def clear_lookup_status(self, lookup_id: str) -> None:
        # Only the inner client's answer memo (a dict); never the jobs, whose
        # pins are released through `discard`.
        self._inner.clear_lookup_status(lookup_id)

    def close(self) -> None:
        with self._cv:
            self._closed = True
            for job in self._queue:
                job.state = _CANCELLED
                job.future.cancel()
            self._queue.clear()
            self._cv.notify_all()
        self._thread.join(timeout=2.0)
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()

    def __getattr__(self, name):
        # Anything else (lookup_cache, supports_producer_reuse, ...) is the
        # inner client's. Only reached for attributes not set above.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._inner, name)


__all__ = ["ThreadedLookupClient"]
