"""Quota scheduler for the (API key, model) buckets.

Groq rate-limits per key per model, so three keys and two model tiers give six
independent token-per-minute buckets. This module decides which bucket serves a
call. It knows nothing about Groq clients, HTTP, prompts or user requests; it
only tracks how many tokens each bucket has been charged in a rolling window.

Design rules:

* Pools are per model. The 20B and 120B pools never share a lock, a cooldown or
  a counter, so one tier cannot block the other.
* Admission is atomic: choosing the bucket and reserving tokens in it happen
  under one lock, so concurrent callers cannot both claim the same headroom.
* A reservation (input estimate + max_tokens, which is what Groq itself counts
  as "requested") is a scheduling device for ONE bucket. It is replaced by the
  actual usage when the call settles. It is never a per-user-request budget and
  never changes what a call asks for.
* Cooldowns are per bucket. A 429 on key0/120B leaves key1/120B and key0/20B
  untouched. An authentication failure disables one key across all models.
* Waiting uses a condition variable per pool, woken whenever capacity is freed.
* Waiting is fair. Small requests may backfill spare capacity ahead of an older,
  larger waiter, but only `max_bypass` times; after that they queue behind it.
  Without this, large requests (the answer stage) starve behind a steady stream
  of small ones and are eventually forced into a bucket that cannot hold them.

State is per process. With several worker processes each has its own view; the
provider feeds Groq's own rate-limit headers back through ``sync_provider_used``
so the views converge on the real number.
"""

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .exceptions import AuthenticationError, RateLimitError, RequestTooLargeError

logger = logging.getLogger(__name__)


class _Entry:
    """One charge in a bucket's rolling window (reservation, then actual)."""

    __slots__ = ("ts", "tokens", "closed")

    def __init__(self, ts: float, tokens: int, closed: bool = False):
        self.ts = ts
        self.tokens = tokens
        self.closed = closed


class _Bucket:
    __slots__ = (
        "key_index",
        "window",
        "cooldown_until",
        "last_assigned",
        "requests",
        "tokens_total",
    )

    def __init__(self, key_index: int):
        self.key_index = key_index
        self.window: deque[_Entry] = deque()
        self.cooldown_until = 0.0
        self.last_assigned = 0.0
        self.requests = 0
        self.tokens_total = 0


class _Waiter:
    """A call inside acquire(); `bypassed` counts younger calls admitted ahead."""

    __slots__ = ("bypassed",)

    def __init__(self):
        self.bypassed = 0


class _Pool:
    """All buckets for one model, guarded by one lock."""

    def __init__(self, key_count: int):
        self.cond = threading.Condition()
        self.buckets = [_Bucket(i) for i in range(key_count)]
        self.waiters: list[_Waiter] = []  # arrival order, oldest first


@dataclass(frozen=True)
class Lease:
    """A granted reservation. Pass it back to ``settle`` or ``release``."""

    model: str
    key_index: int
    reserved: int
    waited: float
    bucket_used: int  # bucket's window total right after this reservation
    _entry: _Entry


class QuotaScheduler:
    def __init__(
        self,
        *,
        key_count: int,
        limit_for: Callable[[str], int],
        admission_ratio: float = 0.90,
        window_seconds: float = 60.0,
        max_wait: float = 20.0,
        max_bypass: int = 3,
    ):
        self._key_count = key_count
        self._limit_for = limit_for
        self._ratio = admission_ratio
        self._window = window_seconds
        self._max_wait = max_wait
        self._max_bypass = max_bypass
        self._pools: dict[str, _Pool] = {}
        self._pools_lock = threading.Lock()
        # Written under _pools_lock, read lock-free (single float reads).
        self._key_disabled_until = [0.0] * key_count

    # ------------------------------------------------------------------ pools

    def _pool(self, model: str) -> _Pool:
        pool = self._pools.get(model)
        if pool is None:
            with self._pools_lock:
                pool = self._pools.get(model)
                if pool is None:
                    pool = self._pools[model] = _Pool(self._key_count)
        return pool

    def _used(self, bucket: _Bucket, now: float) -> int:
        """Tokens charged in the rolling window. Pool lock must be held."""
        window = bucket.window
        horizon = now - self._window
        while window and window[0].ts <= horizon:
            window.popleft()
        return sum(entry.tokens for entry in window)

    def _blocked_until(self, bucket: _Bucket) -> float:
        return max(bucket.cooldown_until, self._key_disabled_until[bucket.key_index])

    @staticmethod
    def _fits(used: int, tokens: int, ceiling: float, limit: int) -> bool:
        # An idle bucket may take anything up to the provider limit itself.
        return used + tokens <= ceiling or (used <= 0 and tokens <= limit)

    def _time_until_fit(
        self, bucket: _Bucket, tokens: int, now: float, ceiling: float, limit: int
    ) -> float:
        """Seconds until this bucket could admit `tokens`. Pool lock held."""
        blocked = max(self._blocked_until(bucket) - now, 0.0)
        remaining = self._used(bucket, now)
        if self._fits(remaining, tokens, ceiling, limit):
            return blocked
        for entry in bucket.window:  # oldest first
            remaining -= entry.tokens
            if self._fits(remaining, tokens, ceiling, limit):
                return max(blocked, entry.ts + self._window - now, 0.0)
        return blocked

    # --------------------------------------------------------------- acquire

    def acquire(
        self, model: str, tokens: int, avoid: Iterable[int] = ()
    ) -> Lease:
        """Pick a bucket for `tokens` and reserve them there.

        Least-loaded admissible bucket wins; ties go to the one assigned longest
        ago, which degenerates to round-robin on an idle pool. Buckets in
        `avoid` are chosen only when nothing else admits.

        When nothing admits, wait (bounded) for the earliest opening, in fair
        order (see module docstring). After the bound: if some bucket is merely
        full by our own estimate, use the least-loaded one and let Groq be the
        authority; if every bucket is cooling or disabled, raise instead of
        hammering a key that told us to back off.
        """
        limit = self._limit_for(model)
        if tokens > limit:
            raise RequestTooLargeError(
                f"Request of ~{tokens} tokens exceeds the {limit} tokens-per-minute "
                f"limit of {model}; no key can serve it."
            )
        ceiling = limit * self._ratio
        avoid = frozenset(avoid)
        pool = self._pool(model)
        started = time.monotonic()
        deadline = started + self._max_wait
        me = _Waiter()

        with pool.cond:
            pool.waiters.append(me)
            try:
                while True:
                    now = time.monotonic()
                    remaining = deadline - now

                    if remaining > 0 and self._older_waiter_starving(pool, me):
                        # Stand in line; woken when that waiter leaves.
                        pool.cond.wait(timeout=remaining)
                        continue

                    chosen = self._pick(pool, tokens, avoid, now, ceiling, limit)
                    if chosen is not None:
                        return self._admit(pool, me, chosen, model, tokens, now, started)

                    usable = [
                        b
                        for b in pool.buckets
                        if self._key_disabled_until[b.key_index] <= now
                    ]
                    if not usable:
                        raise AuthenticationError(
                            f"No usable Groq key for {model}: all keys are disabled."
                        )

                    soonest = min(
                        self._time_until_fit(b, tokens, now, ceiling, limit)
                        for b in usable
                    )
                    open_buckets = [b for b in usable if b.cooldown_until <= now]

                    if soonest <= remaining:
                        logger.info(
                            "[LLM_SCHEDULER] model=%s all buckets full/cooling; "
                            "waiting %.2fs requested=%s",
                            model,
                            soonest,
                            tokens,
                        )
                        pool.cond.wait(timeout=max(soonest, 0.005) + 0.01)
                        continue
                    if not open_buckets:
                        # Cooldowns are not freed by in-flight calls finishing, so
                        # waiting would only delay the failure.
                        raise RateLimitError(
                            f"All {model} buckets are cooling down; earliest "
                            f"opening in {soonest:.1f}s."
                        )
                    if remaining > 0:
                        # Windows will not expire in time, but a call settling
                        # with less usage than it reserved can free capacity
                        # sooner; the condition is notified when that happens.
                        pool.cond.wait(timeout=remaining)
                        continue

                    chosen = min(
                        open_buckets,
                        key=lambda b: (self._used(b, now), b.last_assigned, b.key_index),
                    )
                    logger.warning(
                        "[LLM_SCHEDULER] model=%s no bucket admits requested=%s "
                        "within %.0fs; forcing key=%s and deferring to the provider",
                        model,
                        tokens,
                        self._max_wait,
                        chosen.key_index,
                    )
                    return self._admit(pool, me, chosen, model, tokens, now, started)
            finally:
                pool.waiters.remove(me)
                pool.cond.notify_all()  # the next in line may now proceed

    def _pick(
        self,
        pool: _Pool,
        tokens: int,
        avoid: frozenset,
        now: float,
        ceiling: float,
        limit: int,
    ) -> _Bucket | None:
        """Least-loaded admissible, non-blocked bucket, or None. Pool lock held."""
        best = None
        best_rank = None
        for bucket in pool.buckets:
            if self._blocked_until(bucket) > now:
                continue
            used = self._used(bucket, now)
            if not self._fits(used, tokens, ceiling, limit):
                continue
            rank = (bucket.key_index in avoid, used, bucket.last_assigned, bucket.key_index)
            if best_rank is None or rank < best_rank:
                best, best_rank = bucket, rank
        return best

    def _older_waiter_starving(self, pool: _Pool, me: _Waiter) -> bool:
        for waiter in pool.waiters:
            if waiter is me:
                return False
            if waiter.bypassed >= self._max_bypass:
                return True
        return False

    def _admit(
        self,
        pool: _Pool,
        me: _Waiter,
        bucket: _Bucket,
        model: str,
        tokens: int,
        now: float,
        started: float,
    ) -> Lease:
        for waiter in pool.waiters:  # everyone older than `me` was just bypassed
            if waiter is me:
                break
            waiter.bypassed += 1
        return self._reserve(pool, bucket, model, tokens, now, started)

    def _reserve(
        self,
        pool: _Pool,
        bucket: _Bucket,
        model: str,
        tokens: int,
        now: float,
        started: float,
    ) -> Lease:
        entry = _Entry(now, int(tokens))
        bucket.window.append(entry)
        bucket.last_assigned = now
        bucket.requests += 1
        return Lease(
            model=model,
            key_index=bucket.key_index,
            reserved=int(tokens),
            waited=now - started,
            bucket_used=self._used(bucket, now),
            _entry=entry,
        )

    # --------------------------------------------------------- settle/release

    def settle(self, lease: Lease, actual_tokens: int | None) -> int:
        """Replace the reservation with what the call really cost.

        If the provider reported no usage (`None`), the reservation stays as the
        conservative charge until it ages out of the window. Idempotent.
        Returns the bucket's window total afterwards.
        """
        pool = self._pool(lease.model)
        with pool.cond:
            entry = lease._entry
            bucket = pool.buckets[lease.key_index]
            if not entry.closed:
                entry.closed = True
                if actual_tokens is not None:
                    entry.tokens = max(int(actual_tokens), 0)
                bucket.tokens_total += entry.tokens
                pool.cond.notify_all()
            return self._used(bucket, time.monotonic())

    def release(self, lease: Lease) -> None:
        """Drop the reservation of a call that failed before consuming quota."""
        self.settle(lease, 0)

    # ------------------------------------------------------------- feedback

    def cool(self, model: str, key_index: int, seconds: float) -> None:
        """Take one (key, model) bucket out of rotation for `seconds`."""
        pool = self._pool(model)
        with pool.cond:
            pool.buckets[key_index].cooldown_until = time.monotonic() + seconds
            pool.cond.notify_all()

    def disable_key(self, key_index: int, seconds: float) -> None:
        """Take a key out of rotation for every model (e.g. after a 401)."""
        with self._pools_lock:
            self._key_disabled_until[key_index] = time.monotonic() + seconds
            pools = list(self._pools.values())
        for pool in pools:
            with pool.cond:
                pool.cond.notify_all()

    def sync_provider_used(
        self, model: str, key_index: int, provider_used: int | None
    ) -> None:
        """Adopt the provider's view if it says the bucket is fuller than ours.

        Covers other worker processes and anything we mis-estimated. It can only
        make the scheduler more cautious, never less.
        """
        if not provider_used or provider_used <= 0:
            return
        pool = self._pool(model)
        with pool.cond:
            bucket = pool.buckets[key_index]
            now = time.monotonic()
            tracked = self._used(bucket, now)
            if provider_used > tracked:
                bucket.window.append(_Entry(now, provider_used - tracked, closed=True))

    def snapshot(self, model: str) -> list[dict]:
        """Read-only view of a pool, for logs and tests."""
        pool = self._pool(model)
        with pool.cond:
            now = time.monotonic()
            return [
                {
                    "key": b.key_index,
                    "used": self._used(b, now),
                    "cooling_for": max(b.cooldown_until - now, 0.0),
                    "disabled_for": max(self._key_disabled_until[b.key_index] - now, 0.0),
                    "requests": b.requests,
                    "tokens_total": b.tokens_total,
                }
                for b in pool.buckets
            ]