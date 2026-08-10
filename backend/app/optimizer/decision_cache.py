"""
Remembers which plan was chosen for a query, so repeat traffic does not
re-derive it.

## Why

`optimizer/planner.py` makes the economic case for the production path: N
cheap `EXPLAIN`s to choose, one real execution to serve. That arithmetic only
works while planning is small relative to what it saves. Measured on a 4-table
join against this database:

    execution_ms: 21.8    optimizer_overhead_ms: 25.2    n_candidates_planned: 10

The optimizer cost more than the query it was optimizing. Ten planning round
trips is not free at this scale, and every one of them was repeated in full the
next time the same query arrived -- to reach the identical conclusion, because
nothing that fed it had changed.

Caching the *decision* rather than the plan is what makes this safe. A hit
skips the planning round trips and goes straight to executing the chosen hint;
the query still runs for real, is still logged, and is still measured. What is
reused is the choice, not the result.

## What invalidates an entry

A decision is only valid while the things that produced it hold:

  * **The model.** A retrain or rollback swaps the policy that made every
    cached choice, so `app.main` clears the cache on promotion. Without that, a
    freshly promoted model would keep serving its predecessor's decisions and
    look like it had changed nothing.
  * **Time.** Data drifts, statistics are re-analysed, and a hint that was
    right when it was cached can stop being right. Entries expire after
    `ttl_seconds` rather than living until the process restarts.
  * **The regression guard.** Checked by the caller on every request, before
    the cache is consulted -- a query that has started regressing must stop
    being served its cached deviation immediately, not when the entry ages out.

## What is deliberately not cached

Stochastic policies. `thompson` samples a different ensemble member per
decision *by design*: that sampling is how the bandit keeps learning about
plans it currently believes are bad. Replaying one frozen sample would keep the
exploration machinery in place while quietly disabling it, which is worse than
not caching at all -- it would look like it was still exploring.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

# Policies whose choice is drawn fresh each time. Caching these would turn
# exploration into a single frozen sample; see the module docstring.
STOCHASTIC_POLICIES = frozenset({"thompson"})


class DecisionCache:
    """
    A bounded, expiring map from query identity to the hint chosen for it.

    Least-recently-used eviction keeps a long-running server from accumulating
    an entry per distinct ad-hoc query forever, which on a dashboard that
    fingerprints every query it is handed is a real way to leak memory.
    """

    def __init__(self, max_entries: int = 512, ttl_seconds: float = 300.0):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        # FastAPI serves requests from a thread pool, so two concurrent
        # requests for the same query can reach this at once.
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str | None) -> dict | None:
        """The cached decision for `key`, or None if absent or expired."""
        if key is None:
            return None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            stored_at, decision = entry
            # `>=` so that a zero TTL means "expired immediately" rather than
            # "never expires", which is what `>` gives when the clock has not
            # ticked between the write and the read.
            if (time.monotonic() - stored_at) >= self.ttl_seconds:
                del self._entries[key]
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return dict(decision)

    def put(self, key: str | None, decision: dict) -> None:
        if key is None:
            return
        with self._lock:
            self._entries[key] = (time.monotonic(), dict(decision))
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        """Drop everything -- called when the served model changes."""
        with self._lock:
            self._entries.clear()

    def stats(self) -> dict:
        with self._lock:
            looked_up = self.hits + self.misses
            return {
                "entries": len(self._entries),
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": (self.hits / looked_up) if looked_up else None,
            }
