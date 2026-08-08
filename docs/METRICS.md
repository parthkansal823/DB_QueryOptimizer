# How every number is worked out

The precise definition of each metric the system reports, why it is defined
that way, and where the code lives.

`DASHBOARD.md` explains how to read these in the UI. This file is the
reference for what they actually mean.

---

## 1. The unit of measurement: a run

Almost everything here is built on one idea.

**A run is one decision.** It holds the time PostgreSQL's own plan took and
the time the plan that actually ran took, for the same query, measured seconds
apart.

Runs are reconstructed from `plan_execution_log` by grouping on
`(query key, created_at)`:

```sql
SELECT query_key, created_at,
       MIN(actual_total_time_ms) FILTER (WHERE is_baseline) AS native_ms,
       MIN(actual_total_time_ms) FILTER (WHERE is_chosen)   AS served_ms,
       MIN(actual_total_time_ms)                            AS best_ms
FROM plan_execution_log
WHERE actual_total_time_ms IS NOT NULL AND selector_used <> 'collection'
GROUP BY 1, 2
HAVING MIN(actual_total_time_ms) FILTER (WHERE is_baseline) IS NOT NULL
   AND MIN(actual_total_time_ms) FILTER (WHERE is_chosen)   IS NOT NULL
```

Grouping on `created_at` works because the column defaults to `now()`, and
PostgreSQL holds `now()` fixed for the whole transaction. Every row a single
`/query/analyze` request writes therefore shares one timestamp, which is what
makes a run identifiable afterwards.

The `HAVING` clause is the important part: a run only counts when **both**
sides were measured. Half-logged runs are dropped rather than counted on one
side of the comparison.

Code: `backend/app/stats.py`

### The query key

```
query_key = COALESCE(query_id, 'adhoc:' || md5(whitespace-normalised sql_text))
```

Named workload queries carry their own id. Anything typed into the dashboard
gets a fingerprint of its SQL, so repeat runs of the same query group together
and build history.

The `adhoc:` prefix keeps the two apart. Training excludes ad-hoc rows, because
`/query/analyze` runs each candidate once with no repetitions, and that noise
would go straight into the labels.

Code: `backend/app/logging_store.py` — `query_fingerprint`, `QUERY_KEY_SQL`

### What is excluded, and why

| Excluded | Reason |
|---|---|
| `selector_used = 'collection'` | The offline training sweep. It generates labels; it never served anyone a query |
| Runs missing either side | Cannot be compared without inventing a number |
| `actual_total_time_ms IS NULL` | Never executed |

The dashboard reports the excluded counts rather than hiding them.

---

## 2. Why matched pairs, and not two averages

The dashboard once divided two independent averages:

```
AVG(latency) WHERE is_chosen  ÷  AVG(latency) WHERE is_baseline
```

That ratio is meaningless, and not in a subtle way. A baseline row is written
for **every** query analysed. A chosen row was written only when the optimizer
was confident enough to deviate — which happens on the queries it understands,
and those skew cheap.

On the sample database this produced:

| | Value |
|---|---|
| Average of the plans it chose | 2.54 ms (7 runs) |
| Average of all PostgreSQL plans | 94.16 ms (52 runs) |
| Reported improvement | **97.3%** |
| Actual paired improvement | **~18%** |

The expensive queries — 513 ms, 647 ms — were on the PostgreSQL side of that
division and nowhere on the other side. The system had left almost every one
of them exactly as PostgreSQL planned it, and the dashboard called that a 97%
win.

Pairing fixes it because a run where the optimizer holds contributes the same
number to both totals, which is 0% improvement rather than absence.

---

## 3. Aggregate metrics

All sums are over the matched runs described above.

### Improvement

```
improvement_pct = (Σ native_ms − Σ served_ms) / Σ native_ms × 100
```

**Sums, not a mean of per-run percentages.** A mean over runs treats a 2 ms
query and a 600 ms query as equally important, which is how you end up
claiming a large win for shaving milliseconds off the cheapest thing in the
workload. Total time is what someone actually waits.

Negative means the optimizer made things slower overall.

### Headroom and headroom captured

```
headroom_ms           = Σ native_ms − Σ best_ms
headroom_captured_pct = (Σ native_ms − Σ served_ms) / headroom_ms × 100
```

`best_ms` is the fastest plan measured on that run — the **oracle**, the
ceiling any selector could have reached.

This is the fairer score. A database where PostgreSQL is already optimal
offers no headroom, and scoring 0% improvement there is not a failure. When
headroom is zero the metric reports `null` instead of dividing by zero.

Knowing the oracle is only possible because `/query/analyze` executes every
candidate. In production you never learn what the plans you skipped would have
cost, which makes this an offline diagnostic.

### Deviation rate

A run counts as deviating when the served row carries a hint:

```sql
bool_or(is_chosen AND hint IS NOT NULL)
```

A low rate is not automatically bad. Holding PostgreSQL's plan is a legitimate
decision, and the calibration work in `WRITEUP.md` §2.4.2 found that forcing
more deviation produced *less* net gain and started causing regressions.

---

## 4. Decision quality

An improvement percentage cannot distinguish a correct hold from a missed win.
Both score 0%. Only one is a good decision.

Every run is classified into exactly one outcome:

| Outcome | Condition |
|---|---|
| `deviated_win` | Changed the plan, and served was materially faster than native |
| `deviated_loss` | Changed the plan, and served was materially slower than native |
| `deviated_wash` | Changed the plan, difference inside the noise bar |
| `held_missed` | Kept native, but `best` was materially faster than native |
| `held_correct` | Kept native, and nothing materially faster existed |

### What "materially" means

```
materially_faster(a, b)  ⟺  b − a ≥ 2 ms  AND  (b − a) / b ≥ 5%
```

Both conditions must hold. The percentage alone would flag a 1 ms change on a
3 ms query; the absolute alone would flag a 3 ms change on a 600 ms query.

Those thresholds are deliberately the same shape as the gate the optimizer
applies to itself (`DEFAULT_MIN_GAIN_MS = 2.0`, `MIN_RELATIVE_GAIN = 0.05` in
`optimizer/learned.py`). Grading it against a stricter bar than it plays by
would manufacture failures that are not real.

### Attributed time

```
saved_ms      = Σ (native − served)  over deviated_win
regression_ms = Σ (served − native)  over deviated_loss
missed_ms     = Σ (native − best)    over held_missed
```

`regression_ms` and `missed_ms` are the two numbers worth reading first. One
is time the system *added*; the other is time it declined to save. Neither is
visible in the headline improvement figure.

Code: `backend/app/stats.py` — `classify`, `_decision_quality`

---

## 5. Regret

Regret for one decision is how much slower the served plan was than the best
plan available:

```
regret = max(served_ms − best_ms, 0)
```

Always at least zero, in milliseconds, and directly readable as "we have spent
this much more than a perfect optimizer would have."

Cumulative regret shows the *shape* over time in a way an average cannot. A
healthy learner's regret climbs quickly at first — it is exploring, and paying
for that — then flattens as it converges. Regret that keeps climbing in a
straight line means the same mistake is being repeated.

### Regret ratio

```
native_regret = Σ (native_ms − best_ms)
regret_ratio  = Σ regret ÷ native_regret
```

Below 1.00 means the optimizer has accumulated less regret than always
trusting PostgreSQL would have. This is the single number that says whether
the system was worth building.

The dashboard computes both from the same matched runs as everything else, so
the decision counts across panels always agree. `GET /stats/regret` uses its
own SQL in `optimizer/regret.py` and can differ slightly, because it also
counts production-path runs that have no native plan to pair against.

---

## 6. Cost model correlation

PostgreSQL chooses plans by minimising `total_cost`. The gap between that
ordering and the real one is the entire space a learned optimizer works in.

**Spearman rank correlation** between estimated cost and measured time across
every logged plan. Rank rather than Pearson because only the *ordering*
matters: cost is in arbitrary planner units, time is in milliseconds, and a
linear fit between them would not mean anything.

- `1.00` — cost orders plans exactly as real time does. Nothing to learn.
- `~0.67` — the value measured on the sample database.
- Vertical spread at a fixed cost is the exploitable gap.

### Disabled plans are excluded

PostgreSQL does not remove a disabled node type. It adds `disable_cost` (10¹⁰)
so the arithmetic buries it. Operator hints like `Set(enable_hashjoin off)`
therefore produce plans whose `total_cost` is a sentinel, not an estimate —
670 of 1618 rows on the sample database.

They are excluded, for two reasons. They would compress every real estimate
into the first pixel of a log axis, and they would flatter the correlation,
since a disabled plan is reliably both top-of-scale and slow. They are not
predictions, so they say nothing about prediction quality.

Code: `backend/app/stats.py` — `cost_vs_latency`, `_spearman`

---

## 7. Model and safety metrics

### Selection policies

| Policy | Behaviour |
|---|---|
| `greedy` | Take the ensemble's mean prediction |
| `thompson` | Sample one ensemble member per decision, so it keeps exploring |
| `risk_averse` | Penalise predictions the ensemble disagrees about |
| `pairwise_rank` | Lero-style learn-to-rank over plan pairs |

### Two safety mechanisms

**The safety veto is forward-looking.** Before running anything, it refuses a
plan PostgreSQL costs far above native. It catches catastrophes, but it
reasons about the very cost estimates this project exists to distrust, so it
cannot catch a plan that looks cheap on paper and is slow in practice.

**The regression guard looks backwards.** It reads the system's own history
and blocks the learned path for queries where it has measurably been slower:

```
blocked  ⟺  avg(chosen) > avg(native) × (1 + tolerance)
            AND observations ≥ min_observations
```

Defaults: `tolerance = 0.10`, `min_observations = 3`.

It is deliberately asymmetric. A query must prove a regression before it is
blocked, but a blocked query keeps being measured and can recover. Blocking is
a brake, not a ban.

This mattered more than prediction quality did: adding it moved mean captured
headroom from **+2% to +29%** and removed the negative runs (`WRITEUP.md`
§2.4).

### New data to learn from

Executions logged since the deployed model trained, counting **only rows a
retrain would actually use** — excluding ad-hoc dashboard traffic and rows
with no measured time.

Counting everything overstated it badly, and not harmlessly: the figure also
gates automatic retraining, so uncounted rows would buy retrains on data the
trainer never sees.

Code: `backend/app/retrain.py` — `rows_since_last_training`

### Champion / challenger

A retrain never deploys on faith:

1. Train a challenger on all accumulated feedback.
2. Score challenger and champion on the *same* held-out queries.
3. Deploy only if the challenger wins by more than `min_improvement` (2%).

Offline evaluation is optimistically biased and noisy (`WRITEUP.md` §2.2.1),
so promoting on any improvement at all would mean promoting on noise about
half the time. Requiring a margin makes the ratchet one-directional.

---

## 8. Measurement caveats

Worth knowing before quoting any of these numbers.

**Latency is `Actual Total Time` from the plan root.** It excludes planning
time and client round-trip. Consistent across every comparison here, but not
the same as wall-clock time your application sees.

**One execution per candidate on the dashboard path.** Latency has a heavy
right tail — a candidate can collide with autovacuum or a checkpoint. Offline
collection runs repetitions and takes the **median**, which is why training
data is cleaner than dashboard data. Aggregating to the median rather than the
mean removed every regression in an earlier round of results (`WRITEUP.md`
§2.2): same model, better labels.

**Cache state is not controlled.** The baseline runs first in each analyse
request, so it can pay a cold-cache cost the candidates do not. Spot checks on
the sample database showed no systematic ordering effect — later candidates
were often slower, not faster — but it is not eliminated.

**Row counts in the schema panel are estimates.** `pg_class.reltuples` is
maintained by autovacuum, not a live `COUNT(*)`.

---

## 9. Where each metric is produced

| Metric | Endpoint | Module |
|---|---|---|
| Matched runs, improvement, headroom | `GET /stats/trend` | `app/stats.py` |
| Decision quality | `GET /stats/trend` | `app/stats.py` |
| Per-query and per-day breakdowns | `GET /stats/trend` | `app/stats.py` |
| Cost vs. time correlation | `GET /stats/cost-model` | `app/stats.py` |
| Cumulative regret | `GET /stats/regret` | `app/optimizer/regret.py` |
| Deployed version, blocked queries | `GET /model/status` | `app/model_store.py`, `app/optimizer/regression_guard.py` |
| Planning overhead vs. execution | `POST /query/optimize` | `app/optimizer/planner.py` |
| Schema and cardinalities | `GET /schema` | `app/schema_graph.py` |

Behaviour is pinned by tests in `backend/tests/test_stats.py`, including the
two-averages bug this document opens with.
