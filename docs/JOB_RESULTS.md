# JOB/IMDB stretch goal: results

From a real run, not simulated. `data/job/README.md` has the exact commands.
This records what actually happened, including the parts that turned out
smaller than the synthetic-schema evaluation.

## What was loaded

The real IMDB CSV dump from the JOB paper (May 2013 snapshot), all 21 tables,
loaded with `data/job/load_job.sh`:

| table | rows |
|---|---|
| cast_info | 36,236,973 |
| movie_info | 14,818,380 |
| movie_keyword | 4,523,930 |
| name | 4,168,204 |
| person_info | 2,963,418 |
| movie_companies | 2,609,129 |
| title | 2,528,239 |
| char_name | 3,140,333 |
| movie_info_idx | 1,380,035 |
| aka_name | 901,343 |
| aka_title | 361,472 |
| company_name | 234,997 |
| complete_cast | 135,086 |
| keyword | 134,170 |
| movie_link | 29,997 |
| (7 small lookup tables) | <120 each |
| **total** | **~74.5M rows** |

`schema_introspection.discover_table_cardinalities` found all 21 tables on its
own (`n_tables_in_schema: 21` in the training output below).
`optimizer/features.py` never needed to know those table names in advance.

## What was run

This was a **smoke test, not a full retrain**. Eight real JOB query files
(`3a, 3b, 3c, 2a, 2b, 2c, 2d, 5a`), picked for being among the smaller and
faster of the 113, run through the unchanged `app.collect_data` and
`app.train` pipeline (via `app.collect_data_job`, with `DATABASE_URL` pointed
at the `job` database).

### The first attempt was invalid, and said so if you read it closely

The first run produced this:

```json
{
  "n_rows_total": 72, "test_mae_ms": 53.19, "n_tables_in_schema": 21,
  "n_held_out_queries": 2,
  "avg_latency_ms": { "native": 363.27, "heuristic": 363.27, "learned": 363.27 }
}
```

Native, heuristic and learned agreeing to the *second decimal place* is not a
coincidence. An earlier draft of this file read it as a harmless "the
optimizer correctly deferred to Postgres" result. It was not. It was the
signature of the `pg_hint_plan` bug in `docs/WRITEUP.md` §2.0: hints were
being silently ignored, so every "candidate" was the same native plan. Three
selectors choosing between eight copies of one plan will of course tie
exactly.

That reading was available at the time and was missed. **Identical numbers
from independent methods should always be treated as a bug signal first and a
finding second.**

### Re-run, after the fix

Same 8 queries, same pipeline, with `shared_preload_libraries=pg_hint_plan`
set and the hint moved ahead of `EXPLAIN`:

```json
{
  "n_rows_raw": 194, "n_rows_total": 65, "n_features": 109,
  "n_tables_in_schema": 21, "n_held_out_queries": 2,
  "test_mae_ms": 1656.7, "test_mean_uncertainty_ms": 401.2,
  "avg_latency_ms": {
    "native": 3702.98, "heuristic": 3702.98, "oracle_best_possible": 918.84,
    "learned_greedy": 3702.98, "learned_risk_averse": 3702.98,
    "learned_pairwise_rank": 3702.98
  }
}
```

Two things stand out, and they point in opposite directions.

**The headroom on JOB is huge.** Native Postgres averages 3703 ms on these
held-out queries. The best available candidate averages 919 ms. That is a
**75% reduction sitting on the table**, an order of magnitude more opportunity
than the synthetic schema's ~8%. This is exactly what Leis et al. built JOB to
expose: on queries with correlated filters and many joins, PostgreSQL's row
estimates degrade badly and its chosen plan ends up far from the best one. The
premise of the whole field is visible in one line of JSON.

**The model captures none of it.** Every selector returns the native plan, and
this time it is not the hint bug. Hints do bind here: the feature vector is
109 wide across 21 discovered tables, and the oracle differs from native,
which is only possible if the candidates actually differ. The real reason is
that a test MAE of **1657 ms** on queries averaging 3703 ms means the model
has no usable signal. 194 raw executions across 8 queries, 65 after median
aggregation, with 2 held-out queries, is nowhere near enough to learn a
17-table join space. With no signal to go on, the selectors fall back to the
candidate that happens to be the native plan, and the safety veto would have
caught them if they had not.

The honest reading: **JOB confirms the opportunity is real, and confirms this
system is far from capturing it.** That is more useful than a flattering
number, and it puts a figure on the gap (75% available, 0% captured) instead
of leaving it vague.

## Reading this honestly

Two held-out queries is nowhere near enough to claim the model learned
anything JOB-specific, in either direction. What this run does establish is
narrower and still worth having: the pipeline runs end to end against a real
74.5M-row research benchmark with **zero code changes**. The
schema-introspection work (§3 of the writeup) meant `features.py` discovered
all 21 tables by itself. The scale caveats in `docs/WRITEUP.md` §2.2.2 apply
here at least as strongly as they do on the synthetic schema.

## What a real JOB evaluation would need

Future work, not attempted here.

- **All 113 queries collected.** This ran 8. The query-level test split needs
  enough queries to mean anything: about 28 held out at 25%, not 2.
- **Several repetitions per candidate.** JOB queries against 10M+ row tables
  vary more between runs than the synthetic schema does, and this run used
  `--reps 1`.
- **The join-method stretch goal enabled.** It was turned off for this run
  (`include_join_methods=False` in `app.collect_data_job`) to keep the smoke
  test fast against unindexed multi-million-row joins. A full run would enable
  it, at a large cost in runtime.
- **The index experiment.** JOB's `fkindexes.sql` was applied, but a real
  evaluation would also want the with-and-without-indexes comparison that
  `data/schema.sql` suggests for the synthetic schema, since indexes directly
  affect how much headroom a learned optimizer has over the cost-based one.

---

## Re-run on PostgreSQL 18, 2026-09-05

The `job` database did not survive the upgrade from PostgreSQL 16 and was
reloaded from `data/job/csv` with `bash data/job/load_job.sh`: **8.7 GB, 73.0M
live rows, 22 tables**, no errors. `app.collect_data_job` now runs from the host
as well as from inside the backend container (`--query-dir`).

Everything built since the result above -- shape coverage, the 59-query
workload, the worst-case promotion metric, the PostgreSQL 18 features, and the
corrections in `WRITEUP.md` 2.42-2.54 -- had never been tested against real IMDB
data.

### The measurement

All 30 of the smallest JOB queries. **1614 raw rows** against the original run's
194, **8 held out** against its 2:

| selector | avg latency | headroom captured |
|---|---|---|
| native PostgreSQL | 3960.2 ms | — |
| cost heuristic | 2285.3 ms | +42% |
| **oracle (best candidate)** | **695.6 ms** | 100% |
| **`risk_averse`** | **713.7 ms** | **+99.4%** |
| `pairwise_rank` | 722.8 ms | +99.2% |
| `greedy` | 726.9 ms | +99.0% |
| `thompson` | 863.9 ms | +94.8% |

Held-out native averages **3960 ms** against the original run's **3703 ms** --
the same expensive-query regime, which is what makes this a comparison rather
than a different measurement. Where that run captured **0%**, every policy now
captures **more than 94%**, and the spread between the top three (99.0-99.4%) is
far inside the noise.

That all four converge matters more than which one leads. A single policy at
99% invites the question of whether it got lucky; four independent selection
rules landing within half a point of the oracle is a property of the *candidate
set and the model*, not of the rule reading them.

### Where it comes from, because an aggregate that high is a question

| query | native | oracle | served | vs native | captured |
|---|---|---|---|---|---|
| `17c` | 21603.7 | 1963.3 | **1963.3** | **0.09x** | **100.0%** |
| `17d` | 8822.2 | 2447.9 | **2447.9** | **0.28x** | **100.0%** |
| `4b` | 64.6 | 33.2 | **33.2** | **0.51x** | **100.0%** |
| `1b` | 61.4 | 0.1 | 0.2 | 0.00x | 99.8% |
| `6c` | 4.7 | 4.4 | 4.9 | 1.04x | regression, +0.2 ms |
| `2b` | 463.6 | 460.9 | 522.7 | 1.13x | regression, +59 ms |
| `2d` | 586.1 | 571.7 | 715.4 | 1.22x | regression, +129 ms |
| `5b` | 75.3 | 83.3 | 95.0 | 1.26x | nothing to win |
| **total** | **31681.6** | **5564.8** | **5782.7** | **0.18x** | **99.2%** |

**On three queries it picks the oracle plan exactly** -- the same plan, not a
close one -- and lands within 0.1 ms of it on a fourth. `17c` goes from **21.6
seconds to 2.0**, an 11x speedup, and is 75.8% of all the saving; `17d` is
another 24.6%. Top-1 exact-oracle accuracy is **3 of 8**, against 25% on the
original collection.

**Four of the eight regress**, totalling **-208.2 ms** against 26,099 ms saved.
Three of those four are queries with almost nothing to win: `2b` had 2.7 ms of
headroom, `2d` had 14.4 ms, and `5b`'s oracle (83.3 ms) is *slower* than native
(75.3 ms), so no candidate in the action space beats PostgreSQL and any
deviation is a loss by construction. `6c` gave up 0.2 ms on a 4.7 ms query.

So: half the held-out queries get slightly worse, the aggregate is carried by
two very expensive ones, and the trade is 26 seconds saved for 0.2 seconds lost.
That is the right trade for this benchmark and it is not "the model is right
about every query".

> **Superseded in part.** Re-collecting under the widened action space (§2.56)
> lowers the oracle on most of these queries, so the headroom column below is
> the *narrow* space's. `5b` in particular is described here as a query where no
> candidate beats native; with scan methods collected it has **9.4%** available
> (oracle 68.2 ms against 75.3 ms native). `2b` goes 0.6% → 34.0% and `2d`
> 2.5% → 40.0%. The 99.2% figure is against a ceiling now known to be too high.

**These are ungated numbers.** `_evaluate_selection` reports what each policy
picks, with no confidence gate applied. The gate exists precisely for queries
like `2b` and `2d` -- a predicted win smaller than the model's own error -- and
would decline most of them. The 99.2% is the ceiling the gate is choosing from,
not what a deployment serves.

### Correction: "almost nothing to win" was true of the action space, not the query

The four regressions above are explained as queries with nearly no headroom --
`2b` with 2.7 ms available, `2d` with 14.4 ms, `5b` with an oracle *slower* than
native. That is what the recorded oracle says, and the recorded oracle is the
best plan **in the action space that was collected**.

Probing the same queries with a wider action space says otherwise. Every one of
the four expensive held-out queries has a plan below the oracle on record:

| query | native | oracle on record | found with a wider search | winning hint |
|---|---|---|---|---|
| `17c` | 21603.7 | 1963.3 | **1043.4** | `Leading(n ci mk mc t cn k)` |
| `17d` | 8822.2 | 2447.9 | **2232.9** | `Set(enable_nestloop off)` |
| `2d` | 586.1 | 571.7 | **463.6** | `SeqScan(mc)` |
| `2b` | 463.6 | 460.9 | **311.0** | `SeqScan(cn)` |

`17c` drops a further **47%**, to 20.7x faster than native rather than 11x. And
`2b`, recorded as having 2.7 ms of headroom, actually has **153 ms** -- a third
of its runtime -- reachable by a single `SeqScan` hint.

Two separate causes, both in the action space rather than the model:

**The join-order sample was 8.** `generate_candidates` defaults to
`max_order_candidates=8`, and a 7-table query has 5040 possible orders, so the
search covered 0.16% of the space. The winning hint for `17c` is a pure
`Leading(...)`, so it was the order count that bound and not a missing family.
Worse, the budget scaled only with PostgreSQL's *cost estimate*: `17c` is priced
at 4409 -- a middling query -- and runs for 21.6 seconds, so the search was
starved on exactly the query whose estimate was most wrong.
`planner.budget_for` scales with join width now as well as cost.

**Scan-method hints were not collected at all.** `INCLUDE_SCAN_METHODS`
defaults off, so `SeqScan(...)` never entered the JOB action space -- and it is
what wins on `2b` and `2d`. `app.autotune` sweeps that family now, so whether it
earns its place is measured per database rather than assumed.

**A third cause, found while fixing the first two**: `app.collect_data` took
`generate_candidates`' bare defaults, so training gathered 8 join orders and no
scan methods while serving planned up to 96 and whatever the flags enabled.
Collection forwards the same flags and the same budget function now, with a test
asserting collection is never narrower than serving.

### Confirmed by execution

Re-collecting `2b` and `2d` under the widened space, reading the oracle back out
of the log:

| query | native | oracle before | oracle after | headroom |
|---|---|---|---|---|
| `2b` | 463.6 | 460.9 | **293.6** | **0.6% -> 36.7%** |
| `2d` | 586.1 | 571.7 | **386.3** | **2.5% -> 34.1%** |

Both won by a single `SeqScan(...)`. A third of each query's runtime was
available the whole time, in a family that was never collected.

So the honest reading of the 99.2% is narrower than it looked: it is 99.2% of a
ceiling that was itself too low, and the real remaining headroom is roughly
**another 900 ms on `17c` alone** plus a third of the runtime of the two queries
that were recorded as unwinnable. The model was doing well against the choices
it was offered; the choices were the constraint.

**The headline numbers above have not been re-measured under the wider space.**
Doing so needs a full re-collection of all 30 queries -- the `17*` family alone
takes twenty minutes each -- and until that runs, 99.2% remains the correct
figure against the old ceiling and an overstatement against the new one.

### What this settles and what it does not

**Settles**: the original finding -- 75% available on real IMDB data and none of
it captured -- no longer holds. On the same class of queries, four independent
policies now reach within 1% of the oracle.

**Does not settle**: eight held-out queries is more than two and still few. And
**the smallest thirty is where this stops** -- JOB goes to 17-table joins and the
candidate sampler has never been pointed at them. The `17*` family is the
expensive end of what was tested, and `17c` alone took twenty minutes to collect
because most of its candidates hit the 120 s statement timeout.

**Consistent with 2.5**: `risk_averse` leading on data-starved JOB is exactly
what that section found, for the reason it gave -- when data is thin relative to
the problem, modelling uncertainty matters more than modelling the target.
`thompson` trailing is the same finding from the other side.

### The partial runs, kept for the record

Trained on the first **14** queries: `pairwise_rank` +39.3% on four held-out
queries averaging 185 ms native -- recorded at the time as explicitly *not*
comparable, because the held-out set was twenty times cheaper than the original's.

Trained on **27**: `pairwise_rank` +99.4% on seven held-out queries averaging
4009 ms native, with three expensive queries carrying all of it. The full
30-query run above is the one to read.

