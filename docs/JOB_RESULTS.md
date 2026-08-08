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
