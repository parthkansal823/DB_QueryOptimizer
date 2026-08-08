# JOB/IMDB stretch goal: results

Generated from a real run, not simulated -- see `data/job/README.md` for the
exact commands. This documents what actually happened, including the parts
that turned out smaller in scope than the synthetic-schema evaluation.

## What was loaded

The real IMDB CSV dump from the JOB paper (May 2013 snapshot), all 21
tables, via `data/job/load_job.sh`:

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

`schema_introspection.discover_table_cardinalities` picked up all 21 tables
automatically (`n_tables_in_schema: 21` in the training output below) --
`optimizer/features.py` never needed to know these table names existed
ahead of time.

## What was run

A deliberate **smoke test, not a full retrain**: 8 real JOB query files
(`3a, 3b, 3c, 2a, 2b, 2c, 2d, 5a`), chosen for being among the smaller/faster
queries in the 113-query set, through the unmodified `app.collect_data` /
`app.train` pipeline (`app.collect_data_job`, `DATABASE_URL` pointed at the
`job` database):

### The first attempt was invalid -- and said so if you read it closely

The initial run produced this:

```json
{
  "n_rows_total": 72, "test_mae_ms": 53.19, "n_tables_in_schema": 21,
  "n_held_out_queries": 2,
  "avg_latency_ms": { "native": 363.27, "heuristic": 363.27, "learned": 363.27 }
}
```

Native, heuristic and learned agreeing to the *second decimal place* is not
a coincidence and not, as an earlier draft of this file claimed, a benign
"the optimizer correctly deferred to Postgres" result. It was the signature
of the `pg_hint_plan` bug documented in `docs/WRITEUP.md` §2.0: hints were
being silently ignored, so every "candidate" was the identical native plan.
Three selectors choosing between eight copies of the same plan will of
course tie exactly.

That reading was available at the time and was missed. Identical numbers
across independent methods should always be treated as a bug signal first
and a finding second.

### Re-run, after the fix

Same 8 queries, same pipeline, with `shared_preload_libraries=pg_hint_plan`
set and the hint hoisted ahead of `EXPLAIN`:

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

**The headroom on JOB is enormous.** Native Postgres averages 3703 ms on
these held-out queries; the best available candidate averages 919 ms. That
is a **75% latency reduction sitting on the table** -- an order of magnitude
more opportunity than the synthetic schema's ~8%. This is exactly what Leis
et al. built JOB to expose: on queries with correlated predicates and many
joins, PostgreSQL's cardinality estimates degrade badly and its chosen plan
is far from optimal. The premise of the whole field is visible in one line
of JSON.

**The model captures none of it.** Every selector returns the native plan,
and the reason is not the earlier hint bug (hints bind here -- the feature
vector is 109 wide across 21 discovered tables, and the oracle differs from
native, which is only possible if candidates differ). It is that a test MAE
of **1657 ms** on queries averaging 3703 ms means the model has essentially
no usable signal. 194 raw executions across 8 queries -- 65 after median
aggregation, 2 held-out queries -- is nowhere near enough to learn a
17-table join space. Faced with no signal, the selectors default to the
candidate that also happens to be the native plan, and the safety veto
would have caught them if they hadn't.

The honest reading: **JOB confirms the opportunity is real and confirms this
system is far from capturing it.** That is a more useful result than a
flattering number would have been, and it quantifies the gap -- 75%
available, 0% captured -- rather than leaving it qualitative.

## Reading this honestly

Two held-out queries is nowhere near enough to claim the model learned
anything JOB-specific in either direction. What this run does establish is
narrower and still worth having: the pipeline runs end to end against a
real, 74.5M-row research benchmark with **zero code changes** -- the
schema-introspection work (§3 of the writeup) means `features.py` discovered
all 21 tables by itself. The scale caveats from `docs/WRITEUP.md` §2.3 apply
here at least as strongly as on the synthetic schema.

## What a real JOB evaluation would need (future work, not attempted here)

- All 113 queries collected (this ran 8), so the query-level test split has
  enough queries to be meaningful (~28 held out at 25%, not 2)
- Multiple repetitions per candidate -- JOB queries against 10M+ row tables
  have more execution-time variance than the synthetic schema's, and this
  run used `--reps 1`
- The join-method stretch goal was disabled for this run
  (`include_join_methods=False` in `app.collect_data_job`) to keep the
  smoke test fast against un-indexed multi-million-row joins; a full run
  would enable it, at significant extra runtime cost
- Indexes: JOB's `fkindexes.sql` was applied, but a real evaluation would
  also want the comparison-with-and-without-indexes experiment
  `data/schema.sql` suggests for the synthetic schema, since it directly
  affects how much headroom a learned optimizer has over the CBO
