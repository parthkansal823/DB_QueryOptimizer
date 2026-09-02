# Techniques backlog — read out of the PostgreSQL manual

Source: `postgresql-18-A4.pdf` (3156 pages), read chapter by chapter. Each entry
says what the technique is, **why it matters for this project specifically**,
where the code would go, and what it costs.

## Read this first: the manual is 18, the server is 16

The PDF documents **PostgreSQL 18**. `docker compose exec postgres psql -c
"select version()"` reports **16.15**, with `pg_hint_plan 1.6.3`. Several things
the manual describes do not exist on the running server, and a few of the most
useful ones are among them. Every entry below is tagged:

- **PG16 ✓** — available now, verified against `pg_settings` on the live server.
- **PG17+ / PG18** — documented in this PDF, **not on this server**. Needs an
  upgrade first, so it is a reason to upgrade rather than a task.

Anything not tagged is version-independent.

---

# Part A — Query optimization

## A1. Operator toggles this project does not use yet — PG16 ✓

`HINT_SETS` in `backend/app/optimizer/hints.py` currently toggles nine GUCs.
The server has **21** `enable_*` settings. The twelve unused ones:

| GUC | default | why it is worth a candidate |
|---|---|---|
| `enable_partitionwise_join` | **off** | Joins matching partitions pairwise instead of appending first. **Off by default**, so no `off` toggle can reach it — this is a *turn-on* action, a shape the action space cannot currently express at all. |
| `enable_partitionwise_aggregate` | **off** | Same, for GROUP BY. Also off by default. |
| `enable_bitmapscan` | on | The middle ground between seq and index scan. Banning it forces the planner to commit either way. |
| `enable_indexonlyscan` | on | Separate from `enable_indexscan`. Turning it off while leaving index scans on isolates the visibility-map decision. |
| `enable_sort` | on | Discourages explicit Sort nodes, pushing the planner toward index order or hash aggregation. |
| `enable_gathermerge` | on | Gather Merge preserves worker sort order at a cost; plain Gather does not. |
| `enable_parallel_hash` | on | Shared hash table across workers vs. one copy per worker — a large memory difference on a big build side. |
| `enable_parallel_append` | on | Spreads workers across child plans instead of marching through them in order. |
| `enable_partition_pruning` | on | Only matters on partitioned tables. |
| `enable_presorted_aggregate` | on | Affects `ORDER BY`/`DISTINCT` inside aggregates. |
| `enable_async_append` | on | Foreign-table appends. |
| `enable_tidscan` | on | Only with a `ctid` predicate. |

**The two partitionwise settings are the interesting ones**, and they are the
only two in the whole list that are off by default. `HINT_SETS` is structured as
"combinations of flags to turn **off**" — a `ENABLE_SETTINGS` tuple is needed
alongside it for flags worth turning **on**. Nothing in the current code can
produce `Set(enable_partitionwise_join on)`.

Cost: small. Caveat: widening the space without retraining sent 17/25 queries
into the safety veto once already (`docs/WRITEUP.md` §2.9). Re-run
`app.collect_data` after.

## A2. Planner cost constants beyond `random_page_cost` — PG16 ✓

`TUNING_SETTINGS` varies `random_page_cost`. §19.7.2 lists eight more, and two
of them are the ones that actually gate parallel plans:

- **`parallel_setup_cost`** (default 1000) and **`parallel_tuple_cost`** (0.1).
  §15.3.5 says outright: *"If a query that is expected to do so does not produce
  a parallel plan, you can try reducing parallel_setup_cost or
  parallel_tuple_cost."* Setting `max_parallel_workers_per_gather` raises a
  *ceiling*; these two change whether the planner wants to go near it. The
  project currently only varies the ceiling.
- **`min_parallel_table_scan_size`** (8MB) / **`min_parallel_index_scan_size`**
  (512kB) — the hard size floor below which parallelism is never considered.
  This is why `Set(max_parallel_workers_per_gather 4)` is a no-op on a small
  table, and the reason family 7 in `famalies.txt` (per-table `Parallel(t N
  hard)`) was added.
- **`effective_cache_size`** (default 4GB) — the planner's assumption about how
  much of the table is already in RAM. Raising it makes index scans look
  cheaper. On a 32GB box with a small database this default is *low*, and it is
  a one-line server change that shifts every plan.
- **`cpu_tuple_cost`** (0.01), `cpu_index_tuple_cost` (0.005),
  `cpu_operator_cost` (0.0025) — raising these relative to the page costs models
  a machine where CPU, not I/O, is the constraint. That is exactly this
  machine: the whole database fits in RAM.
- **`seq_page_cost`** — only the *ratio* to `random_page_cost` matters, so
  varying both together changes how much I/O matters relative to CPU at all.

The manual's own warning is worth keeping: *"there is no well-defined method for
determining ideal values… changing them on the basis of just a few experiments
is very risky."* That is an argument for putting them in the action space and
measuring, which is what this project is for.

## A3. `plan_cache_mode` — generic vs custom plans — PG16 ✓

`force_generic_plan` / `force_custom_plan` / `auto`.

This is a genuinely different axis from everything in `famalies.txt`. A generic
plan is built once, ignoring parameter values, and reused; a custom plan is
built per execution. **This is the parameterized-query problem the project
already has a file for** — `backend/app/query_template.py` gives `city='Mumbai'`
and `city='London'` the same `query_id`, on the assumption one plan serves both.
`plan_cache_mode` is PostgreSQL's own answer to whether that assumption holds,
and the two should be reasoning about the same thing.

Where: a `TUNING_SETTINGS` entry, and a diagnostic in `app/diagnostics.py` for
templates whose members disagree about the best plan.

## A4. `cursor_tuple_fraction` — first-row vs total latency — PG16 ✓

Default 0.1 — the planner assumes a cursor reads 10% of its rows. Lowering it
biases toward fast-start plans.

Worth noting because **the project's whole objective function is total
execution time**. A plan that returns the first row in 5ms and everything in
900ms loses to one that takes 400ms flat, even though a paginated UI would
much prefer the first. Not a bug, but an unstated assumption; `cursor_tuple_fraction`
is the knob that makes it explicit.

## A5. `recursive_worktable_factor` — PG16 ✓

Default 10.0. The planner's guess at how much a recursive CTE grows per step.
Irrelevant to this workload (no recursive queries) — listed so the sweep is
complete.

## A6. `from_collapse_limit` — the other collapse limit — PG16 ✓

`TUNING_SETTINGS` has `join_collapse_limit 1`. §14.3 explains they are a pair:
`join_collapse_limit` flattens explicit `JOIN`s, `from_collapse_limit` flattens
**subqueries**. A query built from a view is exactly the case where the second
one bites and the first does nothing.

Also from §14.3, a fact worth encoding in the guard: **setting either at or
above `geqo_threshold` can trigger GEQO**, whose plans are non-deterministic.
Two identical runs can then produce different plans, which would show up in this
project's own measurements as unexplained variance. Worth a `diagnostics.py`
warning.

## A7. GEQO knobs beyond the threshold — PG16 ✓

`geqo_effort` (1–10, default 5), `geqo_pool_size`, `geqo_generations`,
`geqo_selection_bias`, **`geqo_seed`**.

`geqo_seed` is the interesting one: *"Varying the value changes the set of join
paths explored, and may result in a better or worse best path being found."*
That is a random-restart search — run the same query under several seeds and
keep the best plan. On JOB's 17-table queries, where GEQO is unavoidable, this
is a cheap way to widen the search that costs nothing but planning time.

It is also a **measurement hazard**: with GEQO active and the seed unpinned, the
benchmark's paired A/B is comparing two draws from a random search. Pin
`geqo_seed` in `app/experiment_job.py` or the numbers are noise.

## A8. Index-name and regexp scan hints — implemented, needs wiring

`generate_index_candidates` and `generate_disable_index_candidates` are now in
`hints.py` but need a `{table: [index...]}` map from the catalog.

```sql
SELECT c.relname AS table, i.relname AS index
FROM pg_index x
JOIN pg_class c ON c.oid = x.indrelid
JOIN pg_class i ON i.oid = x.indexrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND c.relkind = 'r';
```

Fetch once per process and cache — the catalog does not change per query.

`DisableIndex` has a second use worth building: it is the **missing half of the
unused-index advisor** in `app/dbadvisor/indexes.py`. That advisor reports zero
scans in `pg_stat_user_indexes`, which is evidence about the past.
`DisableIndex` re-plans the workload as if the index were already dropped, which
is evidence about the future. Dropping an index is irreversible without a
rebuild; this makes the recommendation checkable first.

## A9. Skip scan — PG18

B-tree skip scan: an index on `(four, unique1)` where `four` has 4 distinct
values can serve `WHERE unique1 = 42` by searching once per distinct leading
value. It removes the classic "wrong column order in the composite index"
penalty, and would change what `dbadvisor` should recommend.

**Not on PG16.** A reason to upgrade, not a task.

## A10. `enable_self_join_elimination`, `enable_distinct_reordering`, `enable_group_by_reordering` — PG17/PG18

Three more `enable_*` toggles in the PG18 manual that are **absent from this
server's `pg_settings`** — verified. Do not add them to `HINT_SETS`; a `Set()`
on an unknown GUC is an error, not a no-op.

---

# Part B — Database optimization

## B1. `default_statistics_target` and per-column `SET STATISTICS` — PG16 ✓

Default 100 MCVs and histogram buckets. §14.2.1: *"Raising the limit might allow
more accurate planner estimates, particularly for columns with irregular data
distributions."*

This belongs in `dbadvisor` and is not there. The advisor recommends `CREATE
STATISTICS` for *correlated* columns; this is the fix for a *single* column
whose distribution is too irregular for 100 buckets. They are different failures
with different fixes, and the second is currently invisible.

Detection: a column where the estimate is badly off but no multi-column
correlation explains it. `ALTER TABLE t ALTER COLUMN c SET STATISTICS 1000` is
cheap and reversible.

There is a second effect the manual notes and that matters here: **the statistics
target also sets the ANALYZE sample size, so raising it improves extended
statistics too.** If `CREATE STATISTICS` is recommended and does not close the
gap, raising the target is the next thing to try, not a different object.

## B2. `pg_stat_statements` — PG16 ✓ (extension, needs install)

The single highest-value item in Part B.

- **`queryid`** is PostgreSQL's own normalized query identity — computed from
  the post-parse-analysis tree, not the text. `app/query_template.py` implements
  this by hand. The built-in one is more correct: it distinguishes queries whose
  text is identical but whose meaning differs under a different `search_path`,
  and it merges `IN (1,2,3)` with `IN (1,2,3,4)`. Worth comparing the two and
  either switching or documenting why not.
- **`plans` / `total_plan_time` vs `calls` / `total_exec_time`** — separate
  planning and execution accounting. This project's central cost is
  `optimizer_overhead_ms`, and this measures the baseline it is added to.
- **`shared_blks_hit` / `shared_blks_read`** — the cache-hit ratio per query,
  which is the *cause* of the measurement bias documented in the writeup (the
  "80.5% headroom" that was really 26.0%).
- **`temp_blks_read/written`** — a query spilling to disk. This is `work_mem`
  being too small, measured rather than guessed. `TUNING_SETTINGS` already
  gambles on `work_mem 65536`; this says which queries need it.
- **`wal_records` / `wal_bytes`** — write amplification.
- **`jit_*` columns** (functions, generation/inlining/optimization/emission
  time) — exactly how much JIT cost, which decides whether `Set(jit off)` is
  worth a candidate slot on this workload rather than a guess.
- **`parallel_workers_to_launch` vs `parallel_workers_launched`** — planned
  against actually obtained. §15.2 warns a plan built for 4 workers that gets 0
  can be *worse* than the serial plan. If these two columns disagree on this
  server, every parallel candidate the optimizer picks is being measured under
  conditions the planner did not assume.
- **`min_exec_time` / `max_exec_time` / `stddev_exec_time`** — per-query
  variance, from the server, free. The benchmark harness computes its own by
  re-running queries.

Setup: `shared_preload_libraries = 'pg_stat_statements'`, `compute_query_id =
on`, plus `track_io_timing = on` for the block-time columns. Needs a restart.

## B3. `auto_explain` — PG16 ✓ (extension, bundled)

Logs the plan of any statement over a duration threshold, with
`log_analyze` giving **actual row counts alongside the estimates**.

This is a training-data pipeline the project does not have. Right now
`app.collect_data` executes a fixed synthetic workload. `auto_explain` captures
plans from whatever really runs, with the estimate/actual pairs that
`optimizer/cardinality.py` learns from — no application changes at all.

The settings that matter:
- `auto_explain.log_min_duration = '250ms'` — only slow statements.
- `auto_explain.log_analyze = on` — actual rows. **Read the warning**: per-node
  timing is then instrumented for *every* statement, logged or not, which the
  manual calls *"an extremely negative impact on performance."*
- `auto_explain.log_timing = off` — the stated mitigation. Keeps actual row
  counts, drops per-node timing. **For q-error learning this is exactly the
  right trade** — the corrector needs rows, not times.
- `auto_explain.sample_rate = 0.1` — the other mitigation.
- `auto_explain.log_format = json` — parseable by the existing plan encoder.
- `auto_explain.log_settings = on` — records which planner GUCs were non-default
  when the plan was made. Without it, a logged plan cannot be reproduced.
- `auto_explain.log_buffers = on` — see B4.

## B4. Buffer counts as a stability-corrected objective — PG16 ✓

`EXPLAIN (ANALYZE, BUFFERS)` reports `shared hit` / `read` / `dirtied` /
`written` per node. On PG18 `ANALYZE` turns `BUFFERS` on implicitly; on PG16 it
must be asked for.

**This may be the most useful measurement idea in the manual for this project.**
The writeup records that the same query timed twice differs by ~13%, that a
"80.5% improvement" was really 26.0% once cold-cache bias was removed, and that
the confidence machinery is fighting a noisy target. Wall-clock time mixes the
plan's quality with the cache's state. `shared_read` — pages the query actually
had to fetch — is a property of the plan and the data, and it barely moves
between runs.

Concretely:
- Log `shared_blks_read` and `shared_blks_hit` next to `runtime_ms` in
  `logging_store.py`.
- Use buffer counts as a **tie-breaker** when two candidates' times are inside
  the noise band, instead of `experiment_job.py` reporting *unresolved*.
- Add it as a model feature. It is deterministic where runtime is not, so it
  should reduce the variance the ensemble is trying to model — and the writeup
  already establishes this model's error is bias-dominated, meaning a cleaner
  target helps where more ensemble members do not.

## B5. `pg_prewarm` — the fix for cold-cache bias — PG16 ✓ (extension, bundled)

`pg_prewarm('orders')` loads a relation into shared buffers.

The benchmark harness measures a plan whose tables happen to be cached against
one whose tables are not, and calls the difference an improvement. That produced
the 80.5%-vs-26.0% error, which had to be corrected to the user mid-project.
A prewarm before each measured pair makes both sides start from the same cache
state — which is what "paired" is supposed to mean.

Three modes: `prefetch` (async, OS), `read` (sync, OS), `buffer` (into
PostgreSQL's own cache — the one to use).

Also `pg_prewarm.autoprewarm = on` restores shared buffers after a restart,
which matters because the containers here get restarted often and the first
benchmark after a restart is currently measuring an empty cache.

## B6. `pg_buffercache` — PG16 ✓ (extension, bundled)

Shows what is actually in shared buffers right now. The diagnostic partner to
B5: it can say *why* a measurement was fast, and can confirm a prewarm worked
rather than assuming it. `pg_buffercache_summary()` and
`pg_buffercache_usage_counts()` are the cheap aggregate forms.

## B7. `pg_visibility` — PG16 ✓ (extension, bundled)

The visibility map decides whether an index-only scan can skip the heap. An
`IndexOnlyScan` hint on a table whose VM is stale produces a plan with `Heap
Fetches: N` — an index-only scan doing exactly the heap I/O it exists to avoid.

`pg_visibility_map_summary(rel)` gives all-visible and all-frozen page counts.
Since `IndexOnlyScan` was just added to `SCAN_METHODS`, this is what says
whether that family can work on a given table, and — more usefully for
`dbadvisor` — that a `VACUUM` would make it work. That is a **fix**, not just a
finding: the query gets faster because the map got refreshed.

## B8. `pgstattuple` — measured bloat, not estimated — PG16 ✓ (extension, bundled)

`app/dbadvisor/tables.py` estimates bloat from `pg_stat_user_tables` counters.
`pgstattuple(rel)` measures it: live/dead tuple counts and percentages, free
space. `pgstattuple_approx()` is the cheap version, `pgstatindex()` covers
B-tree fragmentation.

It reads the whole relation, so it is not something to run per request — a
scheduled job, or on demand when the estimate crosses a threshold. But an
advisor recommending `VACUUM FULL` (which takes an `ACCESS EXCLUSIVE` lock)
off an estimate is recommending an outage off a guess.

## B9. Alternative index access methods — PG16 ✓

`dbadvisor` only ever proposes B-tree. The manual documents several others that
fix things B-tree cannot:

- **`pg_trgm`** — trigram GIN/GiST indexes make `LIKE '%foo%'` and `ILIKE`
  indexable. Currently such a predicate is simply a sequential scan with no
  advice offered, which is a *category* of unfixable query the advisor stays
  silent on.
- **`btree_gin` / `btree_gist`** — put scalar columns into a GIN or GiST index
  so a single multi-column index can mix `=` on an int with `@>` on an array or
  a range overlap. Without them those are two indexes and a BitmapAnd.
- **`bloom`** — a bloom-filter index for tables queried on many different
  combinations of columns. One small index instead of a combinatorial pile of
  B-trees. Lossy, so it always needs a recheck.
- **BRIN** (core, no extension) — tiny index for naturally-ordered columns like
  an append-only `created_at`. Orders of magnitude smaller than a B-tree.
  `orders.created_at` in this schema is exactly the shape.
- **Partial indexes** (§11.8) — `WHERE status = 'pending'` on a table that is
  99% completed rows. The advisor recommends whole-column indexes only, and a
  partial index is often a tenth the size for the same benefit.
- **Expression indexes** (§11.7) — `CREATE INDEX ON t (lower(email))`. A
  predicate wrapped in a function is unindexable otherwise, and this is a
  detectable pattern in the query text.
- **Covering indexes / `INCLUDE`** (§11.9) — adds non-key columns so an
  index-only scan can answer the query without the heap. Directly enables the
  `IndexOnlyScan` family. Detection: the query's selected columns are a small
  superset of an existing index's keys.

## B10. Table partitioning — PG16 ✓

Two of the three partition-related planner settings are **off by default**
(`enable_partitionwise_join`, `enable_partitionwise_aggregate`), and
`enable_partition_pruning` only pays on a partitioned table. None of it applies
to the current schema, which has no partitions — so this is a schema-level
recommendation `dbadvisor` could make (a large `orders` table with a
time-ordered key is the textbook case) rather than a planner action.

Related: `constraint_exclusion` (default `partition`) for the older
inheritance-based scheme.

## B11. Maintenance settings that change plan quality — PG16 ✓

- **`maintenance_work_mem`** — speeds `CREATE INDEX` and `ALTER TABLE ADD
  FOREIGN KEY`. Relevant because `dbadvisor` recommends index creation and says
  nothing about how long it will take or how to make it faster.
- **`max_wal_size`** and **`checkpoint_timeout`** — fewer checkpoints during
  bulk load. `data/schema.sql` inserts 50,000 rows on every rebuild.
- **`log_min_duration_statement`**, `log_temp_files`, `log_lock_waits`,
  `log_checkpoints` — server-side observability the advisor could check are
  configured, the way `dbadvisor/server_config.py` already checks other
  settings. `log_temp_files` in particular is a free `work_mem` diagnostic.
- **`track_io_timing`** — required for the per-query I/O time columns in B2.
  Off by default.

## B12. Bulk-load recipe — PG16 ✓

§14.4, applies to `data/schema.sql`: one transaction, `COPY` rather than
`INSERT`, **create indexes after loading not before**, raise
`maintenance_work_mem` and `max_wal_size`, and `ANALYZE` at the end.

The last one is not optional here. The manual: *"With no statistics or obsolete
statistics, the planner might make poor decisions."* This project's entire
premise is the planner's estimates, so a seed script that skips `ANALYZE` is
measuring a planner that has no idea what it is looking at.

## B13. Non-durable settings for the benchmark database — PG16 ✓

§14.5: `fsync=off`, `synchronous_commit=off`, `full_page_writes=off`, unlogged
tables, or the data directory on a RAM disk.

**For the benchmark container only** — it makes crash recovery impossible, and
the compose file now bind-mounts `./data/pgdata`, so this would put real local
data at risk. The reason it is worth considering is that durability overhead is
noise in every measurement here, and this workload is read-only and
regenerable. If it is done, it needs to be a separate compose profile, not a
change to the running one.

---

# Part C — Measurement and infrastructure

## C1. `EXPLAIN (SERIALIZE)` — PG17+

Measures the time spent converting rows to wire format, which `EXPLAIN ANALYZE`
otherwise omits entirely. A plan returning 100,000 wide rows and one returning
100 can look equal in `Execution Time` and be nothing like equal to the client.

Not on PG16.

## C2. `Disabled: true` in EXPLAIN output — PG17+

When a plan uses a node type that was disabled, PG17+ marks it. That is a direct
signal that a `Set(enable_x off)` candidate **did not do anything** — the
planner had no alternative and used it regardless.

The project currently detects this only indirectly, via `plan_fingerprint`
dedup, which catches "same plan came back" but cannot distinguish "the toggle
was ignored" from "the toggle was honoured and the result was identical
anyway".

Not on PG16 — the `Disabled` field is absent. A PG16 approximation: compare the
plan's node types against the disabled set and flag any that still appear.

## C3. `pg_overexplain` — PG18

`EXPLAIN (DEBUG)` exposes `Disabled Nodes` raw counters, `Parallel Safe` per
node, `Plan Node ID`, `extParam`/`allParam`. `EXPLAIN (RANGE_TABLE)` dumps the
range table with RTIs.

`Parallel Safe` per node is the interesting one for `plan_tree.py`'s feature
encoder — it is a structural property of the plan that currently is not encoded
and that determines whether a whole class of alternatives is even reachable.

Not on PG16.

## C4. `Rows Removed by Filter` — PG16 ✓

Already visible in `EXPLAIN ANALYZE`. It is a direct measure of wasted work: a
scan that reads 10,000 rows to emit 40 is a missing index or a missing partial
index, and the number says how much is being thrown away.

`dbadvisor` reasons about scan counts; this is the sharper signal and it is
already in the plan JSON being parsed.

## C5. `Index Searches` — PG18

Per-node count of index searches. Distinguishes a `IN (1,500,700,999)` doing 4
separate index descents from `IN (1,2,3,4)` doing 1 because the values share a
leaf page. Not on PG16.

## C6. Measurement artifacts the manual warns about — PG16 ✓

Worth encoding as guards in `benchmark.py` and `experiment_job.py`, because each
one produces a number that looks like a result and is not:

- **`LIMIT`** — child node costs and row counts are printed as if run to
  completion; the actual values are lower because the parent stopped early. Not
  an estimation error, but it looks exactly like one to any code computing
  q-error from the plan.
- **Merge join** — stops reading one input early, and *re-scans* the inner side
  for duplicate outer keys. `EXPLAIN ANALYZE` counts those repeats as real rows,
  so the inner child's reported actual row count can exceed the relation's size.
  Any q-error computed on that node is meaningless.
- **`BitmapAnd` / `BitmapOr`** — always report actual rows as **zero**, by
  implementation limitation. A q-error walker that divides by actual rows hits a
  division by zero or an infinite ratio here. `optimizer/cardinality.py` walks
  the whole plan tree — this needs an explicit skip.
- **`Subplans Removed: N`** — nodes pruned at execution time do not appear in
  the output at all, so plan-tree features computed from `EXPLAIN ANALYZE` and
  from `EXPLAIN` disagree in structure for the same query.
- **EXPLAIN ANALYZE overhead** — per-node `gettimeofday()` is significant on
  some systems. `pg_test_timing` measures it. If it is large here, every
  training runtime is inflated relative to production.

## C7. Parallel-worker shortfall — PG16 ✓

§15.2: a plan built assuming 4 workers may get 0, and the leader then runs the
whole thing serially. The manual: *"The optimal plan may depend on the number of
workers that are available, so this can result in poor query performance."*

This project **forces parallel candidates** (`max_parallel_workers_per_gather 4`
in `TUNING_SETTINGS`, and now `Parallel(t 4 hard)`). If `max_worker_processes`
or `max_parallel_workers` is exhausted — trivially possible with concurrent
benchmark runs — those candidates get measured under conditions the planner did
not assume, and a real regression gets attributed to the plan.

`pg_stat_statements.parallel_workers_to_launch` vs `parallel_workers_launched`
detects it (B2). A cheaper check: `Workers Planned` vs `Workers Launched` in the
plan JSON, which is already being parsed.

## C8. Nested-loop inner sides are never parallel — PG16 ✓

§15.3.2, and it constrains the action space in a way nothing currently encodes:

- **Nested loop** — inner side is *always* non-parallel.
- **Merge join** — inner side is *always* non-parallel, and if it needs a sort,
  that sort is duplicated in every worker.
- **Hash join** — without parallel hash, every worker builds its own full copy
  of the hash table.

So `NestLoop(...)` and `Parallel(t 4 hard)` in the same candidate are close to
contradictory, and `enable_parallel_hash off` makes every parallel hash join
much worse. A candidate generator that composes families independently will
produce these combinations and pay a planning round trip to learn they are bad.

---

# Ranked, by value per unit of work

| # | item | why first |
|---|---|---|
| 1 | **B4 — buffer counts as objective** | Attacks the noisy-measurement problem at the root, which is what the confidence machinery has been fighting all along. |
| 2 | **B5 — `pg_prewarm` before measurement** | Removes the exact bias that produced the 80.5%-vs-26.0% error. Small, bundled, no restart. |
| 3 | **B2 — `pg_stat_statements`** | Twelve signals for one extension. Also settles whether `query_template.py` should exist. |
| 4 | **C6 — measurement-artifact guards** | Cheap, and each one is a silent wrong number today. The `BitmapAnd` zero-rows case is a live bug in the q-error walker. |
| 5 | **A1 — partitionwise + the other toggles** | The two off-by-default GUCs are a plan shape the action space provably cannot reach. |
| 6 | **A2 — parallel cost constants** | The manual names these as the fix when a parallel plan is not chosen; the project varies only the ceiling. |
| 7 | **B3 — `auto_explain`** | A training-data pipeline for free, with the right settings (`log_timing off`). |
| 8 | **B1 — statistics target** | A whole class of estimation failure `dbadvisor` is currently blind to. |
| 9 | **B9 — non-B-tree index recommendations** | Widens the advisor from one index type to seven. |
| 10 | **A8 — wire up index pinning** | The generators exist; they need a catalog map. |
| 11 | **C7/C8 — parallel guards** | Prevents attributing a worker shortfall to the plan. |
| 12 | **A7 — pin `geqo_seed`** | One line, and without it large-join measurements are partly random. |

Everything tagged PG17+/PG18 (A9, A10, C1, C2, C3, C5) is blocked on a server
upgrade. The `Disabled: true` flag (C2) is the one worth upgrading *for* — it
turns "the candidate did nothing" from an inference into a fact.
