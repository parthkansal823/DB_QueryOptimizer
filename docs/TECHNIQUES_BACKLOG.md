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

## A8. Index-name and regexp scan hints — done, off by default

**Catalog map: done.** `optimizer/catalog.py` has `index_map(cur)` and
`indexes_by_alias(...)`, fetched once per process and cached, because the
catalog does not change per query. The second function exists because
pg_hint_plan addresses relations by the **alias** the query uses, not the table
name -- measured, and the failure is silent: `Parallel(order_items 4 hard)` cost
7870 and was ignored, while `Parallel(oi 4 hard)` cost 1774.

**`generate_index_candidates`: wired, default off.** `planner.py` reads
`INCLUDE_INDEX_PINNING` (default `"0"`) and needs a live cursor; missing either
means the family generates nothing, which is how it behaved before it existed.
It is generated **last**, and `planner.py`'s adaptive budget truncates from the
end -- so on any query that hits the budget this family is the first thing
dropped. That ordering was deliberate when index pinning was the only family
that could be empty for a reason outside the query, but it now doubles as a
second reason the family is unmeasured. Turning it on is a measurement, not an
edit: enable it, run `app.benchmark`, compare captured headroom against the
family ordering that WRITEUP 2.18 established by measurement.

**`generate_index_regexp_candidates`: deliberately not wired.** It takes a
`patterns` tuple that defaults to `()` and returns `[]` without one. With the
catalog map we have the exact index names, and `IndexScan(t idx_a idx_b)` is
strictly more precise than `IndexScanRegexp(t "idx_.*")` -- the regexp form
would generate candidates resolving to plans the explicit form already covers,
spending budget to say the same thing less exactly. It stays for the case where
index names are not knowable ahead of time.

**`DisableIndex`: blocked, and not by us.** The hint does not exist in
pg_hint_plan **1.6.3**, which is what is installed here; the PG18 manual
documents a newer pg_hint_plan than this server runs. It is rejected at `INFO`
severity, which the default `client_min_messages` discards -- so an unsupported
hint looks exactly like a hint that did nothing.
`generate_disable_index_candidates` is kept and unwired.

That blocks the genuinely interesting use: `DisableIndex` is the **missing half
of the unused-index advisor** in `app/dbadvisor/indexes.py`. That advisor reports
zero scans in `pg_stat_user_indexes`, which is evidence about the *past*.
Re-planning the workload as if the index were already dropped is evidence about
the *future*, and dropping an index is irreversible without a rebuild.

There is no clean substitute on 1.6.3. `IndexScan(t <every index except X>)` is
an allow-list, not a deny-list, and it also *forces* an index scan -- so it
cannot express "plan this however you like, but without X", which is exactly the
case that matters most: the one where dropping the index should send the planner
to a sequential scan.

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

---

# Part D — From Chapter 11 (Indexes)

Everything here is about what `app/dbadvisor/indexes.py` recommends, and how.

## D1. `CREATE INDEX CONCURRENTLY` — PG16 ✓

**The advisor currently emits DDL that takes an outage.**

§11.1: *"By default, PostgreSQL allows reads to occur on the table in parallel
with index creation, but writes (INSERT, UPDATE, DELETE) are blocked until the
index build is finished. In production environments this is often
unacceptable."*

Every `CREATE INDEX` the dashboard hands the user blocks writes on the table for
the duration of the build. On `order_items` (500,000 rows) that is not
instantaneous. The advisor's own text says *"Check them before you run them:
`CREATE INDEX` takes locks and disk space"* — which is honest but leaves the
user to discover `CONCURRENTLY` themselves.

The fix is a keyword. The caveats worth stating alongside it: it takes two table
scans so it is slower overall, it cannot run inside a transaction block, and a
failed build leaves an **invalid** index behind that must be dropped. That is
still the right default for a recommendation aimed at a live database.

## D2. Anchored `LIKE` needs `text_pattern_ops` off the C locale — PG16 ✓

§11.2.1: a B-tree index can serve `col LIKE 'foo%'` or `col ~ '^foo'` — *"However,
if your database does not use the C locale you will need to create the index with
a special operator class."*

So `CREATE INDEX ON t (name)` does **not** accelerate `WHERE name LIKE 'Volt%'`
on this database; `CREATE INDEX ON t (name text_pattern_ops)` does. A prefix
match is a common predicate and the advisor would currently recommend an index
that cannot be used for it.

Detection: an anchored LIKE/regex in the query text, plus `SHOW lc_collate` not
being `C`. Both are one line.

Unanchored (`LIKE '%bar%'`) is not indexable by B-tree at all — that is the
`pg_trgm` case in B9.

## D3. Partial indexes do not work with parameterized queries — PG16 ✓

§11.8, and it lands directly on this project: *"Matching takes place at query
planning time, not at run time. As a result, parameterized query clauses do not
work with a partial index. For example a prepared query with a parameter might
specify `x < ?` which will never imply `x < 2`."*

`app/query_template.py` normalizes literals into parameters so that
`city='Mumbai'` and `city='London'` share a `query_id`. A partial index
recommendation derived from the *template* can therefore be unusable by the
actual query, and would show as an index with zero scans — which
`dbadvisor/indexes.py` would then report as unused and recommend dropping.

Two recommendations chasing each other. Any partial-index suggestion has to be
derived from a literal predicate and labelled as only applying when the value is
a constant.

## D4. Do not recommend a fan of partial indexes — PG16 ✓

§11.8, Example 11.4, stated bluntly: a set of non-overlapping partial indexes
(`WHERE category = 1`, `= 2`, …) is *"a bad idea"* — *"the system does not
understand the relationship among the partial indexes, and will laboriously test
each one to see if it's applicable."* One multicolumn `(category, data)` index is
almost always better, and past a certain size the answer is partitioning.

A guard for the advisor: if it is about to recommend more than one partial index
on the same table differing only in the predicate constant, recommend the
multicolumn index instead.

## D5. Covering indexes with `INCLUDE` — PG16 ✓

§11.9. `CREATE INDEX tab_x_y ON tab(x) INCLUDE (y)` puts `y` in the index as
payload so `SELECT y FROM tab WHERE x = 'key'` never touches the heap.

This is what makes the new `IndexOnlyScan` family (family 5 in `famalies.txt`)
usable rather than theoretical. Detection is mechanical: the query's selected
columns are a small superset of an existing index's key columns.

Constraints to encode: B-tree, GiST and SP-GiST only; expressions are not
allowed as included columns; the uniqueness of a `UNIQUE ... INCLUDE` index
applies to the key columns only; and wide payload columns bloat the index and
can exceed the index tuple size limit outright.

And the condition that decides whether it pays at all — *"there is little point
in including payload columns unless the table changes slowly enough that an
index-only scan is likely to not need to access the heap"* — is exactly the
visibility-map question in B7. The two findings should be produced together: a
covering index on a table whose visibility map is stale buys nothing.

## D6. Index sort order — PG16 ✓

§11.4. `ORDER BY x ASC, y DESC` cannot be satisfied by any plain two-column
index; it needs `CREATE INDEX ON t (x ASC, y DESC)`. Also `NULLS FIRST` /
`NULLS LAST`.

Narrow, but entirely invisible to an advisor that only proposes column lists —
and the failure is a full sort the index was supposed to remove.

Related, from §11.4: `ORDER BY ... LIMIT n` is the case where a matching index
wins biggest, because an explicit sort has to process everything to find the
first n rows.

## D7. What an unused index actually costs — PG16 ✓

§11.1: *"Indexes can also prevent the creation of heap-only tuples."*

The advisor reports unused indexes as wasted disk. The larger cost is that every
index on a table blocks the HOT-update optimization for updates touching its
columns, forcing a new index entry and more bloat. That strengthens the
recommendation and is a better reason to give than size.

Also §11.6: *"There's no need to manually create indexes on unique columns;
doing so would just duplicate the automatically-created index."* A primary key
already has a unique B-tree — a recommendation to index a PK column is always
wrong, and worth an explicit guard.

## D8. Index types beyond B-tree, from §11.2 — PG16 ✓

Expands B9 with what the manual says each is actually for:

| type | for | notes |
|---|---|---|
| B-tree | `< <= = >= >`, `BETWEEN`, `IN`, `IS NULL`, anchored `LIKE`, sorted output | the only type that can be `UNIQUE`, and the only one that produces sorted output |
| Hash | `=` only | no range, no sort, no uniqueness |
| GiST | geometric, ranges, nearest-neighbour `ORDER BY location <-> point` | supports index-only scans for some operator classes only |
| SP-GiST | quadtrees, k-d trees, radix trees | also does nearest-neighbour |
| GIN | arrays, jsonb, full-text — values with multiple components | **cannot** do index-only scans: entries hold only part of the value |
| BRIN | columns correlated with physical row order | tiny; `pages_per_range` is the tuning knob |

The nearest-neighbour case is worth noting because no amount of B-tree advice
helps an `ORDER BY distance LIMIT 10` query, and that is a shape an advisor
should be able to name.

## D9. Multicolumn index guidance — PG16 ✓

§11.3: *"equality constraints on leading columns, plus any inequality constraint
on the first column that does not have an equality constraint, will always be
used to limit the portion of the index that is scanned."* Column order is the
whole ballgame, and the advisor proposes column *sets*.

Also: *"Indexes with more than three columns are unlikely to be helpful unless
the usage of the table is extremely stylized"* — a cap worth enforcing.

And §11.5 on the alternative: separate indexes on `x` and `y` can be combined
with `BitmapAnd`, which loses index ordering (so a sort is still needed) and
costs an extra scan. Multicolumn versus index-combination is a genuine trade-off
the advisor currently does not present.

---

# Part E — From Chapter 24 (Routine Maintenance)

## E1. `VACUUM` is a plan-quality tool, not just housekeeping — PG16 ✓

§24.1.1 lists four reasons `VACUUM` has to run, and **two of them are about the
planner**:

2. update the statistics the planner uses;
3. **update the visibility map, which speeds up index-only scans**;

(the other two being space reuse and transaction-ID wraparound).

`dbadvisor/tables.py` currently frames stale statistics and bloat as separate
findings. They have a single cause and a single fix, and reason 3 is the one
nothing in the project mentions at all: an `IndexOnlyScan` candidate (family 5 in
`famalies.txt`) can only pay on a table whose visibility map is current, and
`VACUUM` is what makes it current. That turns "this query could use an
index-only scan but will not" into an actionable finding rather than a curiosity.

## E2. The advisor's `VACUUM FULL` recommendation understates the cost — PG16 ✓

§24.1.2 and its Tip: `VACUUM FULL`, `CLUSTER`, and the table-rewriting `ALTER
TABLE` variants all take an **ACCESS EXCLUSIVE** lock and *"temporarily use extra
disk space approximately equal to the size of the table, since the old copies of
the table and indexes cannot be released until the new ones are complete."*

So the recommendation is: take the table fully offline, and have a spare copy's
worth of disk free. Both belong in the finding text.

The manual's own preference is stronger than the advisor's: *"administrators
should strive to use standard VACUUM and avoid VACUUM FULL"*, and *"the usual
goal of routine vacuuming is to do standard VACUUMs often enough to avoid needing
VACUUM FULL."* If bloat is high enough to want `VACUUM FULL`, the real finding is
usually that autovacuum is not keeping up — which is a settings fix, not a
one-off outage.

Also §24.1.2: a table whose entire contents are periodically deleted should use
`TRUNCATE`, which reclaims the space immediately with no `VACUUM` at all.

## E3. Autovacuum never ANALYZEs three things — PG16 ✓

Three explicit gaps in §24.1.3, each one a table whose statistics silently rot:

- **Partitioned tables.** *"Tuples changed in partitions and inheritance children
  do not trigger analyze on the parent table. If the parent table is empty or
  rarely changed, it may never be processed by autovacuum."*
- **Inheritance parents** — same mechanism.
- **Foreign tables** — *"the autovacuum daemon does not issue ANALYZE commands
  for foreign tables, since it has no means of determining how often that might
  be useful."*

A parent table that has never been analyzed is a planner with no idea what it is
looking at, on the largest queries in the system. Detection is one query against
`pg_stat_user_tables.last_analyze IS NULL` restricted to `relkind = 'p'` and
foreign tables — cheap, and a class of finding `dbadvisor` produces none of
today.

## E4. Expression statistics for function selectivity — PG16 ✓

§24.1.3, Tip: *"by default there is limited information available about the
selectivity of functions. However, if you create a statistics object or an
expression index that uses a function call, useful statistics will be gathered
about the function."*

A **third** use of `CREATE STATISTICS` the project does not exploit.
`dbadvisor/extended_stats.py` proposes multi-column objects for correlated
columns; this is a single-expression object for a predicate the planner is
guessing at:

```sql
CREATE STATISTICS ON lower(email) FROM users;
```

`WHERE lower(email) = ...` currently gets a hardcoded default selectivity. This
makes it measured. Detection is the same pattern as the expression-index
recommendation in D8, and the two should be offered together — the statistics
object is the cheaper half and does not carry write-time cost.

## E5. Transaction-ID wraparound — PG16 ✓

§24.1.5. Nothing in this project looks at it, and it is the one database problem
here that ends in the server refusing to accept writes:

```sql
SELECT c.oid::regclass AS table_name,
       greatest(age(c.relfrozenxid), age(t.relfrozenxid)) AS age
FROM pg_class c
LEFT JOIN pg_class t ON c.reltoastrelid = t.oid
WHERE c.relkind IN ('r', 'm')
ORDER BY age DESC;

SELECT datname, age(datfrozenxid) FROM pg_database;
```

Against `autovacuum_freeze_max_age` (default 200 million) this is a percentage
of the way to a forced anti-wraparound vacuum; against 2 billion it is a
percentage of the way to an outage. The escalation is documented: a `WARNING` at
40 million transactions remaining, then `ERROR: database is not accepting
commands` at 3 million.

A single-row check with a clear threshold and a clear fix — squarely what
`dbadvisor/server_config.py` is for.

## E6. Freeze settings for append-only tables — PG16 ✓

§24.1.5: for **static tables, including tables that receive inserts but no
updates or deletes**, there is no space to reclaim, so the only thing forcing an
autovacuum is the freeze horizon. Raising `autovacuum_freeze_max_age` (or
lowering `vacuum_freeze_min_age`) maximises the interval between those forced
scans.

`orders` and `order_items` in this schema are append-only, and 500,000 rows is
large enough for a full-table freeze scan to be noticeable — including, if it
lands mid-benchmark, as unexplained variance in a measurement.

The cost is bounded and stated: `pg_xact` grows to about half a gigabyte at the
maximum setting of two billion, `pg_commit_ts` to about 20GB if
`track_commit_timestamp` is on. The manual's own advice: *"If this is trivial
compared to your total database size, setting autovacuum_freeze_max_age to its
maximum allowed value is recommended."*

Related knobs worth surfacing: `vacuum_freeze_table_age` (capped at 0.95 ×
`autovacuum_freeze_max_age`), and `vacuum_max_eager_freeze_failure_rate` for
eager freezing.

## E7. `log_autovacuum_min_duration` — PG16 ✓

Default 10 minutes, so most autovacuum activity is invisible. Setting it lower
records what ran, on which table, and how long it took — which is how you find
out that a benchmark run overlapped an aggressive vacuum on `order_items`.

`VACUUM VERBOSE` reports the same detail on demand, including how
`relfrozenxid` advanced and how many pages were newly frozen.

---

# Reading status

| chapter / section | pages | state |
|---|---|---|
| 11. Indexes | 455–471 | read in full → Part D |
| 14. Performance Tips | 529–556 | read in full → Parts A, C |
| 15. Parallel Query | 556–562 | read in full → C7, C8 |
| 19.7 Query Planning (all GUCs) | 671–679 | read in full → A1–A7 |
| 24. Routine Maintenance | 789–801 | read in full → Part E |
| F.3 `auto_explain` | 2828–2831 | read in full → B3 |
| F.25 `pg_buffercache` | 2932–2937 | read → B6 |
| F.29 `pg_overexplain` | 2953–2955 | read in full → C3 |
| F.30 `pg_prewarm` | 2955–2957 | read in full → B5 |
| F.32 `pg_stat_statements` | 2959–2969 | read → B2 |
| F.36 `pg_visibility` | 2982–2984 | read in full → B7 |
| 27. Monitoring (`pg_stat_*` views) | 838–902 | read → F5, F6, F7 |
| 5.12 Partitioning | ~100–112 | **not yet read** (B10 is from §19.7 only) |
| 30. JIT | 946–950 | **not yet read** (A2 covers `jit_above_cost`) |
| 61. GEQO internals | — | **not yet read** (A7 is from §19.7) |
| 69. How the Planner Uses Statistics | — | read in full → F1–F4 |
| VI. Reference — EXPLAIN | 1998–2000 | read in full → C9 |
| VI. Reference — VACUUM, CREATE INDEX/STATISTICS | 1604+ | **not yet read** |

Chapters 27 and 69 were the two flagged as most likely to add something, and
both did -- 69 gave F1-F4 (the selectivity formulas `optimizer/cardinality.py`
is modelling, and the fact that join size is estimated before any join plan
exists) and 27 gave F5-F7 (what `idx_scan` actually counts, which is not
queries, and that the statistics views report asynchronously).

The **EXPLAIN reference page** was the third flagged, and it gave C9: of the
twelve options it documents, this project passes three, and `SETTINGS` is a
PG16-available readback for whether a `Set()` hint actually bound -- the failure
mode that has cost this project the most time.

Still unread: **5.12** (partitioning; B10 comes from 19.7 only), **30** (JIT;
A2 already covers `jit_above_cost`), **61** (GEQO internals; A7 comes from
19.7), and the VACUUM / CREATE INDEX / CREATE STATISTICS reference pages.

The rest of the 3156 pages — client interfaces, procedural languages,
replication, data types, SQL syntax reference — was swept for planner-relevant
material and has none that bears on this project.

## C9. `EXPLAIN (SETTINGS)` — the direct test that a `Set()` hint bound — PG16 ✓

From the EXPLAIN reference page (manual p. 1999), checked against every EXPLAIN
this project issues:

| where | options |
|---|---|
| `plan_extractor.get_plan` — the main path | `FORMAT JSON`, plus `ANALYZE, BUFFERS` when analyzing |
| `main.py` `/query/analyze` | `FORMAT JSON, VERBOSE` |

Against the full option list — `ANALYZE, VERBOSE, COSTS, SETTINGS,
GENERIC_PLAN, BUFFERS, SERIALIZE, WAL, TIMING, SUMMARY, MEMORY, FORMAT` —
`SERIALIZE` (C1) and `MEMORY` are PG17+, `WAL` is irrelevant to a read-only
`SELECT` workload, `COSTS`/`TIMING`/`SUMMARY` default to what is wanted, and
`VERBOSE` is already used where it earns its cost. One remaining option is worth
having.

> **SETTINGS** — Include information on configuration parameters. Specifically,
> include options affecting query planning **with value different from the
> built-in default**. This parameter defaults to FALSE.

Available since PG12, so it works on this server.

**Why it matters here specifically.** The single most expensive recurring bug
class in this project is the *silent* one: a hint that was ignored is
indistinguishable from a hint that ran and changed nothing. It has now happened
three separate ways —

* `Parallel(order_items 4 hard)` addressed a table by name where the query used
  the alias `oi`. Cost 7870, silently unchanged; `Parallel(oi 4 hard)` cost 1774.
* `DisableIndex` does not exist in pg_hint_plan 1.6.3. Rejected at `INFO`
  severity, which the default `client_min_messages` discards.
* Stacked hint comments: pg_hint_plan reads only the **first** `/*+ ... */` and
  discards the rest without complaint (fixed by merging in `apply_hint`).

A large slice of the action space is `Set(...)` hints —
`hints.ENABLE_SETTINGS`, `EXTENDED_HINT_SETS`, `COST_MODEL_SETTINGS`, and the
resource/cost-model family that WRITEUP 2.18 measured as the most valuable of
the three. Every one of them is exposed to exactly this failure, and nothing
currently checks that the GUC actually took effect. `EXPLAIN (SETTINGS)` reports
the planner GUCs that differ from their built-in defaults at plan time, which is
a direct readback of whether the hint bound — no severity juggling, no cost
comparison against a control, in the plan JSON the project already parses.

**The check this buys.** For a candidate whose hint is `Set(enable_seqscan off)`,
the plan's `Settings` object should contain `enable_seqscan: off`. If it does
not, the hint did not bind, and the candidate is not what it claims to be. That
turns an entire silent failure class into an assertion — and it can run in
`prefilter.py`, which already exists to discard candidates that provably cannot
differ from the baseline.

**Caveat to verify before relying on it.** pg_hint_plan applies `Set()` for the
duration of planning. Whether that is still visible to `EXPLAIN (SETTINGS)` at
output time is a question about pg_hint_plan's implementation, not about the
manual, and it must be measured before this is built on. If it turns out not to
be visible, that is itself worth recording: it would mean there is no readback
for `Set()` binding at all, and the fallback is a control comparison.

Two others from the same page, noted and not pursued:

* **`VERBOSE`** — already in use, and correctly scoped. `main.py:377` passes it
  on `/query/analyze` because covering-index opportunities need the query's
  output column list, which only `VERBOSE` reports; the comment there notes it
  is one extra planning round trip that the analyze path can afford (it already
  issues around thirty) and `/query/optimize` cannot. It also *"always label[s]
  variables in expressions with their range table alias"* — the information the
  `Parallel(oi ...)` bug turned on — but `catalog.py::indexes_by_alias` already
  gets that from `scan_relations` without the round trip, and `VERBOSE` would
  roughly double the size of every plan JSON stored in `plan_execution_log`.
* **`GENERIC_PLAN`** (PG16) — plans a statement containing `$1` placeholders
  without values. It pairs with A3 (`plan_cache_mode`) and would let the advisor
  show how a prepared statement plans versus how the literal-value query plans.
  Cannot be combined with `ANALYZE`, so it is a second EXPLAIN per query, and
  nothing in this workload is currently parameterized.


---

# Implementation status

## Built and verified

| item | where | evidence |
|---|---|---|
| **C6** — q-error artifacts | `dbadvisor/extended_stats.py::_plan_qerror` | A nested-loop probe reported **124931**; true worst was **2.0**. Three guards added: no loop multiplication, skip `BitmapAnd`/`BitmapOr`, skip subtrees under a `Limit`. `test_qerror_artifacts.py` |
| **A1** — six more operator toggles | `hints.EXTENDED_HINT_SETS` | All accepted by the server; all no-ops on the sample join, which is why they are opt-in |
| **A1** — partitionwise (on-toggles) | `hints.ENABLE_SETTINGS` | The only two planner features that ship off, so no `off` toggle could reach them |
| **A2** — cost-model constants | `hints.COST_MODEL_SETTINGS` | `seq_page_cost 0.5` → **6422.7** from 7870.2; `min_parallel_table_scan_size 0` → 6637.5; `parallel_setup_cost 0` → 6870.2 |
| **A6** — `from_collapse_limit` | `hints.COST_MODEL_SETTINGS` | Accepted; no-op on a query with no subqueries |
| **B7 / E1** — visibility map | `dbadvisor/maintenance.py::index_only_scan_readiness` | Fired on a never-vacuumed 500-page table at 0% all-visible |
| **E3** — autovacuum ANALYZE gaps | `dbadvisor/maintenance.py::never_analyzed` | Fired on a real partitioned parent |
| **E5** — XID wraparound | `dbadvisor/maintenance.py::wraparound_risk` | Thresholds unit-tested; cannot be forced live without 120M transactions |
| **D7** — HOT-update pressure | `dbadvisor/maintenance.py::hot_update_pressure` | 15,000 updates to an indexed column → **0 HOT updates** |
| **C7** — parallel shortfall | `plan_extractor.py::parallel_shortfall` | With `max_parallel_workers = 0`: 4 planned, **0 launched**, correctly flagged |
| **§69.2.3** — measured correlation | `dbadvisor/extended_stats.py::measured_correlation` | `('SG','Singapore')` occurs **22×** more often than per-column statistics predict — read from `pg_mcv_list_items`, no probe query |

## Corrections to earlier entries in this document

- **D1 was wrong.** `advisor.py:162` and `:209` already emit `CREATE INDEX
  CONCURRENTLY`. The only plain `CREATE INDEX` left is in `hypothetical.py`,
  where the index is never actually built — correct as it stands.
- **C6 named the wrong file.** `optimizer/cardinality.py` was never affected: it
  gates on `"Relation Name" in plan`, which excludes `BitmapAnd`/`BitmapOr`, and
  its comment shows the loops bug was already found and fixed there. The live bug
  was the *second copy* in `dbadvisor/extended_stats.py`, which never got that
  fix.
- **A3 (`plan_cache_mode`) turned out to be unusable as a candidate.** Accepted
  by pg_hint_plan, and measured at exactly the baseline cost for both values: it
  "is considered when a cached plan is to be executed, not when it is prepared",
  and this project plans ad-hoc statements. Removed from the candidate tuple with
  the reason recorded in `hints.py`. It still belongs as a *diagnostic* for
  templates whose members disagree about the best plan.

## New findings from Chapters 69 and 27, not in Parts A–E

### F1. The exact selectivity formulas — §69.1

Worth having written down, because `optimizer/cardinality.py` is learning a
correction to these and they are not guesses:

| clause | estimate |
|---|---|
| range (`x < k`) | histogram bucket interpolation, MCVs excluded from the histogram |
| equality, value in MCV list | `most_common_freqs[i]` — exact |
| equality, value not in MCV | `(1 - sum(mcv_freqs)) / (n_distinct - num_mcv)` |
| `A AND B` | `sel(A) × sel(B)` — **the independence assumption** |
| join `a.x = b.x` | `(1 - null_frac1)(1 - null_frac2) / max(rows1, rows2)` |

Two consequences:

- An equality predicate on a **non-MCV** value depends entirely on `n_distinct`,
  which is sampled and famously unreliable. That is the concrete mechanism
  behind B1 (raising `default_statistics_target` / per-column `SET STATISTICS`) —
  more MCVs means more predicates land in the exact branch instead of the
  divided-guess branch.
- `n_distinct` is **negative when it is a fraction** of the row count (`-1` means
  unique, `-0.568` means 56.8% distinct). Any code reading `pg_stats.n_distinct`
  as a count is wrong on exactly the columns that matter most.

### F2. Join size is estimated before any join plan exists — §69.1

*"the join relation size is estimated before any particular join plan has been
considered."*

This is the justification for how `optimizer/cardinality.py` keys its `Rows()`
corrections — by **relation set**, not by plan node. The estimate is a property
of which tables are joined, not of how. It also explains why `EXPLAIN` can show
`50 × 1` at a nested loop and a join estimate that is not their product.

### F3. Which kind of extended statistics to recommend — §69.2

`dbadvisor/extended_stats.py` requests all three kinds on every recommendation.
Defensible, but the manual gives each a distinct trigger and a distinct cost:

| kind | fixes | limits |
|---|---|---|
| `dependencies` | `A AND B` equality underestimates | **equality only**; assumes the values are compatible, so `city='SF' AND zip='90210'` is estimated as if it matched |
| `ndistinct` | `GROUP BY a, b` group-count estimates | only worth it on columns actually grouped — "otherwise the ANALYZE cycles are just wasted" |
| `mcv` | both, plus **range clauses**, and knows which value combinations are impossible | most expensive in ANALYZE, storage **and planning time** |

The two things `mcv` does that `dependencies` cannot: `a = 1 AND b = 10` on
perfectly-correlated columns is correctly estimated at ~0 rows, and `a <= 49 AND
b > 49` is handled at all.

### F4. Statistics silently unavailable on a permissions boundary — §69.3

If the current user lacks `SELECT` on the table or columns and the operator is
not `LEAKPROOF`, *"the selectivity estimator will behave as if no statistics are
available"*, and *"There is no direct feedback about that, except that the plan
might be suboptimal."*

A silent, permissions-dependent degradation to default estimates — worth a
`diagnostics.py` check whenever the optimizer runs as a different role than the
application.

### F5. `idx_scan` is not a count of queries — §27.2.20

Three corrections to how `dbadvisor/indexes.py` reads index usage:

- *"Index scans may sometimes perform multiple index searches per execution.
  Each index search increments `idx_scan`"* — an `IN (...)` list, an
  `OR`-to-array rewrite, or a skip scan all inflate it. `idx_scan` is not
  comparable to a query count.
- A bitmap scan increments `idx_tup_read` for the index and `idx_tup_fetch` for
  the **table**, but **not** `idx_tup_fetch` for the index. So a heavily-used
  index can show `idx_tup_fetch = 0`.
- *"The optimizer also accesses indexes to check for supplied constants whose
  values are outside the recorded range"* — the planner itself touches indexes,
  so a nonzero counter does not prove a query used one.

`last_idx_scan` and `last_seq_scan` (PG16+) are the better signal: a timestamp
survives reasoning that a counter reset destroys.

### F6. `pg_stat_all_tables` columns worth using — §27.2.19

- `n_mod_since_analyze` — the exact ANALYZE-staleness number (`tables.py` uses
  this already).
- `n_ins_since_vacuum` — insert-only tables need vacuum for the visibility map
  and the freeze horizon, tracked separately from updates and deletes.
- `n_tup_newpage_upd` — updates whose new version landed on a different heap
  page. Distinguishes "an indexed column changed" from "the page was full", and
  the second is fixed by lowering `fillfactor`.
- `total_vacuum_time` / `total_autovacuum_time` / `total_analyze_time` — how
  much wall-clock maintenance is actually costing, which is the missing half of
  every "just run VACUUM" recommendation.

### F7. The statistics views report asynchronously — measured

`pg_stat_user_tables` showed `(0, 0, 0)` for a table that had just taken 15,000
committed updates in the same session, and `pg_stat_clear_snapshot()` did not
change it. A second connection a moment later showed `(15000, 0, 10000)`.

Any advisor check that reads a counter it has just caused will read zero. That
is why `index_only_scan_readiness` sizes tables from `pg_class.relpages` rather
than `n_live_tup` — one is written by VACUUM, the other by the cumulative
statistics system, and only the first is immediately visible.

---

# Would C++ help? — measured, not argued

The question came up, and it is answerable rather than a matter of taste.
Profiling one four-table query through the full serving path, 20 runs, median:

| stage | ms | share | what is actually running |
|---|---|---|---|
| 1. baseline `EXPLAIN` | 0.57 | 2% | PostgreSQL + libpq |
| 2. generate candidate hints | 0.66 | 3% | **Python** |
| 3. `EXPLAIN` × 31 candidates | 19.13 | **75%** | PostgreSQL + libpq |
| 4. features + model inference | 5.16 | 20% | Python → **LightGBM (already C++)** |
| **total** | **25.52** | | 32 round trips to PostgreSQL |

**77% of the time is PostgreSQL planning and network round trips.** No language
change touches that number — it is the server doing work in C, plus 32 trips
across a socket.

Of the 23% that is Python-side, most is stage 4, and the model there is
LightGBM, whose inference is already compiled C++ behind a thin binding. The
part that is genuinely interpreted Python is the plan-tree walk that builds the
feature vector, plus stage 2's string assembly at **0.66 ms — 3% of the total**.

So the ceiling on rewriting every line of Python in this project is somewhere
under 23%, and realistically far less. Meanwhile the 75% sits untouched.

## Where the real win is, and it is not a language

**Fewer round trips.** 31 candidates produced **12 distinct plans** — 19 of the
32 trips bought a plan already seen. `plan_fingerprint` deduplicates, but only
*after* paying for the EXPLAIN. Predicting which hints collapse to the same plan
before planning them is worth more than any rewrite, and it is arithmetic, not a
language.

**Or move inside the planner.** The version of this that eliminates round trips
entirely is a PostgreSQL extension using the `planner_hook`, the way
`pg_hint_plan` itself works and the way Bao's production form works — choose
among plans while already inside the planner, with the paths in hand and no
socket involved.

That is worth knowing precisely because it is the one place a compiled language
genuinely enters: **PostgreSQL's extension API is C, not C++.** Extensions are
built against `postgres.h` with C linkage, and the backend is not exception-safe
in the way C++ assumes — `elog(ERROR)` unwinds with `longjmp`, which walks
straight past C++ destructors. Extensions that do use C++ have to wrap every
call at a C boundary and catch everything before it crosses back.

So "use C++" and "make this fast" point in different directions. The fast path is
a C extension, and the honest current answer is that the Python is not what is
slow.

## What is worth doing first, in order

1. **Predict duplicate plans before planning them.** 19 of 32 round trips here
   produced nothing. Pure algorithm.
2. **Tighten the adaptive budget.** `ADAPTIVE_CANDIDATE_BUDGET` already scales
   candidates to estimated cost; 31 candidates for a query that plans in 0.57 ms
   is the case it exists to prevent, and it did not.
3. **Cache aggressively.** `decision_cache.py` exists for repeat traffic and
   costs nothing on a hit.
4. **Then, if it still matters, the planner hook** — in C.

Rewriting the 3% is the last thing on this list, not the first.

---

# Round two — round trips and index type

## The prefilter: dropping candidates before paying for them

Profiling put **75%** of the optimizer's latency in EXPLAIN round trips, and
found 31 candidates producing 12 distinct plans. `plan_fingerprint` catches
duplicates *after* the round trip. `optimizer/prefilter.py` catches a subset of
them before.

**The argument.** PostgreSQL does not remove a disabled operator; it prices it at
`disable_cost` and lets the arithmetic bury it. So if the baseline plan contains
no node of the type being disabled, its cost is unchanged by the ban — and it
was already the cheapest, so it stays the cheapest. `Set(enable_mergejoin off)`
on a plan with no Merge Join cannot do anything.

**Verified, not argued.** A harness planned every candidate the prefilter
dropped, across three query shapes, and compared against the baseline
fingerprint and cost. **It caught two real bugs in the first version:**

1. `Set(enable_nestloop off) Set(enable_mergejoin off)` was dropped because the
   *second* flag was a no-op — while the first took a 7870 plan to 12234.
   Reasoning per-flag and returning on the first match is wrong: a combination
   candidate is precisely where the flags are meant to interact.
2. `Set(enable_hashagg off)` was dropped on a plan that was hash aggregating.
   EXPLAIN's **JSON** writes `"Node Type": "Aggregate"` with a separate
   `"Strategy": "Hashed"`; `HashAggregate` is the *text* format's name and
   matches nothing. Measured 492.05 against a real 1970.82.

A third edge case came from the unit tests: an **empty** plan was being read as
"contains no Merge Join", so a stub baseline stripped candidates on no
information at all. Absence of evidence is not evidence of absence, and
`candidate_hints` is reachable from paths that pass a stored or synthetic plan.

After the fixes: **PASS, 0 wrongly dropped** on all three shapes.

**Effect on the serving path**, same query, before and after:

| | before | after |
|---|---|---|
| round trips | 32 | **28** |
| total | 25.52 ms | **21.01 ms** |
| distinct plans found | 12 | **12** |

17.7% faster with no loss of coverage. Savings are 8–9% on joins and **31%** on
a single-table aggregate, where most operator toggles cannot apply.

**What it does not reach.** 56 of the 83 kept candidates on the four-table join
were still duplicates. Those are join-order collapses, and whether a different
`Leading()` produces the same plan is exactly what the planner is for — there is
no cheap way to know without asking. The prefilter takes the provable slice and
leaves the rest honestly on the table.

Two existing tests broke, both asserting `served == generated`. They were
rewritten to assert something stronger: every candidate missing from the served
space must come with a proof from `is_provable_noop`. A family that vanished for
any other reason still fails the test.

## Index type: the advisor was recommending indexes the planner ignores

`advisor.py` found a wasteful sequential scan and recommended `CREATE INDEX ON t
(col)` for every predicate shape. For three shapes that index gets built, costs
disk and write amplification, and is **never used** — so the next advisor run
reports it as unused and suggests dropping it. Two recommendations chasing each
other.

Measured against the live planner, building both the old and new recommendation:

| predicate | naive `CREATE INDEX ON t (col)` | recommended |
|---|---|---|
| `name LIKE 'Volt%'` | Seq Scan — **ignored** | Index Scan on `text_pattern_ops` |
| `name LIKE '%olti%'` | Seq Scan — **ignored** | Bitmap Heap Scan, trigram GIN |
| `lower(city) = 'x'` | Seq Scan — **ignored** | Bitmap Heap Scan, expression index |

Why each (§11.2.1, §11.7):

- **Anchored pattern.** A B-tree can serve it *"However, if your database does
  not use the C locale you will need to create the index with a special operator
  class."* This database is `en_US.utf8`. `SHOW lc_collate` errors on PG16, so
  the locale is read from `pg_database.datcollate`.
- **Unanchored pattern, and all ILIKE.** No prefix to descend on — not indexable
  by B-tree at all. This was a category the advisor had no answer for; the answer
  is `pg_trgm`.
- **Expression predicate.** An index on `city` cannot serve `lower(city)`. The
  recommendation also offers the cheaper half from §24.1.3: `CREATE STATISTICS ON
  lower(city)` fixes the planner's selectivity with no write-time cost.

Every specialised recommendation now carries its cost — GIN is larger and slower
to write, `text_pattern_ops` does not serve `ORDER BY`, expression indexes are
recomputed per write.

**And one thing no longer recommended at all.** §11.6: *"There's no need to
manually create indexes on unique columns; doing so would just duplicate the
automatically-created index."* Single columns already carrying a unique index —
every primary key — are now excluded.

## Unused-index findings argue from write cost, not disk

`last_idx_scan` (PG16+) is a timestamp, and survives reasoning a reset counter
destroys: "never scanned while the table was read 37,383,054 times" is a
different claim from "the counter reads zero".

And the cost is stated properly. §11.1: *"Indexes can also prevent the creation
of heap-only tuples."* Where `n_tup_upd` against `n_tup_hot_upd` shows most
updates are not heap-only, the finding now says how many writes had to touch
every index — a recurring cost, against disk which is a fixed one.

## Two more corrections to this document

- **E2 was wrong.** Nothing in the project recommends `VACUUM FULL`;
  `tables.py` already emits the non-blocking `VACUUM (ANALYZE)`.
- **D1 was wrong** (already noted above): `CREATE INDEX CONCURRENTLY` was
  already in use.

Both were written from reading the manual rather than the code. The pattern is
worth naming: a backlog entry derived from documentation is a hypothesis, and
three of them so far have not survived contact with the repository.

---

# Round three — pg_stat_statements, and a limitation worth more than the feature

## Enabled

`docker-compose.yml` now preloads `pg_stat_statements` alongside `pg_hint_plan`,
with `compute_query_id=on` and `track_io_timing=on`. `CREATE EXTENSION` alone
gives a view with nothing in it -- the module hooks the executor, so it has to be
preloaded, which is a server restart.

`backend/app/dbadvisor/query_stats.py` reads five things out of it:
planning time separated from execution, per-query cache hit ratio, spilling,
JIT cost, and runtime variance -- all across the whole workload rather than the
one query someone pasted in, at no measurement cost.

## The limitation that turned out to matter more

**`pg_stat_statements` cannot see this project's action space.** Measured
directly on the running server:

```
/*+ Set(enable_hashjoin off) */  SELECT count(*) FROM orders o JOIN users u ...
/*+ Set(enable_mergejoin off) */ SELECT count(*) FROM orders o JOIN users u ...
                                 SELECT count(*) FROM orders o JOIN users u ...
```

Three statements → **one row**, `calls = 3`, one `queryid`, and a stored `query`
text showing whichever hint arrived first.

It normalises to the post-parse-analysis tree, and a `pg_hint_plan` hint is a
*comment*. So every candidate the optimizer plans for a query folds into a single
entry, and three readings mean something other than they appear to:

- `stddev_exec_time` is variance **across plans**, not instability of one plan.
- `temp_blks_written` is the **sum over every candidate**. Checked: the
  four-table query reports thousands of temp blocks in this view and writes
  **zero** when planned natively under `EXPLAIN (ANALYZE, BUFFERS)`. Reporting
  "this query spills" would have been wrong.
- The stored query text is actively misleading about what was measured.

Every finding now states its attribution, and the dashboard panel repeats the
caveat. The findings are still worth having -- "some plan for this query wrote
118,088 temp blocks" is a real thing to know about the action space -- but only
with the qualifier attached.

Also flagged rather than filtered: `from_optimizer_sweep`. On a dashboard-driven
database the candidate sweep *is* most of the workload, and hiding that would
misrepresent the panel.

## Version mismatches, three and four

The PG18 manual documents `parallel_workers_to_launch` /
`parallel_workers_launched`, which would have answered §15.2's worker-shortfall
question straight from the view. **Not in `pg_stat_statements` 1.10.** And
`shared_blk_read_time` is `blk_read_time` here.

The column list is read from `information_schema` so a rename degrades instead of
erroring, and `plan_extractor.parallel_shortfall` answers the worker question
from the plan instead.

That makes four documentation-versus-installed mismatches: `DisableIndex`,
`pg_overexplain`, and these two. Same lesson each time — the manual describes a
version, the server is the authority.

## Dashboard

`DatabaseAdvisor.jsx` now renders two sections that had no UI at all:
**Maintenance** (never-analyzed parents, wraparound risk, HOT-update pressure,
visibility-map readiness) and **Across the whole workload**. Nine new component
tests, including one asserting the panel explains *why* it is empty when
`pg_stat_statements` is absent -- an empty panel and "not installed" are
different answers.

## Totals

814 backend tests, 82 frontend tests, all passing. Live `/dbadvisor` returns
five sections.

## Still open

- **D5** covering indexes with `INCLUDE` -- needs the query's select list, which
  `analyze_plan` does not currently receive.
- **B8** `pgstattuple` for measured rather than estimated bloat.
- **B9** BRIN for physically-correlated columns (trigram and expression indexes
  are done).
- **F3** picking `dependencies` vs `mcv` vs `ndistinct` per trigger rather than
  requesting all three.
- **F4** the statistics-permissions diagnostic from §69.3.
- **A7** pinning `geqo_seed` in the benchmark harness.
- The remaining ~67% of duplicate candidates, which are join-order collapses and
  need a cheaper way to predict than planning them.

---

# Round four — the rest of the list

Everything from the "still open" list, except the one item that is genuinely
hard. Each verified against the live server rather than reasoned about.

## B9 — BRIN, and the check that stops it being wrong

`dbadvisor/physical.py::brin_candidates`. §11.2.6: BRIN is *"most effective for
columns whose values are well-correlated with the physical order of the table
rows"*, and `pg_stats.correlation` is exactly the statistic that decides it.

**On this project's own schema it correctly proposes nothing.**
`orders.created_at` has a correlation of **0.002** — the seed script inserts
random dates, not ordered ones — so BRIN would prune nothing here.

Proved it fires when it should, on a 1.2M-row append-only table with an ordered
timestamp, and measured the size claim rather than repeating it:

| index on `ts` | size |
|---|---|
| BRIN | **24 kB** |
| B-tree | **26 MB** |

A factor of 1,100. The note also states what BRIN cannot do: no sorted output,
range/equality predicates only, and the advantage disappears once rows arrive
out of order.

## B8 — measured bloat instead of estimated

`physical.measured_bloat` uses `pgstattuple`, falling back to
`pgstattuple_approx` above 1GB because the exact form reads every page.

`tables.py` estimates dead space from `pg_stat_user_tables` counters. The action
on the other side of that number is `VACUUM FULL` or `CLUSTER` — an
`ACCESS EXCLUSIVE` lock plus a spare copy's worth of disk. An outage recommended
from an estimate is worth one exact read first. Live: `orders` 0.0% dead, 0.32%
free.

## F3 — the statistics kind now matches the measurement

`_probe_query` builds `WHERE left = %s AND right = %s`. Pure equality. So the
before/after q-error `extended_stats` reports is evidence about **`dependencies`
and nothing else** — while the recommendation asked for all three kinds.

Now it recommends `dependencies`, and offers the other two separately with the
condition that would justify each:

- **`ndistinct`** — only if the columns are also grouped together. The manual:
  on columns nobody groups by, *"the ANALYZE cycles are just wasted."*
- **`mcv`** — if the queries use range clauses, or if impossible combinations
  occur. `dependencies` assumes the values are compatible, so
  `city = 'San Francisco' AND zip = '90210'` is estimated as though it matched.
  Costs more in ANALYZE, storage *and* planning time.

## F4 — statistics the planner is not allowed to read

`server_config.statistics_visibility`. §69.3: if the role lacks `SELECT` and the
operator is not `LEAKPROOF`, *"the selectivity estimator will behave as if no
statistics are available... There is no direct feedback about that, except that
the plan might be suboptimal."*

Worse here than in most systems: every estimate the model learns to correct
assumes the planner *read* the statistics. Under a restricted role it silently
did not, and the whole system would be compensating for a permissions boundary
rather than a correlated column. Returns nothing for an owner or superuser,
which is why it is cheap to leave on.

## A7 — pinned session settings

`measurement.make_deterministic`, called from `benchmark.py` and
`collect_data.py`. `geqo_seed` defaults to 0, which sounds fixed and is not —
GEQO reseeds per planning cycle, and §19.7.3 says *"Varying the value changes the
set of join paths explored, and may result in a better or worse best path being
found."* On a 12-relation join that turns a paired A/B into a comparison of two
draws from a random search.

Does nothing below `geqo_threshold`, which is every query in the current
workload — insurance for the JOB-scale benchmark rather than a fix for today.
Failures are reported, not raised: an unsettable GUC means the measurement is
less controlled, which is worth recording.

## D5 — covering indexes

`index_advice.covering_opportunities`, on the analyze path only: it needs
`EXPLAIN VERBOSE` for the `Output` list, which is one extra round trip that
`/query/analyze` can afford (it already issues ~30) and `/query/optimize`
cannot.

Verified end to end:

```
SELECT o.status, o.created_at FROM orders o WHERE o.user_id = 42
  before                                              Seq Scan
  CREATE INDEX ... (user_id) INCLUDE (status, created_at)
  after                     Index Only Scan on idx_orders_user_id_covering
```

This is what makes the `IndexOnlyScan` hint family usable rather than
theoretical — the optimizer can force an index-only scan, but only a covering
index makes one possible.

**The first gate was wrong and the live check caught it.** Gating on "the node
returns many rows" suppressed every real finding: `WHERE user_id = 42` returns
about four rows while sequentially scanning 200,000, which is exactly the case a
covering index fixes. `Plan Rows` on a Seq Scan is the post-filter count. Scan
**cost** is the right signal, and it is what `planner.budget_for` already uses.

The payload limit is three columns, not four, for the same kind of reason: on
`orders` — five columns, one of them the filter — a limit of four accepted
`SELECT *` and proposed an index duplicating 80% of the table.

## Totals

831 backend tests, 82 frontend, 15 integration. `/dbadvisor` returns six
sections; `/query/analyze` returns plan choice, fairness, index type, and
covering opportunities.

## The one thing left

**Predicting which join orders collapse to the same plan.** The prefilter removes
the provable no-ops; 56 of the 83 remaining candidates on a four-table join are
still duplicates, and they are `Leading()` variants. Whether a different join
order produces a different plan is precisely what the planner decides, so there
is no cheap way to know without asking it. Anything less than a proof here
shrinks the action space silently, which is the failure this project keeps
rediscovering.

That is also where the compiled-language question actually lands: the way to stop
paying for round trips is to be inside the planner, via `planner_hook` — which is
a C extension, not a C++ one.

---

# A user-reported bug: optimising the optimiser's own output

Reported workflow: optimise a query, copy the optimised SQL, paste it back to
improve it further. It was silently broken, and the failure is this project's
recurring one -- a hint that quietly stops existing.

## What was actually happening

`apply_hint` prepended unconditionally, so a second pass produced two comments:

```
/*+ Set(enable_nestloop off) */
/*+ Leading(u o) */
SELECT ...
```

**pg_hint_plan reads only the first such comment and discards the rest.** Not as
an error, not in the "not used" or "error hint" sections of its own debug
output -- the second comment does not exist to it. Measured, one query, one
join, cost against `used hint:`:

| input | cost | used |
|---|---|---|
| native | 5332.3 | — |
| `Leading(u o)` | 5332.3 | `Leading(u o)` |
| `Set(enable_hashjoin off)` | 14365.1 | `Set(enable_hashjoin off)` |
| A then B, stacked | 5332.3 | **`Leading(u o)` only** |
| B then A, stacked | 14365.1 | **`Set(...)` only** |
| merged into one comment | 14365.1 | **both** |

Whichever comment comes first wins outright. So on the second pass the
optimiser's new hint was applied and **the hint the user pasted in was thrown
away** -- the second pass replaced the first rather than building on it.

### A correction to make

An earlier reading of this said pg_hint_plan rejected the whole thing and
planned natively. That was wrong. `error hint:` is an empty section header
printed on every statement by `debug_print`, not a message about the statement
in hand. The mechanism is first-comment-wins, and the table above is what
settled it.

## The fix

**`apply_hint` merges.** `/*+ Leading(u o) Set(enable_hashjoin off) */` is one
comment carrying two directives, which is the documented form. Conflicts resolve
last-one-wins, matching pg_hint_plan's own rule for a repeated hint type, so
chained passes converge instead of accumulating contradictions:

```
pass 1   /*+ Leading(o u) */
pass 2   /*+ Leading(o u) Set(enable_nestloop off) */
pass 3   /*+ Leading(u o) Set(enable_nestloop off) */          <- order replaced
pass 4   /*+ Leading(u o) Set(enable_nestloop off) Set(work_mem 65536) */
```

Verified against the live planner -- all three directives appear in
`used hint:` at pass 3, where previously only the last would have.

**The baseline is the unhinted query.** `/query/analyze` now strips incoming
hints before planning the baseline. Otherwise the previous pass's plan becomes
the thing every candidate is compared against while the panel labels it
"native": the speedup would be measured from an already-optimised plan, and a
converged query would keep reporting "no improvement available". Both numbers
wrong in a way nothing on the page could reveal.

Three chained passes through the live API:

```
PASS 1  incoming []                                    chose Set(enable_hashjoin off)
PASS 2  incoming [Set(enable_hashjoin off)]            chose Leading(oi p o u)
        output: /*+ Set(enable_hashjoin off) Leading(oi p o u) */
PASS 3  incoming [Set(enable_hashjoin off), Leading(oi p o u)]
        output: /*+ ... Set(enable_indexscan off) */
```

**The panel says so.** `OptimizedQuery` now names the hints carried forward and
states that the baseline is still unhinted, so the speedup is not misread as
"faster than my last pass".

## Two bugs the tests caught while fixing it

**A bushy hint was silently dropped from the merge.** The directive pattern
allowed one level of nesting; `Leading(((a b) (c d)))` has two, so it matched
nothing and vanished. Chaining a pass onto a bushy result would have discarded
the bushy plan -- the same failure class, inside the fix for it. Replaced with a
balanced-paren scan.

**A JSX block comment describing hint syntax ended itself.** The comment
contained the character sequence that closes a block comment, so everything
after it parsed as code. Rewritten with line comments, and the reason recorded
in place.

## Totals

851 backend tests, 85 frontend, 15 integration.

---

# Round six — the rest of the backlog, and two bugs it exposed

## Identity: chained passes were three different queries

The `apply_hint` bug had a second half. `query_template` normalised literals but
kept the hint comment, so the fingerprint changed every pass:

    pass 1 (bare)   adhoc:c047519472b703a65c7118fcbd4c2f4e
    pass 2          adhoc:988d3b8e205ba3509f222cd3fde00af5
    pass 3          adhoc:2df438b04eb11d355f379fa1743d6e85

Everything per-query keys on that id. So each pass reached the **regression
guard**, the **decision cache** and the **regret curve** as a brand-new query --
the guard could not have blocked a bad hint however many times it had seen one.
Silently, because a fresh id is indistinguishable from a first visit.

Hints are now stripped from the identity. A hint says *how* to run a query, not
*which* query it is.

The existing test asserted the opposite -- "stripping it would make a hinted and
an unhinted execution indistinguishable" -- and that premise was already false:
`main.py` logs the baseline **and every candidate** with the same `sql_text`,
recording the difference in the `hint`, `is_baseline` and `is_chosen` columns.
`query_id` is deliberately shared, because that is what makes served-vs-native
pairing possible at all. pg_stat_statements takes the same view.

## `count_findings` was undercounting by 39

It enumerated section names, so the four sections added after it -- maintenance,
workload, physical, observability -- did not count. The live report said **4
findings** while 43 existed. A headline that undercounts is worse than none,
because the number looks authoritative.

Now counted structurally: a finding is a dict in a list. That skips the scalars
sections carry alongside (`pgstattuple_available`, `current`, the `note`
explaining an empty panel) without naming those either, and a section written
next week counts without anyone remembering to add it.

## And then the honest number was too loud

43 findings, **29 of them "noisy"**, 14 of those the tool observing itself.

The noisy check's own note offers "the spread may be between plans rather than
within one" as a possibility. For an optimizer-sweep row it is a **certainty**:
twenty-odd candidates for one query fold into a single normalised entry, and
their runtimes differ because that is the point of running them. Reporting it is
a false positive by construction. Suppressed for sweep traffic; 43 → 30.

`spilling` is still reported for sweep traffic, because "some candidate for this
query spills to disk" is a real fact about the action space -- and `work_mem` is
in that action space on exactly that argument.

## New: observability settings

`dbadvisor/observability.py`. `server_config` checks what makes the database
fast; these are what make it *diagnosable*, and all of them ship off or
effectively off:

| setting | default | what its absence costs |
|---|---|---|
| `track_io_timing` | off | every I/O-time column reads zero, so slow-because-disk looks identical to slow-because-CPU -- the exact 80.5%-vs-26.0% question |
| `log_temp_files` | -1 | which queries actually spill, rather than gambling `work_mem` on all of them |
| `log_lock_waits` | off | lock time is invisible to every plan-level measurement and looks like plan noise |
| `log_autovacuum_min_duration` | **10 min** | an aggressive freeze scan landing mid-benchmark, appearing as unexplained variance |
| `log_min_duration_statement` | -1 | what is actually slow |
| `compute_query_id` | auto | joining a logged slow statement to its accumulated statistics |

Every finding states its cost, because "turn on more logging" is advice that has
taken production systems down. The ten-minute autovacuum default is treated as
off, because it is not `-1` and a naive check reads it as enabled.

## New: B10 partitioning, D9 multicolumn cap

**Partitioning** (`physical.partition_candidates`). §11.8 Example 11.4 is direct:
a fan of partial indexes on a large table is "a bad idea" because the planner
"will laboriously test each one", and partitioning is its named alternative.
Also the only thing that unlocks `enable_partitionwise_join` and
`enable_partitionwise_aggregate` -- family 11, the two planner features that
ship off.

Proposes **no DDL**: it is a schema redesign with migration cost. Finds nothing
on this database, correctly -- the largest table is 21 MB against a 100 MB
floor. Verified the logic at a lowered threshold, where it found
`plan_execution_log` -- this project's own log table, now excluded for the same
reason `extended_stats` excludes it.

**Multicolumn cap.** §11.3: *"Indexes with more than three columns are unlikely
to be helpful unless the usage of the table is extremely stylized."* A
five-column predicate now yields a three-column index and says why, plus the
§11.5 trade-off: separate single-column indexes combined with a BitmapAnd may
serve a mixed workload better, at the cost of an extra scan and a lost ordering.

## Totals

876 backend tests, 85 frontend, 15 integration. `/dbadvisor` returns seven
sections and a headline that is finally correct.

## What is left, and why

- **PG17/PG18 items** (A9, A10, C1, C2, C3, C5) -- blocked on a server upgrade.
  `Disabled: true` in EXPLAIN (C2) is the one worth upgrading *for*: it turns
  "this candidate did nothing" from an inference into a fact.
- **B13 non-durable settings** -- deliberately not done. `fsync=off` on a
  database bind-mounted into the repo is a real risk for a benchmark
  convenience; it needs a separate compose profile, not a change to this one.
- **Join-order duplicate prediction** -- 56 of 83 candidates on a four-table
  join are still duplicates, and they are `Leading()` variants. Whether two join
  orders produce the same plan is what the planner decides; anything less than a
  proof shrinks the action space silently.
