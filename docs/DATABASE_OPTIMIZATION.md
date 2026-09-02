# Fixing the database, not just the query

Most of this project answers one question: *given this query, which plan should
run?* That is the hard part and it is what the learned model is for.

It is also not the first reason most databases are slow.

The plan-picker takes two things as fixed: the planner's row estimates, and how
the server is configured. It then learns to work around them. This package looks
at what makes those wrong in the first place. Each finding is fixed once and then
helps **every** query — including ones this system has never seen, and ones no
hint in its action space could have expressed.

```bash
docker compose exec backend python -m app.dbadvisor
```

Also at `GET /dbadvisor` and in the dashboard's **Database health** panel.

**Nothing here changes anything.** Every finding carries the exact statement
that would apply it. Running DDL against someone's database is their decision,
not a button.

---

## 1. Correlated columns — the project's own premise, fixed at the source

Every explanation panel in this dashboard says a version of this sentence:

> PostgreSQL assumes filters are unrelated and multiplies their odds together.
> When the columns are actually related, that estimate is far off, and the plan
> gets built for the wrong number of rows.

Here is what that means concretely. Suppose 1 user in 10 lives in Mumbai, and 1
in 5 lives in India. PostgreSQL estimates `city = 'Mumbai' AND country = 'IN'`
as 1/10 x 1/5 = **1 in 50**. But every Mumbai user is in India, so the real
answer is 1 in 10 — the planner is wrong by a factor of five, and it builds the
plan for a fifth of the rows it will actually get.

Every hint family in this project is a way of **working around** that bad
estimate: force a different join order, ban an operator, or hand back a
corrected row count.

PostgreSQL has a first-class fix for the same problem, and this project had
never used it:

```sql
CREATE STATISTICS ON city, country FROM users;
ANALYZE users;
```

That tells the planner the two columns are related, so it stops multiplying.
The estimate becomes right, and every plan built from it improves — permanently,
for every query touching those columns, without this optimizer having to think
of the hint first.

### The recommendations are measured, not guessed

A correlation in the data does not mean the planner is currently getting it
wrong, and a wrong estimate does not always change the plan. So each candidate
is actually **tried**: create the statistics object, run `ANALYZE`, re-plan the
query, compare the estimate against reality, then roll the whole thing back.

What comes back is a measurement:

```
products(brand, category): estimate off by 4.9x -> 1.0x
    CREATE STATISTICS ... (dependencies, ndistinct, mcv) ON "brand", "category" FROM "products";
```

That is affordable here in a way `CREATE INDEX` is not — a statistics object
writes no table data and takes no long lock. The cost is the `ANALYZE`, which is
why tables above five million rows are reported without being measured rather
than paid for.

### Why this is allowed to compete with the rest of the project

`docs/WRITEUP.md` §2.8.1 reports that the learned `Rows()` correction raised the
oracle ceiling by **0.0%** — every win it found was already reachable through
existing hints. Extended statistics might do the same thing better, and if they
remove the estimation error that the hints exist to route around, the honest
conclusion is that the database needed statistics more than it needed a learned
optimizer.

That is worth measuring rather than avoiding, which is why it lives in the same
tool.

---

## 2. Server settings — defaults that were never decisions

PostgreSQL ships with settings chosen so it will start on almost anything.
`shared_buffers` defaults to 128MB whether the machine has 1GB of RAM or 128GB.
Nothing fails; queries are simply several times slower than they need to be, and
every layer above spends its effort compensating.

Run against this project's own container:

```
machine memory: 15.4GB

shared_buffers: 128MB -> 3.8GB
    128MB on a machine with 15.4GB of RAM. A default that was never chosen.

random_page_cost: 4 -> 1.1
    4.0 encodes a seek-heavy spinning disk. On SSD it prices index scans far
    above what they cost, biasing the planner toward sequential scans.

max_parallel_workers_per_gather: 2 -> 4
```

### Why this project in particular has to report this

One of the hint families added to the action space sets `work_mem`, and it was
measured taking a three-table join from cost 23264 to 17792 — a 24% drop.

A good part of that win exists **because the server is running on a 4MB
`work_mem`**. A learned optimizer that looks impressive mainly because the
database underneath it is starved is not a result to be pleased with. Reporting
the configuration next to the model's numbers is what stops that being quietly
beneficial.

The suggestions are the conventional ones from any PostgreSQL tuning guide,
expressed as fractions of detected memory. Where memory cannot be detected, the
memory-dependent checks are skipped rather than guessed — a recommendation
derived from a made-up machine size is worse than none.

---

## 3. Indexes that already exist and are not earning their keep

`app/advisor.py` looks for indexes that are *missing*. `app/hypothetical.py`
measures what one would be worth before you build it. Neither looks at the
indexes that are already there, and on a database that has been running a while
those are the more common problem.

Every index is paid for on every write to its table and in every backup, and
that cost is invisible — no slow-query report points at an index nobody reads.

**Unused.** PostgreSQL counts how many times the planner has chosen each index.
Zero, on a table that is being read, means pure overhead. Three caveats are
reported alongside, because they matter: a unique index is a *constraint* as
well as an index, so dropping it changes what the database will accept; a
primary key is never reported; and counters reset on restore, so "never used"
can mean "never used since Tuesday".

**Duplicate.** Two indexes on exactly the same columns in the same order. One of
them does nothing the other does not. Note that `(a, b)` and `(b, a)` are *not*
duplicates — they serve different lookups.

**Redundant.** An index on `(a)` when `(a, b)` exists. The planner can use the
wider one for anything the narrow one serves. Reported rather than assumed,
because a much smaller index can still be worth keeping for a hot lookup.

---

## 4. Table health — the boring reasons estimates go wrong

**Stale statistics.** The planner's row counts come from the last `ANALYZE`. A
table that has changed substantially since then is being planned from a
description of a database that no longer exists.

This one is worth checking first, because *a learned optimizer trained against
stale statistics is learning to compensate for something an `ANALYZE` would
remove*.

**Dead rows.** An `UPDATE` in PostgreSQL writes a new row version and leaves the
old one behind; `VACUUM` reclaims it. A table where dead versions outnumber live
ones does several times the I/O it needs to on every sequential scan, and no
plan choice fixes that.

**Sequential scans on large tables.** Not a problem by itself — scanning a small
table is correct, and so is scanning a large one to aggregate most of it. On a
large table scanned repeatedly it is usually a missing index, which
`POST /advisor/rank` can then price.

---

## 5. Pricing an index without building one

Separate from the advisor above, and worth knowing about:

```bash
curl -X POST localhost:8000/advisor/rank -d '{"sql": "..."}'
```

This uses [HypoPG](https://github.com/HypoPG/hypopg), which registers an index
in the planner's catalogue only — no pages written, no lock taken, nothing
outliving the session. `EXPLAIN` then plans as though the index existed.

```
CREATE INDEX ON "order_items" ("order_id")
    cost 14521 -> 3458  (76% cheaper)   planner would use it: True
```

The last column is the decisive one. An index the planner would ignore is worth
nothing however much cheaper the plan looks on paper, and that is measured
exactly rather than inferred from the cost moving.

What this does *not* measure is latency — a hypothetical index cannot be
scanned, so the query cannot actually run against it. Everywhere else this
project complains that PostgreSQL's cost estimates are unreliable; here the
estimate is all there is. It is still worth having, because "would the planner
*use* this index" is answered exactly.

---

## 6. Reading the output

Findings are grouped, and each carries the statement that applies it:

```
=== correlated columns the planner does not know about ===
  products(brand, category): estimate off by 4.9x -> 1.0x
      CREATE STATISTICS ... ON "brand", "category" FROM "products";

=== server settings ===
    machine memory: 15.4GB
  shared_buffers: 128MB -> 3.8GB
      ALTER SYSTEM SET shared_buffers = '3.8GB'; SELECT pg_reload_conf();

=== table health ===
  scanned      order_items: 19675 sequential vs 6 index scans
```

`--no-statistics` skips the correlated-column check, which is the only expensive
one (an `ANALYZE` per candidate). The `GET /dbadvisor` endpoint has it off by
default for the same reason — a polled endpoint should not be running `ANALYZE`.

---

## 7. What is deliberately not here

**Nothing is applied automatically.** Creating an index, changing
`shared_buffers`, or dropping anything are decisions with consequences this tool
cannot see: a unique index that is also a constraint, a maintenance window, a
replica that has to be kept in step.

**No `pg_stat_statements` integration yet.** That would let the advisor find the
queries actually worth optimising rather than being handed a workload. It is the
obvious next addition and is not built.

**No partitioning or sharding advice.** Those are schema redesigns, not
settings, and recommending one from catalogue statistics alone would be
guessing.
