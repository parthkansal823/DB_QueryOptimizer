# TPC-H dataset

The standard decision-support benchmark: 8 tables in a snowflake around
`lineitem`, with declared foreign keys and a well-understood join graph.

## Why this dataset was added

Measurement, not preference. The synthetic e-commerce schema turned out to
have an **oracle ceiling of ~6.5%** — only 7 of its 25 queries have any plan
that beats PostgreSQL by more than 5%, and none of six model classes could
pick the best plan even once (`docs/WRITEUP.md` §2.9). A learned optimizer
cannot demonstrate anything on a workload the built-in optimizer already
handles correctly.

TPC-H gives a second real reference point that is cheap to obtain: unlike
JOB's 1.2 GB download, it is **generated locally in SQL**, so scale is a
parameter rather than a download.

## Load it

```bash
bash data/tpch/load_tpch.sh 0.1      # ~1.2M rows, about a minute
bash data/tpch/load_tpch.sh 1.0      # standard SF1, ~8.7M rows, much slower
```

Then onboard it like any other database — nothing in the optimizer knows
which dataset it is pointed at:

```bash
docker compose exec -e DATABASE_URL=postgresql://postgres:postgres@postgres:5432/tpch \
    backend python -m app.onboard --queries 20 --reps 3
```

## What's faithful here, and what isn't

The official generator (`dbgen`) is a C program that must be fetched and
compiled. `generate.sql` reproduces what actually matters for join-order
research without that dependency:

- **Faithful**: table set, foreign-key graph, and the spec's cardinality
  *ratios* — supplier 10k·SF, part 200k·SF, customer 150k·SF, orders 1.5M·SF,
  lineitem ~6M·SF at 1–7 lines per order. Relative table sizes are what drive
  join-order decisions, so these are the numbers that count.
- **Faithful**: deliberate skew, which uniform random keys would destroy —
  20% of customers place 60% of orders, 15% of parts appear in 50% of line
  items. `data/schema.sql` learned this lesson the hard way: uniform join
  keys make join order nearly irrelevant.
- **Not faithful**: the literal text values. `dbgen` builds comments and
  names from a specified word list; these are simple generated strings. This
  affects string-predicate selectivity, so **results here are not comparable
  to published TPC-H numbers** — it is a realistic join workload, not a
  certifiable TPC-H run.

If you need certifiable numbers, use the real `dbgen` from
<https://github.com/electrum/tpch-dbgen> and `\copy` its output into the
schema in `schema.sql`, which is spec-accurate.

## Indexes

Primary keys only, deliberately — the same choice `data/schema.sql` makes.
Adding the foreign-key indexes is a worthwhile controlled experiment: the
learned optimizer's advantage should shrink as PostgreSQL gets better access
paths to work with.
