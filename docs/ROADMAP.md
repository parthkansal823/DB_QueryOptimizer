# Roadmap

A phase plan for a solo, one-year timeline. Line these up against your
actual semester structure (synopsis submission, mid-term review, final
defense) rather than treating them as fixed durations.

**Status: Phases 0-5 complete, both stretch goals attempted.** See
`docs/WRITEUP.md` for results and an honest limitations section, and
`data/job/README.md` for the JOB/IMDB stretch goal's status.

## Phase 0 -- Foundation [done]
- Postgres + pg_hint_plan running in Docker
- Synthetic benchmark schema seeded (users / products / orders / order_items)
- Plan extraction (EXPLAIN JSON -> structured metrics) working end to end
- Join-order candidate generation via hints
- Heuristic "learned" optimizer (lowest estimated cost) as a placeholder --
  proves the pipeline before any real ML exists

**Goal: run `python -m app.benchmark` and see real baseline vs. candidate latencies.**

## Phase 1 -- Data collection [done]
- Workload expanded to 25 queries (`backend/app/workload.py`): 2-way,
  3-way, and 4-way joins, a spread of selectivities (tagged per query)
- Every (query, candidate plan, actual latency) triple logged durably to a
  `plan_execution_log` Postgres table (`postgres/init/03_logging.sql`,
  `backend/app/logging_store.py`) -- this is the training data
- Added skew to the seed data: 100 power users draw 40% of all orders, 200
  popular products draw 50% of all order_items (`data/schema.sql`)

## Phase 2 -- Feature engineering [done]
- `backend/app/optimizer/features.py` turns each (query, candidate plan)
  pair into a fixed-length feature vector: join position, selectivity, scan
  type, and join-method counts per table
- Schema-agnostic by construction (see "dataset-agnostic pipeline" below) --
  budgeted real time for this step per the original plan, and it's what
  made the JOB/IMDB stretch goal possible without a rewrite

## Phase 3 -- Train the model [done]
- LightGBM gradient-boosted trees (`backend/app/train.py`) predicting
  candidate latency, falls back to scikit-learn's `GradientBoostingRegressor`
  if the LightGBM native lib isn't available
- Evaluated against both the Phase 0 heuristic AND native Postgres --
  results in `docs/WRITEUP.md` Section 2
- Contextual-bandit stretch goal (Thompson sampling/LinUCB) not attempted --
  out of scope once the two chosen stretch goals (join-method selection,
  JOB/IMDB) were prioritized instead; noted as future work in the writeup

## Phase 4 -- Online integration [done]
- `LearnedOptimizer._select_learned` implemented; `select()`'s interface
  didn't change
- Every `/query/analyze` call and every `app.benchmark` run logs to
  `plan_execution_log`, tagged with which selector produced it -- `/stats/trend`
  aggregates this for "learned vs. native, over time"
- Cold start addressed honestly: falls back to the Phase 0 heuristic until
  a model is trained (see `docs/WRITEUP.md` Section 4)

## Phase 5 -- Dashboard + writeup [done]
- React (Vite) + Recharts dashboard (`frontend/`): paste a query, see
  baseline vs. chosen plan side by side, a latency chart per candidate, and
  a historical-accuracy trend chart
- `docs/WRITEUP.md`: literature review + comparison table (this project vs.
  Neo vs. Bao vs. native Postgres), results, and a limitations section

## Stretch goals
- **Join method selection [done]** -- `generate_join_method_candidates`
  (`backend/app/optimizer/hints.py`) extends beyond join order to forced
  `HashJoin`/`NestLoop`/`MergeJoin` hints per join-order prefix
- **Real Join Order Benchmark (JOB/IMDB) dataset [attempted -- see
  `data/job/README.md` for outcome]** -- loader script
  (`data/job/load_job.sh`) downloads and imports the real 21-table IMDB
  dataset + 113 JOB queries into a second database; the join-order/-method
  pipeline required zero code changes to point at it, per the Phase 2
  schema-introspection work
