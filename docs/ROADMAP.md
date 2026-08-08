# Roadmap

A phase plan for one person over one year. Line these up against your actual
semester dates (synopsis, mid-term review, final defence) rather than treating
the durations as fixed.

**Status: phases 0-5 done, both stretch goals attempted.** See
`docs/WRITEUP.md` for results and an honest list of limitations, and
`data/job/README.md` for where the JOB/IMDB stretch goal landed.

## Phase 0 -- Foundation [done]
- Postgres + pg_hint_plan running in Docker
- Synthetic benchmark schema loaded (users / products / orders / order_items)
- Plan extraction (EXPLAIN JSON to structured metrics) working end to end
- Join-order candidates generated with hints
- A placeholder "learned" optimizer that just picks the lowest estimated cost.
  It proves the pipeline works before any real ML exists

**Goal: run `python -m app.benchmark` and see real baseline and candidate times.**

## Phase 1 -- Data collection [done]
- Workload grown to 25 queries (`backend/app/workload.py`): 2-way, 3-way and
  4-way joins, across a range of selectivities, each one tagged
- Every (query, candidate plan, actual time) triple logged to the
  `plan_execution_log` table (`postgres/init/03_logging.sql`,
  `backend/app/logging_store.py`). This is the training data
- Skew added to the seed data: 100 power users account for 40% of all orders,
  200 popular products account for 50% of all order items (`data/schema.sql`)

## Phase 2 -- Feature engineering [done]
- `backend/app/optimizer/features.py` turns each (query, candidate plan) pair
  into a fixed-length feature vector: join position, selectivity, scan type,
  and join-method counts per table
- Schema-agnostic by design (see "works on any database" below). This step got
  real time budgeted to it in the original plan, and it is what made the
  JOB/IMDB stretch goal possible without a rewrite

## Phase 3 -- Train the model [done]
- LightGBM gradient-boosted trees (`backend/app/train.py`) predicting
  candidate latency. Falls back to scikit-learn's
  `GradientBoostingRegressor` if the LightGBM native library is missing
- Scored against the Phase 0 heuristic, native Postgres, **and** an oracle
  (the best candidate in hindsight). The oracle is what makes the other
  numbers readable. Results in `docs/WRITEUP.md` §2
- **Contextual-bandit stretch goal [done]**: a bootstrapped ensemble
  (`backend/app/optimizer/bandit.py`) supports Thompson sampling, the same
  mechanism Bao uses, plus a risk-averse policy driven by ensemble
  uncertainty. LinUCB was not attempted, because it assumes a linear reward
  model and the ensemble gives uncertainty without that assumption

## Phase 4 -- Online integration [done]
- `LearnedOptimizer._select_learned` built. `select()`'s interface did not
  change: the policy and safety additions are constructor arguments plus a new
  `select_plan()`, so existing callers were untouched
- Safety veto: a learned pick costed far above the native plan is dropped
  rather than served. This is Bao's "never much worse than the optimizer you
  replaced" property
- Every `/query/analyze` call and every `app.benchmark` run logs to
  `plan_execution_log`, tagged with which selector produced it.
  `/stats/trend` turns that into "learned vs. native, over time"
- Cold start handled honestly: it falls back to the Phase 0 heuristic until a
  model is trained (see `docs/WRITEUP.md` §4)

## Phase 5 -- Dashboard + writeup [done]
- React (Vite) + Recharts dashboard (`frontend/`): paste a query and see
  PostgreSQL's plan against the chosen one, every candidate measured, why the
  decision went the way it did, and the history so far.
  `docs/DASHBOARD.md` explains each panel, `docs/METRICS.md` explains each
  number
- `docs/WRITEUP.md`: literature review, a comparison table (this project vs.
  Neo vs. Bao vs. native Postgres), results, and a limitations section

## Beyond the original scope [done]

Additions past phases 0-5. Each one came from a measurement rather than a
feature list. `docs/WRITEUP.md` §2.4 has the evidence.

- **Self-learning loop** (`app/retrain.py`): retrains on accumulated feedback
  and deploys a new model **only if it clearly beats the one running**, scored
  on the same held-out set. The margin exists because offline scores here are
  noisy enough that deploying on any improvement would send the model on a
  random walk.
- **Versioned models with rollback** (`app/model_store.py`): unattended
  retraining is only safe if a bad deployment can be undone.
- **Per-query regression guard** (`app/optimizer/regression_guard.py`): blocks
  the learned path for queries with a measured history of being slower than
  native. This targets the per-query instability that recent research
  identifies as the main barrier to deploying learned optimizers.
- **Pairwise learning-to-rank** (`app/optimizer/ranker.py`): Lero-style.
  Predicting *which plan is faster* is a much easier problem than predicting
  latency, and that matters when prediction error is as large as the spread
  between candidates.
- **Plan-tree structural encoding** (`app/optimizer/plan_tree.py`): a cheap
  stand-in for the tree convolution Neo and Bao use.
- **Model health API and dashboard panel**: deployed version, how much
  feedback is waiting, which queries are blocked, and retrain/rollback
  controls.
- **Works on any PostgreSQL database** (`app/onboard.py`,
  `app/schema_graph.py`, `app/workload_generator.py`): discovers the schema,
  works out join links from column names when no foreign keys are declared,
  and generates a workload with filters sampled from the real data. Checked
  with the same command on the synthetic schema and on the 21-table, 74M-row
  JOB/IMDB benchmark.
- **Production inference path** (`app/optimizer/planner.py`): plans every
  candidate with `EXPLAIN` (nothing runs) and executes only the winner. About
  2.7% overhead, against the demo path's N executions per query.
- **Cumulative regret** (`app/optimizer/regret.py`): 0.80x native over 500
  logged decisions. The trajectory, not just an average.
- **Learned cardinality correction** (`app/optimizer/cardinality.py`): learns
  Postgres's own q-error from `Plan Rows` against `Actual Rows`, the root
  cause Leis et al. identified.
- **Honest dashboard reporting** (`app/stats.py`): every before-and-after
  figure is a matched pair from the same run. The panel previously divided two
  averages taken over different sets of queries and reported a 97% win that
  was not real. `docs/METRICS.md` §2 works through it.

## Stretch goals
- **Join method selection [done]**: `generate_join_method_candidates`
  (`backend/app/optimizer/hints.py`) goes beyond join order to force
  `HashJoin` / `NestLoop` / `MergeJoin` per join-order prefix
- **Real Join Order Benchmark (JOB/IMDB) dataset [attempted, see
  `data/job/README.md` for the outcome]**: a loader script
  (`data/job/load_job.sh`) downloads and imports the real 21-table IMDB
  dataset and 113 JOB queries into a second database. The join-order and
  join-method pipeline needed zero code changes to point at it, thanks to the
  Phase 2 schema-introspection work
