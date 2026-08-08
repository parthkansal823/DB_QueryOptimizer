# Roadmap

A phase plan for a solo, one-year timeline. Line these up against your
actual semester structure (synopsis submission, mid-term review, final
defense) rather than treating them as fixed durations.

## Phase 0 -- Foundation (this scaffold)
- Postgres + pg_hint_plan running in Docker
- Synthetic benchmark schema seeded (users / products / orders / order_items)
- Plan extraction (EXPLAIN JSON -> structured metrics) working end to end
- Join-order candidate generation via hints
- Heuristic "learned" optimizer (lowest estimated cost) as a placeholder --
  proves the pipeline before any real ML exists

**Goal: run `python -m app.benchmark` and see real baseline vs. candidate latencies.**

## Phase 1 -- Data collection
- Expand the workload to 20-30 representative queries: 2-way, 3-way, and
  4+-way joins, a range of selectivities
- Log every (query, candidate plan, actual latency) triple somewhere
  durable (a table, or flat files) -- this becomes your training data
- Add more realistic skew to the data if time allows (a few power users
  with thousands of orders, many products never ordered) -- that's where
  optimizers tend to actually struggle

## Phase 2 -- Feature engineering
- Turn each (query, candidate plan) pair into a fixed-length feature
  vector: join order, table sizes, estimated selectivities, index
  availability, filter predicates
- This step usually decides whether the model works at all -- budget
  real time for it, don't treat it as a formality

## Phase 3 -- Train the model
- Start simple: gradient-boosted trees (LightGBM/XGBoost) predicting
  latency per candidate, then pick the argmin. Much easier to train
  and to explain in a viva than deep RL.
- Stretch goal if time allows: a contextual bandit (Thompson sampling
  or LinUCB) that keeps exploring -- closer to how Bao actually works
- Evaluate against both the Phase 0 heuristic AND native Postgres --
  you want both baselines in the writeup

## Phase 4 -- Online integration
- Swap `LearnedOptimizer._select_heuristic` for the trained model --
  `select()`'s interface shouldn't need to change
- Add basic logging so you can show "learned vs. native" trending over
  time, not just one benchmark run
- Address the cold-start problem honestly in your writeup: what does
  the system do before it has enough data to trust the model?

## Phase 5 -- Dashboard + writeup
- React frontend: paste a query, see baseline vs. chosen plan side by
  side, a latency chart, historical accuracy of the model's picks
- Literature review + comparison table (your system vs. Neo vs. Bao vs.
  native Postgres) + an honest limitations section -- evaluators notice
  and reward candor about what doesn't work

## Stretch goals (only once Phases 0-5 land early)
- Extend beyond join order to join *method* selection (hash join vs.
  nested loop vs. merge join)
- Swap the synthetic dataset for the real Join Order Benchmark
  (JOB / IMDB dataset) so your numbers are directly comparable to
  published papers
