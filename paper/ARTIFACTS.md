# Artifact manifest

What each table and figure in `paper/main.tex` comes from, and how to rerun
it. Historical figures are labelled as such. They come from `docs/WRITEUP.md`
and cannot be regenerated exactly, because the database state they were taken
on no longer exists.

## Environment

| | |
|---|---|
| Machine | Intel Core Ultra 7 155H (16 cores, 22 threads), 32 GB RAM, NVMe SSD, Windows 11, Docker Desktop |
| Database | PostgreSQL 18.6 + `pg_hint_plan` 1.8.1, built by `postgres/Dockerfile`; defaults except `shared_preload_libraries=pg_hint_plan,pg_stat_statements`, `compute_query_id=on`, `track_io_timing=on` (`docker-compose.yml`) |
| Python | 3.11.2; `backend/requirements.txt`; LightGBM 4.5.0, scikit-learn 1.5.2 |
| Code | tag `paper-v2` (see the paper's Artifact Availability) |

## Inputs and their hashes (SHA-256)

| File | SHA-256 | Note |
|---|---|---|
| `backend/models/plan_selector.pkl` | `785289d6fffca71559681d41d62430c991c1b24dbf3db720ef856450dfc8ce97` | deployed synthetic model: 45 features, target `log_ratio_vs_native`, 16 held-out queries (listed in the bundle as `held_out_queries`) |
| `backend/models/gate.json` | `aff1dbf8812a24deea608d812d8f55d988497ffacf4ce4d2aee2883cba6c7184` | `fixed_z`, z = 2.0, g = 0, calibrated with `pairwise_rank` on the 16 held-out queries |
| `backups/lqo_pg18_20260904-1345.dump` | `c426f96f10c8a24be9f25c6265fa1b657be2df5633afeff8c35ce23b2e4a09f1` | database snapshot (PostgreSQL 18.6, 4 Sep 2026). **Not in git** (`backups/` is ignored); attach it to the GitHub release |
| `backend/requirements.txt` | `cbecb4f4c42129a9515408699988caa680d2973ffbe7cb051cd77aa80dc7fc07` | |

Query definitions: `backend/app/workload.py` (65 synthetic queries) and
`data/job/queries` (JOB).

## Restoring the snapshot

```bash
docker compose build postgres && docker compose up -d postgres
docker compose exec -T postgres pg_restore -U postgres -d lqo --clean --if-exists --no-owner < backups/lqo_pg18_20260904-1345.dump
docker compose exec -T postgres psql -U postgres -d lqo -c "VACUUM ANALYZE;"
```

## Regenerating the new measurements

All runs use the default model and gate above. Before each run, the execution
log is reset to the snapshot and PostgreSQL is restarted, so the regression
guard starts from the same history and shared buffers start empty. The exact
driver used is reproduced here:

```bash
reset_state() {
  docker compose exec -T postgres psql -U postgres -d lqo -qc "TRUNCATE plan_execution_log;"
  docker compose exec -T postgres pg_restore -U postgres -d lqo --data-only -t plan_execution_log < backups/lqo_pg18_20260904-1345.dump
  docker compose restart postgres   # then wait for pg_isready
}

# Table V, last column: five independent paired runs, seeds 100-104
# (after the D47 fix: the served arm is decided by planner.plan_query)
for i in 0 1 2 3 4; do
  reset_state
  (cd backend && python -m app.benchmark --protocol paired --repeat 1 --arm-order random \
     --seed $((100 + i)) --json-out models/paired_served_run$i.json)
done

# Section V-C: current noise figure (65 queries x 7 executions of the native plan)
reset_state
(cd backend && python -c "import json; from app.noise import measure_workload; json.dump(measure_workload(reps=7), open('models/noise_pg18.json','w'), indent=2)")

# Table VII: end-to-end latency (5 repetitions per query and path, seed 200)
reset_state
(cd backend && python -m app.e2e_latency --reps 5 --seed 200 --json-out models/e2e_latency_snapshot.json)
```

Raw outputs: `backend/models/paired_served_run{0..4}.json`,
`backend/models/noise_pg18.json`, `backend/models/e2e_latency_snapshot.json`.
`backend/models/paired_snapshot_run{0..4}.json` are the same five runs taken
before the D47 fix (served arm decided by `select_plan` on the benchmark's own
candidates, with no guard caution); the paper cites them only as the evidence
for D47. The earlier five-run sequence without resets is
`backend/models/paired_5runs_random.json`, with the end-to-end run from the same
session in `backend/models/e2e_latency.json`; the paper does not use them.

The noise figure is `median_relative_spread` (12.4%) and
`median_warm_relative_spread` (9.6%, first execution excluded) in
`noise_pg18.json`: per query, (max − min) / median over 7 executions of the
native plan; median over the 65 queries. The older `backend/models/noise.json`
uses the same definition on an earlier 24-query workload and database state
(warm median 29.1%) and is not used.

Intervals: the paper's 95% intervals over runs use Student's *t* with
4 degrees of freedom on the five per-run values. The end-to-end intervals are
percentile bootstraps over queries of the paired per-query differences
(10,000 resamples, seed 0, `app.experiment.bootstrap_ci`).

## Provenance of historical figures

| Paper item | Source | PostgreSQL | Population and timing |
|---|---|---|---|
| Table II, synthetic v1/v2 | WRITEUP §2.9 | 16 | v1: 4-table schema, 25 queries; v2: 23 queries; logged collection |
| Table II, JOB | `docs/JOB_RESULTS.md` (re-run 2026-09-05) | 18 | 8 held-out queries; logged collection |
| JOB correction, Fig. 2 | WRITEUP §2.80 | 18 | 8 held-out queries; two collection runs (3 repetitions, median) |
| Transfer, Table VI | WRITEUP §2.64, §2.86 | 18 | collected JOB queries; logged collection; narrow-space oracle reconstructed |
| Table V, single runs | WRITEUP §2.85 | 18 | 65 queries; live; fixed arm order |
| Action-space sweep (Section VI-E) | WRITEUP §2.94 | 18 | 16 held-out queries; replay over logged rows |
| Ensemble spread (Section III-B) | `docs/CONFIDENCE.md` §3 | not recorded | out-of-bag predictions on training rows |
| Defect inventory | `docs/DEFECTS.md` | — | 47 entries |

## Model and gate settings

Ensemble: 8 LightGBM regressors (200 trees, max depth 5, learning rate 0.05,
otherwise default; L2 objective on the natural-log latency ratio), each on a
bootstrap resample of plan rows, seeded 42 + member index
(`app/optimizer/bandit.py`, `app/train.py`). Ranker: LightGBM classifier with
the same settings on both orderings of within-query candidate pairs, native
included, ties skipped, at most 400 pairs per query
(`app/optimizer/ranker.py`). Gate grid: z in {0, 0.05, 0.1, 0.15, 0.25, 0.5,
0.75, 1.0, 1.5, 2.0}, g in {0, 0.02, 0.05, 0.10, 0.20} (`app/calibrate.py`);
safety bounds 25% regressing deviations and 1.25x worst case
(`app/autotune.py`). Candidate budget: 4/10/20 by cost tier, scaled by join
width, cap 96; up to 8 sampled join orders (`app/optimizer/planner.py`).
Timeouts: 30 s per statement; a failed or timed-out candidate is skipped.
