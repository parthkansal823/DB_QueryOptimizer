CREATE EXTENSION IF NOT EXISTS pg_hint_plan;

-- Hypothetical indexes, so the advisor can measure what an index would be
-- worth instead of only noticing that one is missing. Planner-only: it
-- writes no pages and holds no lock. See backend/app/hypothetical.py.
CREATE EXTENSION IF NOT EXISTS hypopg;

-- Per-query execution statistics: planning time separated from execution time,
-- shared block hits against reads, temp blocks, WAL. `app/dbadvisor/query_stats.py`
-- and `app/dbadvisor/observability.py` read it.
--
-- It was in `shared_preload_libraries` (see docker-compose.yml) but *not* here,
-- which is the same split that made pg_hint_plan silently do nothing in
-- docs/WRITEUP.md 2.0, arriving from the opposite side: preloading starts the
-- collector, `CREATE EXTENSION` registers the view that reads it. With only the
-- first, statistics accumulate where nothing can see them.
--
-- It survived because the PostgreSQL 16 cluster had the extension created by
-- hand at some point. The PG18 upgrade rebuilt the cluster from these scripts
-- and the omission surfaced immediately -- an init script is only as good as
-- the last time someone started from nothing.
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
