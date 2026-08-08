-- Durable sink for (query, candidate plan, actual latency) triples --
-- Phase 1 of the roadmap. Populated by `app.collect_data` during offline
-- data collection and by `app.main` on every live `/query/analyze` call
-- (Phase 4), so it doubles as both training data and the "learned vs.
-- native, trending over time" log the roadmap's Phase 4/5 dashboard needs.
CREATE TABLE IF NOT EXISTS plan_execution_log (
    id BIGSERIAL PRIMARY KEY,
    query_id TEXT,                 -- workload.py id, or NULL for ad-hoc dashboard queries
    sql_text TEXT NOT NULL,
    hint TEXT,                     -- NULL for the native-Postgres baseline row
    is_baseline BOOLEAN NOT NULL DEFAULT FALSE,
    selector_used TEXT NOT NULL DEFAULT 'native', -- native | heuristic | learned
    raw_plan JSONB NOT NULL,
    total_cost DOUBLE PRECISION,
    actual_total_time_ms DOUBLE PRECISION,
    planning_time_ms DOUBLE PRECISION,
    is_chosen BOOLEAN NOT NULL DEFAULT FALSE,      -- was this the candidate the optimizer picked?
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_plan_execution_log_query_id ON plan_execution_log (query_id);
CREATE INDEX IF NOT EXISTS idx_plan_execution_log_created_at ON plan_execution_log (created_at);
