CREATE EXTENSION IF NOT EXISTS pg_hint_plan;

-- Hypothetical indexes, so the advisor can measure what an index would be
-- worth instead of only noticing that one is missing. Planner-only: it
-- writes no pages and holds no lock. See backend/app/hypothetical.py.
CREATE EXTENSION IF NOT EXISTS hypopg;
