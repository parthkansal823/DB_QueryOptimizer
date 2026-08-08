#!/usr/bin/env bash
# Creates a TPC-H database in the running Postgres container.
#
#   bash data/tpch/load_tpch.sh [scale_factor]
#
# Scale factor 0.1 (~600k line items) takes about a minute and is enough for
# join order to matter. 1.0 is the standard SF1 (~6M line items) and takes
# considerably longer. Everything is generated in SQL, so nothing downloads.
set -euo pipefail
export MSYS_NO_PATHCONV=1

cd "$(dirname "$0")/../.."
SF="${1:-0.1}"
DB=tpch
DC="docker compose"

echo "==> (re)creating database '$DB' at scale factor $SF"
$DC exec -T postgres psql -U postgres -c "DROP DATABASE IF EXISTS $DB"
$DC exec -T postgres psql -U postgres -c "CREATE DATABASE $DB"

echo "==> schema"
$DC exec -T postgres psql -U postgres -d "$DB" -v ON_ERROR_STOP=1 < data/tpch/schema.sql

echo "==> feedback table (app.collect_data logs here)"
$DC exec -T postgres psql -U postgres -d "$DB" -v ON_ERROR_STOP=1 < postgres/init/03_logging.sql

echo "==> generating data (this is the slow part)"
$DC exec -T postgres psql -U postgres -d "$DB" -v ON_ERROR_STOP=1 -v sf="$SF" < data/tpch/generate.sql

echo "==> row counts"
$DC exec -T postgres psql -U postgres -d "$DB" -c \
  "SELECT relname, n_live_tup FROM pg_stat_user_tables ORDER BY n_live_tup DESC"

echo
echo "Done. Point the optimizer at it with:"
echo "  docker compose exec -e DATABASE_URL=postgresql://postgres:postgres@postgres:5432/$DB \\"
echo "      backend python -m app.onboard --queries 20 --reps 3"
