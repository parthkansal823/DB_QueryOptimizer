#!/usr/bin/env bash
# Stretch goal: load the real Join Order Benchmark (JOB) dataset into a
# second database ("job") in the same Postgres container, so the existing
# pipeline (hints.py -> plan_extractor.py -> features.py -> train.py) can
# run against it with zero code changes -- that's what schema_introspection
# and the scan_relations-based featurization in Phase 2 were built for.
#
# Prerequisites (see README.md in this directory for the exact commands):
#   1. `docker compose up -d postgres` running
#   2. data/job/imdb.tgz downloaded from http://event.cwi.nl/da/job/imdb.tgz
#      and extracted into data/job/csv/ (21 .csv files, ~3.6GB)
#   3. data/job/queries/ cloned from
#      https://github.com/gregrahn/join-order-benchmark (schema.sql,
#      fkindexes.sql, and the 113 JOB query files)
#
# Usage: bash data/job/load_job.sh
set -euo pipefail
export MSYS_NO_PATHCONV=1  # stop Git Bash from mangling /job_data/... into a Windows path

cd "$(dirname "$0")/../.."  # repo root, so `docker compose` finds docker-compose.yml
DC="docker compose"
DB=job

TABLES="aka_name aka_title cast_info char_name comp_cast_type company_name company_type \
complete_cast info_type keyword kind_type link_type movie_companies movie_info \
movie_info_idx movie_keyword movie_link name person_info role_type title"

echo "==> (re)creating database '$DB'"
$DC exec -T postgres psql -U postgres -c "DROP DATABASE IF EXISTS $DB"
$DC exec -T postgres psql -U postgres -c "CREATE DATABASE $DB"

echo "==> loading schema"
$DC exec -T postgres psql -U postgres -d "$DB" -v ON_ERROR_STOP=1 < data/job/queries/schema.sql

echo "==> creating plan_execution_log (app.collect_data's bookkeeping table --
    only exists in the default 'lqo' database by default, via postgres/init/,
    so the JOB database needs it applied explicitly)"
$DC exec -T postgres psql -U postgres -d "$DB" -v ON_ERROR_STOP=1 < postgres/init/03_logging.sql

echo "==> loading data (21 tables, ~3.6GB -- this takes a while)"
# FORMAT csv on its own assumes RFC4180 quoting ("" to escape a literal
# quote); these CSVs were exported with backslash-escaped quotes instead
# (a documented quirk of this dataset), so ESCAPE '\' has to be explicit or
# COPY chokes on the first embedded-quote row (e.g. aka_name.csv line ~126725).
for t in $TABLES; do
  echo "  -> $t"
  $DC exec -T postgres psql -U postgres -d "$DB" -v ON_ERROR_STOP=1 \
    -c "\\copy $t from '/job_data/csv/$t.csv' with (format csv, escape '\\')"
done

echo "==> foreign key indexes"
$DC exec -T postgres psql -U postgres -d "$DB" -v ON_ERROR_STOP=1 < data/job/queries/fkindexes.sql

echo "==> analyze"
$DC exec -T postgres psql -U postgres -d "$DB" -c "ANALYZE"

echo "==> done. Row counts:"
$DC exec -T postgres psql -U postgres -d "$DB" -c \
  "SELECT relname, n_live_tup FROM pg_stat_user_tables ORDER BY relname"
