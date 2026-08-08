# JOB/IMDB stretch goal

Loads the real [Join Order Benchmark](https://github.com/gregrahn/join-order-benchmark)
(JOB) dataset -- 21 IMDB tables, 113 hand-crafted multi-way-join queries --
into a second database (`job`) in the same Postgres container, so this
project's pipeline can be evaluated against the same benchmark used by the
Neo and Bao papers, not just the synthetic schema.

The pipeline itself (`hints.py`, `plan_extractor.py`, `features.py`,
`train.py`) needed **zero code changes** to support this -- see
`docs/WRITEUP.md` Section 3 for why.

## 1. Download the data

```sh
mkdir -p data/job
curl -o data/job/imdb.tgz -L http://event.cwi.nl/da/job/imdb.tgz   # ~1.2GB
mkdir -p data/job/csv
tar -xzf data/job/imdb.tgz -C data/job/csv                          # ~3.6GB uncompressed
```

Source: the join-order-benchmark repo's README points here as "the CSV
files used in the paper, from May 2013." If this mirror ever goes down, the
same README documents a from-scratch rebuild via `imdbpy2sql.py` against a
current IMDB dump (more work, and the repo warns query results may differ
slightly from the frozen May-2013 snapshot the queries were tuned against).

## 2. Get the schema + queries

```sh
git clone --depth 1 https://github.com/gregrahn/join-order-benchmark.git data/job/queries
```

This provides `schema.sql` (21 `CREATE TABLE`s), `fkindexes.sql` (foreign
key indexes, applied after data load so the bulk COPY isn't slowed by index
maintenance), and the 113 `<n><letter>.sql` query files (e.g. `10a.sql`).

## 3. Load it

```sh
docker compose up -d postgres   # needs the ./data/job:/job_data mount in docker-compose.yml
bash data/job/load_job.sh
```

`load_job.sh` drops and recreates a `job` database each run (idempotent,
safe to rerun), loads the schema, `\copy`s each of the 21 CSVs, adds the FK
indexes, and runs `ANALYZE`. One thing worth knowing if you're debugging a
COPY failure: **these CSVs use backslash-escaped quotes, not standard
RFC4180 doubled-quotes** -- `load_job.sh` passes `ESCAPE '\'` explicitly;
without it, COPY throws `extra data after last expected column` on the
first row containing an embedded quote (e.g. `aka_name.csv` around line
126725).

## 4. Point the pipeline at it

```sh
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/job
docker compose exec -e DATABASE_URL=postgresql://postgres:postgres@postgres:5432/job backend python -m app.collect_data --reps 1
docker compose exec -e DATABASE_URL=postgresql://postgres:postgres@postgres:5432/job backend python -m app.train
```

`collect_data.py` runs against `workload.py` by default, which is written
for the synthetic schema's tables -- against `job`, point it at (a subset
of) the 113 files in `data/job/queries/*.sql` instead. A trained model
against `job` writes to the same `models/plan_selector.pkl` path as the
synthetic schema's model, so keep the two runs' pickles somewhere separate
(e.g. `models/plan_selector.job.pkl`) if you want to compare rather than
overwrite.

## Known limitation

`features.py` gives each *table* one feature slot, keyed by relation name.
JOB queries frequently self-join the same table under multiple aliases
(`movie_info AS mi1, movie_info AS mi2`); those collapse into one slot
rather than getting per-occurrence features. See `docs/WRITEUP.md` for why
that's a documented tradeoff rather than a bug.
