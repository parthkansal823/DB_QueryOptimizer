FROM postgres:18

# `flex` and `bison` are not optional on PG18. pg_hint_plan's PG16 branch
# shipped a pre-generated `query_scan.c`; the PG18 branch generates it from
# `query_scan.l` at build time, and without flex the build fails with
# "make: /usr/bin/flex: No such file or directory" -- after every C file has
# already compiled, so the log reads like a successful build until its last line.
RUN apt-get update && apt-get install -y \
    build-essential \
    postgresql-server-dev-18 \
    flex \
    bison \
    git \
    # pg_overexplain (PG18, contrib): reports range-table entries, per-node
    # plan internals and relation OIDs that no EXPLAIN option exposes. Not in
    # the base image.
    postgresql-contrib-18 \
    && rm -rf /var/lib/apt/lists/*

# pg_hint_plan's branch must match your Postgres major version (PG18 here).
# If this has moved on by the time you build it, check
# https://github.com/ossc-db/pg_hint_plan for the branch matching your PG.
#
# The move from PG16 to PG18 was not cosmetic. On PG16 this project ran
# pg_hint_plan **1.6.3** while the manual it was being read against
# documented a newer one, and the gap was silent: `DisableIndex` simply did
# not exist, and the rejection arrived at INFO severity, which the default
# `client_min_messages` discards. See docs/WRITEUP.md 2.20.
RUN git clone --branch PG18 --depth 1 https://github.com/ossc-db/pg_hint_plan.git /tmp/pg_hint_plan \
    && cd /tmp/pg_hint_plan \
    && make && make install \
    && rm -rf /tmp/pg_hint_plan

# HypoPG: hypothetical indexes. Registers an index in the planner's catalogue
# only -- no pages written, no lock taken, nothing outliving the session -- so
# `EXPLAIN` can be asked what an index *would* do without building it. That is
# what turns app/advisor.py's "these indexes are missing" into "this one would
# cut the plan's cost by 78% and the planner would actually use it".
#
# Cloned from the default branch rather than a pinned one: HypoPG's release
# branches are not named the way pg_hint_plan's are, and a wrong `--branch`
# fails the clone with a git error that `docker compose build` still reports as
# an overall success -- the image builds without the extension, and the failure
# only surfaces later as "could not open extension control file".
RUN git clone --depth 1 https://github.com/HypoPG/hypopg.git /tmp/hypopg \
    && cd /tmp/hypopg \
    && make && make install \
    && rm -rf /tmp/hypopg
