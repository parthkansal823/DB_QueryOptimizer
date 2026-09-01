FROM postgres:16

RUN apt-get update && apt-get install -y \
    build-essential \
    postgresql-server-dev-16 \
    git \
    && rm -rf /var/lib/apt/lists/*

# pg_hint_plan's branch must match your Postgres major version (REL16 here).
# If this has moved on by the time you build it, check
# https://github.com/ossc-db/pg_hint_plan for the branch matching PG 16.
RUN git clone --branch PG16 --depth 1 https://github.com/ossc-db/pg_hint_plan.git /tmp/pg_hint_plan \
    && cd /tmp/pg_hint_plan \
    && make && make install \
    && rm -rf /tmp/pg_hint_plan

# HypoPG: hypothetical indexes. Registers an index in the planner's catalogue
# only -- no pages written, no lock taken, nothing outliving the session -- so
# `EXPLAIN` can be asked what an index *would* do without building it. That is
# what turns app/advisor.py's "these indexes are missing" into "this one would
# cut the plan's cost by 78% and the planner would actually use it".
RUN git clone --branch REL1_4_STABLE --depth 1 https://github.com/HypoPG/hypopg.git /tmp/hypopg     && cd /tmp/hypopg     && make && make install     && rm -rf /tmp/hypopg
