FROM postgres:16

RUN apt-get update && apt-get install -y \
    build-essential \
    postgresql-server-dev-16 \
    git \
    && rm -rf /var/lib/apt/lists/*

# pg_hint_plan's branch must match your Postgres major version (REL16 here).
# If this has moved on by the time you build it, check
# https://github.com/ossc-db/pg_hint_plan for the branch matching PG 16.
RUN git clone --branch REL16 --depth 1 https://github.com/ossc-db/pg_hint_plan.git /tmp/pg_hint_plan \
    && cd /tmp/pg_hint_plan \
    && make && make install \
    && rm -rf /tmp/pg_hint_plan
