"""
Registry of real, publicly available benchmark datasets.

Why this exists: measurement showed the synthetic e-commerce schema has an
oracle ceiling of ~6.5% -- only 7 of its 25 queries have *any* plan that
beats PostgreSQL by more than 5%, and no model class tested could pick the
best plan even once (see docs/WRITEUP.md §2.9). A learned optimizer cannot
demonstrate anything on a workload where the built-in optimizer is already
right. JOB, by contrast, showed ~75% available headroom and models that
actually rank.

So the dataset is a first-class input, not a fixture. Each entry below is a
real benchmark from a public source, with the commands to obtain and load
it. `python -m app.onboard` then works against any of them unchanged.

Adding one means appending a dict here -- nothing in the optimizer needs to
know which dataset it is pointed at.
"""

from __future__ import annotations

DATASETS = {
    "job": {
        "name": "Join Order Benchmark (IMDB)",
        "source": "http://event.cwi.nl/da/job/imdb.tgz",
        "queries": "https://github.com/gregrahn/join-order-benchmark",
        "paper": "Leis et al., 'How Good Are Query Optimizers, Really?', VLDB 2015",
        "tables": 21,
        "approx_rows": 74_000_000,
        "download_gb": 1.2,
        "why": (
            "Built specifically to break cardinality estimators: correlated "
            "predicates, up to 17-way joins, real skewed data. Measured oracle "
            "headroom here is ~75%, versus ~6.5% on the synthetic schema."
        ),
        "loader": "bash data/job/load_job.sh",
        "status": "implemented -- see data/job/README.md",
    },
    "tpch": {
        "name": "TPC-H",
        "source": "https://github.com/electrum/tpch-dbgen (generator)",
        "paper": "TPC-H, the standard decision-support benchmark",
        "tables": 8,
        "approx_rows": "scale-factor dependent (SF1 ~ 8.7M rows)",
        "download_gb": 0.0,  # generated locally, nothing to download
        "why": (
            "The industry-standard join benchmark. Eight tables in a snowflake "
            "shape with a well-understood join graph, and a scale factor that "
            "lets you dial data volume up until join order genuinely matters. "
            "Generated locally, so there is no large download."
        ),
        "loader": "bash data/tpch/load_tpch.sh [scale_factor]",
        "status": "implemented -- see data/tpch/README.md",
    },
    "stack": {
        "name": "Stack Exchange / StackOverflow dump",
        "source": "https://archive.org/details/stackexchange",
        "why": (
            "Real application data with genuinely skewed distributions (a few "
            "users answer everything; most posts get no answers). Smaller "
            "sites like dba.stackexchange are a few hundred MB, so a realistic "
            "workload without IMDB's download."
        ),
        "loader": "not implemented -- XML dumps need conversion first",
        "status": "documented for future work",
    },
    "dsb": {
        "name": "DSB (Decision Support Benchmark)",
        "source": "https://github.com/microsoft/dsb",
        "paper": "Ding et al., 'DSB: A Decision Support Benchmark for "
                 "Workload-Driven and Traditional Database Systems', VLDB 2021",
        "why": (
            "TPC-DS reworked specifically to stress cardinality estimation, "
            "with correlated and skewed data plus a query generator. The "
            "closest thing to a purpose-built benchmark for this project."
        ),
        "loader": "not implemented -- builds on the TPC-DS toolkit",
        "status": "documented for future work",
    },
}


def describe(dataset: str | None = None) -> str:
    entries = DATASETS if dataset is None else {dataset: DATASETS[dataset]}
    lines = []
    for key, meta in entries.items():
        lines.append(f"{key}: {meta['name']}  [{meta['status']}]")
        lines.append(f"    source: {meta['source']}")
        lines.append(f"    why:    {meta['why']}")
        lines.append(f"    load:   {meta['loader']}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
