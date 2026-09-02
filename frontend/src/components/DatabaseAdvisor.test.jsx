import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import DatabaseAdvisor from "./DatabaseAdvisor";

// Findings about the database rather than about one query's plan. The rule the
// panel has to keep is that nothing is applied: each finding shows the
// statement that would apply it, because DDL against someone's database is
// their call and not a button this dashboard owns.

const noop = vi.fn();

function panel(findings, extra = {}) {
  return render(
    <DatabaseAdvisor
      findings={findings}
      onRefresh={noop}
      busy={false}
      deep={false}
      onDeepChange={noop}
      {...extra}
    />,
  );
}

describe("database health", () => {
  it("offers to check before anything has been checked", () => {
    panel(null);

    expect(screen.getByRole("button", { name: /Check the database/ })).toBeInTheDocument();
  });

  it("says so plainly when there is nothing to report", () => {
    panel({ n_findings: 0 });

    expect(screen.getByText(/Nothing to report/)).toBeInTheDocument();
  });

  it("shows the measured estimate error for a correlated column pair", () => {
    panel({
      n_findings: 1,
      extended_statistics: [
        {
          table: "users",
          columns: ["city", "country"],
          measured: true,
          qerror_before: 340,
          qerror_after: 1.2,
          definition: 'CREATE STATISTICS "s" (dependencies) ON "city", "country" FROM "users"',
        },
      ],
    });

    expect(screen.getByText("users(city, country)")).toBeInTheDocument();
    expect(screen.getByText(/340× → 1.2×/)).toBeInTheDocument();
  });

  it("shows the statement rather than applying it", () => {
    panel({
      n_findings: 1,
      server_config: {
        machine_memory: "32.0GB",
        findings: [
          {
            setting: "shared_buffers",
            current: "128MB",
            suggested: "8.0GB",
            note: "a default that was never chosen",
            fix: "ALTER SYSTEM SET shared_buffers = '8.0GB';",
          },
        ],
      },
    });

    expect(screen.getByText(/ALTER SYSTEM SET shared_buffers/)).toBeInTheDocument();
    // Nothing that would run it.
    expect(screen.queryByRole("button", { name: /Apply/ })).not.toBeInTheDocument();
  });

  it("warns that dropping a unique index changes what the database accepts", () => {
    panel({
      n_findings: 1,
      indexes: {
        unused: [
          {
            index: "users_email_key",
            table: "users",
            enforces_uniqueness: true,
            note: "never chosen by the planner",
          },
        ],
        duplicates: [],
        redundant: [],
      },
    });

    expect(screen.getByText(/also enforces uniqueness/)).toBeInTheDocument();
  });

  it("explains that measuring correlated columns is slow before doing it", () => {
    panel({ n_findings: 0 }, { deep: true });

    expect(screen.getByText(/slow on large tables/)).toBeInTheDocument();
    expect(screen.getByText(/Nothing is left behind/)).toBeInTheDocument();
  });

  it("reports a candidate it could not measure rather than hiding it", () => {
    panel({
      n_findings: 1,
      extended_statistics: [
        {
          table: "cast_info",
          columns: ["a", "b"],
          measured: false,
          reason: "table has 36,000,000 rows; measuring would need a full ANALYZE",
          definition: "CREATE STATISTICS ...",
        },
      ],
    });

    expect(screen.getByText(/would need a full ANALYZE/)).toBeInTheDocument();
  });
});

// -- maintenance and workload sections --------------------------------------
//
// Neither existed until the PostgreSQL manual turned up the state they read.
// The visibility map is the one that ties back to the optimizer: it decides
// whether an index-only scan can skip the heap, so a stale one is the reason an
// IndexOnlyScan candidate lost rather than a curiosity about vacuum.

describe("maintenance findings", () => {
  it("shows a partitioned parent autovacuum will never analyze", () => {
    render(
      <DatabaseAdvisor
        findings={{
          n_findings: 1,
          maintenance: {
            never_analyzed: [
              {
                table: "events",
                kind: "partitioned table",
                severity: "high",
                fix: 'ANALYZE "events"',
                note: "changes to child tables do not trigger autoanalyze on the parent",
              },
            ],
          },
        }}
      />,
    );

    expect(screen.getByText("events")).toBeInTheDocument();
    expect(screen.getByText(/partitioned table, never analyzed/)).toBeInTheDocument();
    expect(screen.getByText(/do not trigger autoanalyze/)).toBeInTheDocument();
  });

  it("reports how close a table is to a forced anti-wraparound vacuum", () => {
    render(
      <DatabaseAdvisor
        findings={{
          n_findings: 1,
          maintenance: {
            wraparound_risk: [
              {
                table: "orders",
                severity: "high",
                fraction_of_limit: 0.95,
                fix: 'VACUUM (FREEZE, ANALYZE) "orders"',
                note: "the oldest unfrozen row here is 190,000,000 transactions old",
              },
            ],
          },
        }}
      />,
    );

    expect(screen.getByText(/95% of the way to a forced/)).toBeInTheDocument();
  });

  it("explains why an index-only scan would not pay yet", () => {
    render(
      <DatabaseAdvisor
        findings={{
          n_findings: 1,
          maintenance: {
            index_only_scan_readiness: [
              {
                table: "churn",
                visible_fraction: 0.0,
                fix: 'VACUUM (ANALYZE) "churn"',
                note: "an index-only scan would still have to check the heap",
              },
            ],
          },
        }}
      />,
    );

    expect(screen.getByText(/0% of pages marked/)).toBeInTheDocument();
  });

  it("says nothing when maintenance is healthy", () => {
    render(<DatabaseAdvisor findings={{ n_findings: 0, maintenance: {} }} />);
    expect(screen.queryByText("Maintenance")).not.toBeInTheDocument();
  });
});

describe("workload findings", () => {
  it("labels the optimizer's own candidate sweep", () => {
    // On a dashboard-driven database the sweep *is* most of the workload.
    // Hiding it would misrepresent the panel; leaving it unlabelled would make
    // it read as user traffic.
    render(
      <DatabaseAdvisor
        findings={{
          n_findings: 1,
          workload: {
            spilling: [
              {
                query: "EXPLAIN (FORMAT JSON, ANALYZE) SELECT ...",
                calls: 145,
                temp_blocks_written: 118088,
                from_optimizer_sweep: true,
                note: "some plan for this query wrote 118,088 temporary blocks",
              },
            ],
          },
        }}
      />,
    );

    expect(screen.getByText(/this optimizer's own candidate sweep/)).toBeInTheDocument();
    expect(screen.getByText(/118,088 temp blocks written/)).toBeInTheDocument();
  });

  it("repeats the caveat that a row folds every plan variant together", () => {
    render(
      <DatabaseAdvisor
        findings={{
          n_findings: 1,
          workload: {
            noisy: [
              {
                query: "SELECT ...",
                calls: 145,
                relative_stddev: 1.05,
                from_optimizer_sweep: false,
                note: "most of that spread is the difference between plans",
              },
            ],
          },
        }}
      />,
    );

    expect(screen.getByText(/folds together every plan variant/)).toBeInTheDocument();
  });

  it("shows why the workload panel is empty when the extension is missing", () => {
    // An empty panel and "not installed" are different answers.
    render(
      <DatabaseAdvisor
        findings={{
          n_findings: 0,
          workload: { note: "pg_stat_statements is not installed." },
        }}
      />,
    );

    expect(screen.getByText(/pg_stat_statements is not installed/)).toBeInTheDocument();
  });
});
