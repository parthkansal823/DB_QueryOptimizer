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
