import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import DashboardPage from "./DashboardPage";
import * as api from "./api";

vi.mock("./api");

const FIELDS = [
  {
    name: "selection_policy",
    type: "enum",
    options: ["greedy", "thompson", "pairwise_rank"],
    label: "Selection policy",
    group: "Model",
    help: "How the model turns predictions into a choice.",
  },
  {
    name: "min_relative_gain",
    type: "float",
    min: 0,
    max: 0.9,
    step: 0.01,
    label: "Minimum predicted speedup",
    group: "Safety gate",
    help: "A hinted plan must look at least this much faster.",
  },
  {
    name: "enable_rows_correction",
    type: "bool",
    label: "Learned cardinality corrections",
    group: "Action space",
    help: "Adds a Rows(...) candidate.",
  },
];

const SETTINGS = {
  fields: FIELDS,
  values: { selection_policy: "greedy", min_relative_gain: 0.05, enable_rows_correction: true },
  groups: ["Model", "Safety gate", "Action space"],
};

const DATABASES = {
  active_url: "postgresql://postgres:****@postgres:5432/lqo",
  allow_runtime_change: true,
  profiles: [{ name: "lqo", url: "postgresql://postgres:****@postgres:5432/lqo", active: true }],
  model_fit: { model_trained: true, matches: true, missing_tables: [], new_tables: [] },
};

beforeEach(() => {
  vi.resetAllMocks();
  api.fetchSettings.mockResolvedValue(SETTINGS);
  api.fetchDatabases.mockResolvedValue(DATABASES);
  api.fetchSavedQueries.mockResolvedValue({ queries: [], builtin_count: 24 });
  api.fetchTrainStatus.mockResolvedValue({ status: "idle", is_running: false, log: [] });
});

describe("rendering from backend metadata", () => {
  it("renders a control per declared field, grouped", async () => {
    // The page knows nothing about individual settings: a field added to
    // app/settings.py must appear here with no frontend change.
    render(<DashboardPage />);

    expect(await screen.findByLabelText("Selection policy")).toBeInTheDocument();
    expect(screen.getByLabelText("Minimum predicted speedup")).toBeInTheDocument();
    expect(screen.getByLabelText("Learned cardinality corrections")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Safety gate" })).toBeInTheDocument();
  });

  it("shows the current value, not the default", async () => {
    render(<DashboardPage />);
    expect(await screen.findByLabelText("Selection policy")).toHaveValue("greedy");
  });
});

describe("changing a setting", () => {
  it("sends only the changed field", async () => {
    api.updateSettings.mockResolvedValue({
      values: { ...SETTINGS.values, selection_policy: "thompson" },
      applied: true,
    });
    render(<DashboardPage />);

    await userEvent.selectOptions(await screen.findByLabelText("Selection policy"), "thompson");

    await waitFor(() =>
      expect(api.updateSettings).toHaveBeenCalledWith({ selection_policy: "thompson" }),
    );
  });

  it("reports a rejected value instead of showing it as applied", async () => {
    // The backend validates ranges; a 422 must not leave the UI displaying a
    // value the optimizer never accepted.
    api.updateSettings.mockRejectedValue(new Error("422: confidence_z must be <= 5.0"));
    render(<DashboardPage />);

    await userEvent.selectOptions(await screen.findByLabelText("Selection policy"), "thompson");

    expect(await screen.findByText(/must be <= 5.0/)).toBeInTheDocument();
  });
});

describe("the model/schema mismatch warning", () => {
  it("stays quiet when the model matches the connected database", async () => {
    render(<DashboardPage />);
    await screen.findByLabelText("Selection policy");
    expect(screen.queryByText(/trained on a different schema/i)).not.toBeInTheDocument();
  });

  it("warns when the served model was trained elsewhere", async () => {
    // Without this the dashboard reports learned decisions from a model whose
    // per-table feature slots refer to tables that are not there.
    api.fetchDatabases.mockResolvedValue({
      ...DATABASES,
      model_fit: {
        model_trained: true,
        matches: false,
        missing_tables: ["orders", "users"],
        new_tables: ["title"],
      },
    });
    render(<DashboardPage />);

    expect(await screen.findByText(/trained on a different schema/i)).toBeInTheDocument();
    expect(screen.getByText("orders, users")).toBeInTheDocument();
  });
});

describe("databases", () => {
  it("never displays a password", async () => {
    // The backend redacts before sending; this asserts the page shows what it
    // was given and that no credential reaches the DOM, where it would end up
    // in screenshots and browser network logs.
    render(<DashboardPage />);

    const shown = await screen.findAllByText(/postgres:\*\*\*\*@postgres:5432\/lqo/);
    expect(shown.length).toBeGreaterThan(0);
    expect(document.body.textContent).not.toMatch(/postgres:postgres@/);
  });

  it("does not offer to deactivate the live connection", async () => {
    render(<DashboardPage />);
    await screen.findByLabelText("Selection policy");
    expect(screen.queryByRole("button", { name: "Make active" })).not.toBeInTheDocument();
  });

  it("says so when runtime switching is disabled", async () => {
    api.fetchDatabases.mockResolvedValue({ ...DATABASES, allow_runtime_change: false });
    render(<DashboardPage />);
    expect(await screen.findByText(/Runtime switching is disabled/i)).toBeInTheDocument();
  });

  it("flags a database without pg_hint_plan when testing it", async () => {
    // Without the extension every hint is silently a comment and no candidate
    // can differ from native -- the most expensive failure in this project.
    api.testDatabase.mockResolvedValue({
      ok: true,
      server_version: "PostgreSQL 16.2",
      n_tables: 21,
      pg_hint_plan: false,
    });
    render(<DashboardPage />);

    await userEvent.type(await screen.findByLabelText("Connection URL"), "postgresql://x@y/z");
    await userEvent.click(screen.getByRole("button", { name: "Test" }));

    expect(await screen.findByText(/pg_hint_plan is not installed/i)).toBeInTheDocument();
  });
});

describe("saved queries", () => {
  it("lists the queries the model will be trained on", async () => {
    api.fetchSavedQueries.mockResolvedValue({
      queries: [{ name: "orders-by-country", sql: "SELECT 1", description: "India orders" }],
      builtin_count: 24,
    });
    render(<DashboardPage />);

    expect(await screen.findByText("orders-by-country")).toBeInTheDocument();
    expect(screen.getByText(/India orders/)).toBeInTheDocument();
  });

  it("warns that a single-table query has nothing to optimize", async () => {
    // There is no join order to choose between, so it can never produce a
    // candidate -- better said at Check time than after a 20-minute run.
    api.validateQuery.mockResolvedValue({
      ok: true,
      n_tables: 1,
      tables: ["orders"],
      estimated_cost: 100,
      joins_available: false,
    });
    render(<DashboardPage />);

    await userEvent.type(await screen.findByLabelText("SQL"), "SELECT * FROM orders");
    await userEvent.click(screen.getByRole("button", { name: "Check" }));

    expect(await screen.findByText(/no join order to optimize/i)).toBeInTheDocument();
  });

  it("refuses a query that would modify data", async () => {
    api.validateQuery.mockResolvedValue({
      ok: false,
      error: "only read-only queries can be saved: they are executed repeatedly",
    });
    render(<DashboardPage />);

    await userEvent.type(await screen.findByLabelText("SQL"), "DELETE FROM orders");
    await userEvent.click(screen.getByRole("button", { name: "Check" }));

    expect(await screen.findByText(/only read-only queries can be saved/i)).toBeInTheDocument();
  });

  it("cannot save without both a name and SQL", async () => {
    render(<DashboardPage />);
    await userEvent.type(await screen.findByLabelText("SQL"), "SELECT 1");
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
  });
});

describe("training", () => {
  it("will not start with nothing to train on", async () => {
    render(<DashboardPage />);
    expect(await screen.findByText(/Save a query first/i)).toBeInTheDocument();
  });

  it("counts the built-in workload only when it is included", async () => {
    api.fetchSavedQueries.mockResolvedValue({
      queries: [{ name: "q1", sql: "SELECT 1", description: "" }],
      builtin_count: 24,
    });
    render(<DashboardPage />);

    expect(await screen.findByRole("button", { name: /Start training on 1 query/ })).toBeInTheDocument();

    await userEvent.click(screen.getByLabelText(/include the 24 built-in/i));

    expect(screen.getByRole("button", { name: /Start training on 25 queries/ })).toBeInTheDocument();
  });

  it("shows progress while a run is in flight", async () => {
    api.fetchTrainStatus.mockResolvedValue({
      status: "running",
      stage: "collecting",
      done: 3,
      total: 10,
      rows_collected: 412,
      elapsed_seconds: 47.2,
      is_running: true,
      log: ["10:00:01  [3/10] q3 — 412 rows logged"],
    });
    render(<DashboardPage />);

    // The status line is assembled from several sibling text nodes, so it is
    // asserted as one string rather than matched node by node.
    await screen.findByText(/3\/10 queries/);
    expect(document.body.textContent).toMatch(/3\/10 queries, 412 rows/);
    expect(document.body.textContent).toMatch(/collecting/);
  });

  it("says stopped runs keep their collected rows", async () => {
    api.fetchTrainStatus.mockResolvedValue({
      status: "stopped", stage: "collecting", done: 4, total: 10,
      rows_collected: 500, is_running: false, log: [],
    });
    render(<DashboardPage />);

    expect(await screen.findByText(/rows collected so far are kept/i)).toBeInTheDocument();
  });

  it("surfaces a failed run instead of looking idle", async () => {
    api.fetchTrainStatus.mockResolvedValue({
      status: "failed", stage: "training", done: 10, total: 10, rows_collected: 900,
      is_running: false, error: "RuntimeError: plan_execution_log is empty", log: [],
    });
    render(<DashboardPage />);

    expect(await screen.findByText(/plan_execution_log is empty/)).toBeInTheDocument();
  });
});
