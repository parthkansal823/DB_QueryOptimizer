import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ExperimentPanel from "./ExperimentPanel";
import { fetchSettings } from "../api";

vi.mock("../api", () => ({ fetchSettings: vi.fn() }));

// Every test in this file renders the panel, and the panel asks the backend for
// its policy list on mount. Without a default the mock returns `undefined` and
// the `.then` throws -- which is a mock problem masquerading as seven failures
// in tests that have nothing to do with policies.
beforeEach(() => {
  fetchSettings.mockReset();
  fetchSettings.mockResolvedValue({
    fields: [{ name: "selection_policy", options: ["greedy", "pairwise_rank"] }],
  });
});

// The panel exists so the *rigorous* comparison stops being the inconvenient
// one. Every other panel reports a single sweep; this one interleaves two arms
// and reports an interval. The thing it must never do is let a good-looking
// mean read as a settled result -- docs/WRITEUP.md 2.4 records a headline
// number that rested on three runs.

const resolved = {
  status: "done",
  label: "guard on vs guard off",
  is_running: false,
  elapsed_seconds: 412.0,
  result: {
    verdict: "guard on is better",
    mean_difference: 27.4,
    confidence_interval: [11.2, 41.9],
    sign_test_p_value: 0.004,
    n_pairs: 20,
  },
};

const unresolved = {
  ...resolved,
  result: {
    verdict: "unresolved",
    mean_difference: 6.1,
    confidence_interval: [-9.4, 21.8],
    sign_test_p_value: 0.42,
    n_pairs: 8,
  },
};

describe("settling a comparison", () => {
  it("reports the interval and the sign test, not just the mean", () => {
    render(<ExperimentPanel status={resolved} onStart={vi.fn()} />);

    expect(screen.getByText(/guard on is better/)).toBeInTheDocument();
    expect(screen.getByText(/\[11\.20, 41\.90\]/)).toBeInTheDocument();
    expect(screen.getByText(/p = 0\.004/)).toBeInTheDocument();
    expect(screen.getByText("20")).toBeInTheDocument();
  });

  it("says plainly that an interval spanning zero has not resolved", () => {
    // The whole point. A mean of +6.1 looks like a win and is not one.
    render(<ExperimentPanel status={unresolved} onStart={vi.fn()} />);

    expect(screen.getByText(/has not resolved/)).toBeInTheDocument();
    expect(screen.getByText(/More paired runs would narrow it/)).toBeInTheDocument();
  });

  it("does not claim a result before one exists", () => {
    render(<ExperimentPanel status={null} onStart={vi.fn()} />);

    expect(screen.queryByText(/Verdict/)).not.toBeInTheDocument();
  });

  it("warns that a comparison is slow before one is started", () => {
    render(<ExperimentPanel status={null} onStart={vi.fn()} />);

    expect(screen.getByText(/two full benchmark sweeps/)).toBeInTheDocument();
  });

  it("starts the comparison the form describes", async () => {
    const onStart = vi.fn().mockResolvedValue({});
    render(<ExperimentPanel status={null} onStart={onStart} />);

    await userEvent.selectOptions(screen.getByLabelText("Compare"), "policy");
    await userEvent.selectOptions(screen.getByLabelText("Arm A"), "greedy");
    await userEvent.click(screen.getByRole("button", { name: /Run comparison/ }));

    expect(onStart).toHaveBeenCalledWith(
      expect.objectContaining({ comparison: "policy", arm_a: "greedy" }),
    );
  });

  it("will not start a second comparison while one is running", () => {
    render(
      <ExperimentPanel
        status={{ is_running: true, label: "guard on vs guard off" }}
        onStart={vi.fn()}
      />,
    );

    expect(screen.getByRole("button")).toBeDisabled();
  });

  it("surfaces a failed comparison rather than showing nothing", () => {
    render(
      <ExperimentPanel
        status={{ status: "failed", is_running: false, error: "postgres went away" }}
        onStart={vi.fn()}
      />,
    );

    expect(screen.getByText(/postgres went away/)).toBeInTheDocument();
  });
});

// The policy list is the backend's, not this file's.
//
// It was spelled out here three times and `borda` was added to the seven backend
// places and not to this one, so the dashboard offered an A/B over four policies
// while the server ran five. A panel that cannot select the policy you want to
// compare quietly decides the comparison for you.
describe("the policies it offers", () => {
  it("offers whatever the backend declares", async () => {
    fetchSettings.mockResolvedValue({
      fields: [
        { name: "selection_policy", options: ["greedy", "pairwise_rank", "borda"] },
      ],
    });

    render(<ExperimentPanel status={null} onStart={vi.fn()} />);

    await waitFor(() =>
      expect(screen.getByRole("option", { name: "borda" })).toBeInTheDocument(),
    );
  });

  it("falls back to a known list when the settings call fails", async () => {
    // Stale is not the same as broken: a panel that cannot reach the API still
    // has to render something selectable.
    fetchSettings.mockRejectedValue(new Error("offline"));

    render(<ExperimentPanel status={null} onStart={vi.fn()} />);

    await waitFor(() =>
      expect(screen.getByRole("option", { name: "pairwise_rank" })).toBeInTheDocument(),
    );
  });

  it("ignores a settings response with no policy field", async () => {
    fetchSettings.mockResolvedValue({ fields: [{ name: "confidence_z" }] });

    render(<ExperimentPanel status={null} onStart={vi.fn()} />);

    await waitFor(() =>
      expect(screen.getByRole("option", { name: "greedy" })).toBeInTheDocument(),
    );
  });
});
