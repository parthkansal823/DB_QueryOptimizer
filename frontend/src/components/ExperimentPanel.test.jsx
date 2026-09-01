import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import ExperimentPanel from "./ExperimentPanel";

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
