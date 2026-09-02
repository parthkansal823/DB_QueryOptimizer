import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import HowToUse from "./HowToUse";

// The instructions the page never had. They are worth having and they are not
// what anyone arrives to read, so they sit below the query box and start
// folded -- open, they took the whole first screen and pushed the one control
// that does anything below the fold.

async function open() {
  render(<HowToUse />);
  await userEvent.click(screen.getByRole("button", { name: "Show" }));
}

describe("how to use", () => {
  it("starts folded, with a line saying what is inside", () => {
    render(<HowToUse />);

    expect(screen.getByText(/what to read, and which numbers to trust/)).toBeInTheDocument();
    expect(screen.queryByText(/Paste a SELECT and analyze it/)).not.toBeInTheDocument();
  });

  it("opens when asked", async () => {
    await open();

    expect(screen.getByText(/Paste a SELECT and analyze it/)).toBeInTheDocument();
  });

  it("explains that declining to switch is not a failure", async () => {
    // The single most common thing a reader misreads on this dashboard.
    await open();

    expect(screen.getByText(/That is not a failure/)).toBeInTheDocument();
  });

  it("warns that one run does not settle a comparison", async () => {
    await open();

    expect(screen.getByText(/A single run proves very little/)).toBeInTheDocument();
  });

  it("separates what was predicted from what was measured", async () => {
    await open();

    expect(screen.getByText(/the clock is right/)).toBeInTheDocument();
  });

  it("can be folded again", async () => {
    await open();
    await userEvent.click(screen.getByRole("button", { name: "Hide" }));

    expect(screen.queryByText(/Paste a SELECT and analyze it/)).not.toBeInTheDocument();
  });
});
