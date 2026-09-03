import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import QueryForm from "./QueryForm";

/**
 * The sample picker, which is how most people meet this system.
 *
 * It had no test until the list grew from 6 entries to 15 and needed grouping.
 * That is the same gap that let a chart render as an empty column for weeks —
 * markup perfect, lint green, no test — so the list is now pinned: the groups
 * exist, every sample carries SQL, and choosing one fills the editor.
 */

describe("the sample picker", () => {
  it("groups the samples rather than listing fifteen flat options", () => {
    render(<QueryForm onAnalyze={vi.fn()} loading={false} error={null} />);
    const select = screen.getByRole("combobox");
    const groups = select.querySelectorAll("optgroup");

    expect(groups.length).toBeGreaterThanOrEqual(3);
    const names = [...groups].map((g) => g.label);
    expect(names).toContain("Correlation traps");
    expect(names).toContain("Complex shapes");
    expect(names).toContain("Controls");
  });

  it("keeps the controls separate from the traps", () => {
    /**
     * The controls exist to show where there is *nothing* to win. Filing them
     * beside the traps would invite reading a flat 4% next to a 95% as the same
     * kind of result.
     */
    render(<QueryForm onAnalyze={vi.fn()} loading={false} error={null} />);
    const select = screen.getByRole("combobox");
    const controls = [...select.querySelectorAll("optgroup")].find(
      (g) => g.label === "Controls",
    );
    const labels = [...controls.querySelectorAll("option")].map((o) => o.value);
    expect(labels.every((l) => !l.includes("headroom"))).toBe(true);
  });

  it("every sample carries SQL, so no option is a dead end", () => {
    render(<QueryForm onAnalyze={vi.fn()} loading={false} error={null} />);
    const options = screen.getByRole("combobox").querySelectorAll("option");
    expect(options.length).toBeGreaterThanOrEqual(15);
    for (const option of options) {
      expect(option.value.trim()).not.toBe("");
    }
  });

  it("choosing a sample puts its SQL in the editor", async () => {
    render(<QueryForm onAnalyze={vi.fn()} loading={false} error={null} />);
    const select = screen.getByRole("combobox");
    const target = [...select.querySelectorAll("option")].find((o) =>
      o.value.includes("DISTINCT"),
    );

    await userEvent.selectOptions(select, target.value);

    expect(screen.getByRole("textbox").value).toContain("SELECT DISTINCT");
  });

  it("submits what is in the editor", async () => {
    const onAnalyze = vi.fn();
    render(<QueryForm onAnalyze={onAnalyze} loading={false} error={null} />);

    const editor = screen.getByRole("textbox");
    await userEvent.clear(editor);
    await userEvent.type(editor, "SELECT 1");
    await userEvent.click(screen.getByRole("button", { name: /analy/i }));

    expect(onAnalyze).toHaveBeenCalledWith("SELECT 1");
  });

  it("does not submit an empty query", async () => {
    const onAnalyze = vi.fn();
    render(<QueryForm onAnalyze={onAnalyze} loading={false} error={null} />);

    await userEvent.clear(screen.getByRole("textbox"));
    await userEvent.click(screen.getByRole("button", { name: /analy/i }));

    expect(onAnalyze).not.toHaveBeenCalled();
  });

  it("shows a structured SQL error when one is supplied", () => {
    /**
     * `SqlErrorPanel` is rendered from here, so the wiring is worth pinning:
     * the backend's 400 detail has to reach it rather than being stringified.
     */
    render(
      <QueryForm
        onAnalyze={vi.fn()}
        loading={false}
        error={{
          message: "column u.nmae does not exist",
          suggestion: "u.name",
          suggestion_source: "postgresql",
        }}
      />,
    );
    expect(screen.getByText("column u.nmae does not exist")).toBeInTheDocument();
    expect(screen.getByText("u.name")).toBeInTheDocument();
  });
});
