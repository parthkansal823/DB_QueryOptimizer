import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import SqlErrorPanel from "./SqlErrorPanel";

/**
 * A failed query, explained.
 *
 * Before this panel, a typo produced `500 Internal Server Error` in a red bar —
 * which tells the reader that something went wrong and nothing else. The
 * backend now returns the offending token, an exact position, and often the
 * correction; this asserts the panel actually shows those.
 *
 * The rule under all of it: **a wrong suggestion is worse than none.** Where
 * the backend withheld a corrected statement because it could not make the
 * substitution safely, the panel must not invent a button for it.
 */

const COLUMN_TYPO = {
  message: "column u.nmae does not exist",
  sqlstate: "42703",
  explanation: "That column name is not in any of the tables this query selects from.",
  excerpt: "SELECT u.nmae FROM users u LIMIT 5\n       ^",
  line: 1,
  column: 8,
  offending: "u.nmae",
  suggestion: "u.name",
  suggestion_source: "postgresql",
  suggested_sql: "SELECT u.name FROM users u LIMIT 5",
};

describe("showing what went wrong", () => {
  it("leads with PostgreSQL's message", () => {
    render(<SqlErrorPanel error={COLUMN_TYPO} />);
    expect(screen.getByText("column u.nmae does not exist")).toBeInTheDocument();
  });

  it("explains the error class in plain language", () => {
    render(<SqlErrorPanel error={COLUMN_TYPO} />);
    expect(screen.getByText(/not in any of the tables/)).toBeInTheDocument();
  });

  it("renders the caret excerpt with its whitespace intact", () => {
    /**
     * The caret only lines up under the right character if the newline and the
     * leading spaces survive. Collapsing them would leave a caret pointing at
     * the wrong token, which is worse than showing no excerpt at all.
     */
    render(<SqlErrorPanel error={COLUMN_TYPO} />);
    const excerpt = screen.getByText(/SELECT u\.nmae/);
    expect(excerpt.textContent).toContain("\n");
    expect(excerpt.textContent.split("\n")[1].indexOf("^")).toBe(7);
  });

  it("says where on the line the problem is", () => {
    render(<SqlErrorPanel error={COLUMN_TYPO} />);
    expect(screen.getByText(/line 1, column 8/)).toBeInTheDocument();
  });
});

describe("offering a correction", () => {
  it("names the suggestion and where it came from", () => {
    render(<SqlErrorPanel error={COLUMN_TYPO} />);
    expect(screen.getByText("u.name")).toBeInTheDocument();
    expect(screen.getByText(/PostgreSQL/)).toBeInTheDocument();
  });

  it("distinguishes a catalogue guess from PostgreSQL's own answer", () => {
    /**
     * They carry different authority. PostgreSQL computed its hint against the
     * tables actually in scope; the catalogue fallback is name similarity, and
     * a reader deciding whether to trust it needs to know which they have.
     */
    render(
      <SqlErrorPanel
        error={{ ...COLUMN_TYPO, suggestion: "users", suggestion_source: "catalog" }}
      />,
    );
    expect(screen.getByText(/closest name in this database/)).toBeInTheDocument();
  });

  it("runs the corrected query when the button is used", async () => {
    const onUseSuggestion = vi.fn();
    render(<SqlErrorPanel error={COLUMN_TYPO} onUseSuggestion={onUseSuggestion} />);

    await userEvent.click(screen.getByRole("button", { name: /use this query/i }));

    expect(onUseSuggestion).toHaveBeenCalledWith("SELECT u.name FROM users u LIMIT 5");
  });

  it("offers no button when the backend withheld a corrected statement", () => {
    /**
     * The token appears more than once, so the backend could not know which
     * occurrence was wrong. Showing a button here would apply a substitution
     * nobody verified.
     */
    render(
      <SqlErrorPanel
        error={{ ...COLUMN_TYPO, suggested_sql: undefined }}
        onUseSuggestion={vi.fn()}
      />,
    );
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
    expect(screen.getByText(/has to be applied by hand/)).toBeInTheDocument();
  });

  it("shows no suggestion machinery at all when there is nothing to suggest", () => {
    render(
      <SqlErrorPanel
        error={{
          message: 'syntax error at or near "orders"',
          explanation: "PostgreSQL could not parse the statement at that point.",
        }}
      />,
    );
    expect(screen.queryByText(/Did you mean/)).not.toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });
});

describe("degrading gracefully", () => {
  it("renders nothing when there is no error", () => {
    const { container } = render(<SqlErrorPanel error={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("still shows a plain string error", () => {
    /**
     * A network failure never reaches the backend, so it has no structured
     * detail. It still has to appear somewhere.
     */
    render(<SqlErrorPanel error="Failed to fetch" />);
    expect(screen.getByText("Failed to fetch")).toBeInTheDocument();
  });
});
