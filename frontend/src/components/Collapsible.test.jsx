import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import Collapsible from "./Collapsible";

// Some panels answer the question you arrived with; others are reference tools
// you reach for occasionally. Leaving the second kind expanded is what turned a
// dense page into a wall and buried the panels that do answer it.

describe("a folded panel", () => {
  it("starts closed but still says what is inside", () => {
    render(
      <Collapsible title="Database health" hint="correlated columns">
        <p>the contents</p>
      </Collapsible>,
    );

    expect(screen.getByText("Database health")).toBeInTheDocument();
    expect(screen.getByText("correlated columns")).toBeInTheDocument();
    expect(screen.queryByText("the contents")).not.toBeInTheDocument();
  });

  it("opens when the heading is clicked", async () => {
    render(
      <Collapsible title="Database health">
        <p>the contents</p>
      </Collapsible>,
    );

    await userEvent.click(screen.getByRole("button"));

    expect(screen.getByText("the contents")).toBeInTheDocument();
  });

  it("drops the hint once open, since the contents are the answer", async () => {
    render(
      <Collapsible title="Schema" hint="tables and join edges">
        <p>body</p>
      </Collapsible>,
    );

    await userEvent.click(screen.getByRole("button"));

    expect(screen.queryByText("tables and join edges")).not.toBeInTheDocument();
  });

  it("tells assistive tech whether it is open", async () => {
    render(
      <Collapsible title="Schema">
        <p>body</p>
      </Collapsible>,
    );

    const head = screen.getByRole("button");
    expect(head).toHaveAttribute("aria-expanded", "false");

    await userEvent.click(head);
    expect(head).toHaveAttribute("aria-expanded", "true");
  });

  it("can start open when the panel is worth reading every visit", () => {
    render(
      <Collapsible title="Plans" defaultOpen>
        <p>body</p>
      </Collapsible>,
    );

    expect(screen.getByText("body")).toBeInTheDocument();
  });
});
