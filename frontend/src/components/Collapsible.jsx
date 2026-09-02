import { useState } from "react";

// A card that starts closed.
//
// The page has fourteen panels. Some answer the question you came with; others
// are reference tools you reach for occasionally -- the schema listing, the A/B
// runner, the database checker. Leaving those expanded is what turns a dense
// page into a wall you scroll past, and it buries the panels that do answer the
// question underneath them.
//
// The heading still says what is inside, so nothing is hidden -- only folded.
export default function Collapsible({ title, hint, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <section className="card">
      <button
        type="button"
        className="collapsible-head"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        <span className="collapsible-caret">{open ? "\u2212" : "+"}</span>
        <h2>{title}</h2>
        {hint && !open && <span className="collapsible-hint">{hint}</span>}
      </button>
      {open && <div className="collapsible-body">{children}</div>}
    </section>
  );
}
