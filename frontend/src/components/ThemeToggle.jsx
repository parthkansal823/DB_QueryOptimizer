import { useEffect, useState } from "react";

// Light/dark.
//
// On a first visit there is no choice to honour, so the page follows the
// operating system — `index.css` defines both themes under
// `prefers-color-scheme` and nothing is stamped on the root element. The moment
// someone clicks, that becomes a choice: `data-theme` is set, which the
// `[data-theme=...]` blocks key off, and from then on the button is a plain
// light↔dark switch.
//
// An earlier version cycled through three states including "back to system".
// Nobody wants a third click that puts the page somewhere they cannot name —
// the setting people actually want is "make it the other one".
//
// One icon, a sun, in both themes. Swapping between a sun and a moon means the
// glyph changes at the same moment the whole page changes, and the two readings
// of it -- "this is the mode you are in" and "this is the mode you will get" --
// are opposites, so half of everyone reads it backwards. A fixed icon says
// "theme" and the tooltip says which way it goes.

// Also spelled out inline in `index.html`, which stamps the saved theme
// before the first paint. That script cannot import this one -- it has to
// run before any module loads -- so the two copies are kept next to a note
// rather than pretending they can be shared.
const KEY = "lqo-theme";

function systemPrefersDark() {
  try {
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  } catch {
    return false;
  }
}

/** The stored choice, or null when there is none and the system decides. */
function readTheme() {
  try {
    const stored = localStorage.getItem(KEY);
    return stored === "light" || stored === "dark" ? stored : null;
  } catch {
    // Private windows and blocked site data both throw here. Following the
    // system is the right fallback, not a crash over a colour preference.
    return null;
  }
}

function applyTheme(theme) {
  const root = document.documentElement;
  if (theme === "light" || theme === "dark") {
    root.setAttribute("data-theme", theme);
  } else {
    root.removeAttribute("data-theme");
  }
}

export default function ThemeToggle() {
  const [theme, setTheme] = useState(readTheme);

  // What is actually on screen right now: the choice if there is one, the
  // system's answer if there is not.
  const [systemDark, setSystemDark] = useState(systemPrefersDark);
  useEffect(() => {
    let media;
    try {
      media = window.matchMedia("(prefers-color-scheme: dark)");
    } catch {
      return undefined;
    }
    const onChange = (e) => setSystemDark(e.matches);
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);

  const effective = theme ?? (systemDark ? "dark" : "light");
  const next = effective === "dark" ? "light" : "dark";

  useEffect(() => {
    applyTheme(theme);
    if (theme === null) return;
    try {
      localStorage.setItem(KEY, theme);
    } catch {
      // Not being able to remember the choice is no reason to refuse it.
    }
  }, [theme]);

  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={() => setTheme(next)}
      title={`Switch to ${next} theme`}
      aria-label={`Switch to ${next} theme`}
    >
      <span aria-hidden="true">☀</span>
    </button>
  );
}
