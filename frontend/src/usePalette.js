import { useEffect, useState } from "react";

// Recharts needs literal hex (SVG fill attributes don't reliably resolve
// CSS custom properties across browsers), so mirror the light/dark steps
// from index.css here and pick the active set from the OS color scheme.
// Categorical slots 1-3 carry *identity* (which plan), status colours carry
// *polarity* (was the decision right). Keeping those two jobs on separate
// colours is what stops "orange" meaning both "the served plan" and
// "something went wrong" on the same page.
const LIGHT = {
  native: "#2a78d6", // slot 1
  chosen: "#eb6834", // slot 2
  oracle: "#1baf7a", // slot 3
  candidate: "#b7d3f6",
  neutral: "#898781",
  good: "#0ca30c",
  warning: "#fab219",
  critical: "#d03b3b",
  grid: "#e1e0d9",
  axis: "#898781",
  text: "#52514e",
  surface: "#fcfcfb",
};

const DARK = {
  native: "#3987e5",
  chosen: "#d95926",
  oracle: "#199e70",
  candidate: "#2c2c2a",
  neutral: "#898781",
  good: "#0ca30c",
  warning: "#fab219",
  critical: "#d03b3b",
  grid: "#2c2c2a",
  axis: "#898781",
  text: "#c3c2b7",
  surface: "#1a1a19",
};

export function usePalette() {
  const [dark, setDark] = useState(
    () => window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false
  );

  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = (e) => setDark(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  return dark ? DARK : LIGHT;
}
