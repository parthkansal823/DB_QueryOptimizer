import { useEffect, useState } from "react";

// Recharts needs literal hex (SVG fill attributes don't reliably resolve
// CSS custom properties across browsers), so mirror the light/dark steps
// from index.css here and pick the active set from the OS color scheme.
const LIGHT = {
  native: "#2a78d6",
  chosen: "#eb6834",
  candidate: "#b7d3f6",
  grid: "#e1e0d9",
  axis: "#898781",
  text: "#52514e",
};

const DARK = {
  native: "#3987e5",
  chosen: "#d95926",
  candidate: "#2c2c2a",
  grid: "#2c2c2a",
  axis: "#898781",
  text: "#c3c2b7",
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
