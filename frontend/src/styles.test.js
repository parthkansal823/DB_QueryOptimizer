import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

/**
 * Every class the components use has a rule somewhere.
 *
 * Twice now a stylesheet rewrite has kept a class name in the JSX and dropped
 * the rule behind it, and both times the result was invisible to every other
 * check: lint passes, the build passes, the component tests pass, and the page
 * renders with a piece of it silently missing.
 *
 *   `.delta-track` / `.delta-fill` / `.delta-zero` -- the diverging "time saved
 *   / lost" bar kept its markup and lost its positioning, so the column
 *   rendered as an empty cell.
 *
 *   `.settings-grid` -- had no rule at all, so the experiment panel's labelled
 *   selects rendered as an unaligned stack.
 *
 * This is the cheapest possible guard against that: not that the CSS is
 * *correct*, which no unit test can tell you, but that it exists.
 */

const SRC = path.resolve(__dirname);

function readAll(dir, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) readAll(full, out);
    else out.push(full);
  }
  return out;
}

const files = readAll(SRC);

const css = files
  .filter((f) => f.endsWith(".css"))
  .map((f) => fs.readFileSync(f, "utf8"))
  .join("\n");

const styled = new Set([...css.matchAll(/\.([a-zA-Z][\w-]*)/g)].map((m) => m[1]));

/**
 * Only literal `className="..."` strings. Template literals and ternaries are
 * skipped deliberately: extracting class names from those needs a parser, and a
 * regex that half-does it produces false positives out of ordinary prose, which
 * is worse than a smaller true check.
 */
function classNamesIn(source) {
  const found = new Set();
  for (const m of source.matchAll(/className="([^"{}]+)"/g)) {
    for (const name of m[1].split(/\s+/)) if (name) found.add(name);
  }
  return found;
}

describe("every class used in a component has a rule", () => {
  const components = files.filter((f) => f.endsWith(".jsx") && !f.endsWith(".test.jsx"));

  it.each(components.map((f) => [path.relative(SRC, f), f]))("%s", (_label, file) => {
    const used = classNamesIn(fs.readFileSync(file, "utf8"));
    const missing = [...used].filter((name) => !styled.has(name));
    expect(missing, `no CSS rule for: ${missing.join(", ")}`).toEqual([]);
  });
});

describe("the saved/lost bar keeps the rules that make it visible", () => {
  /**
   * Named individually because this one has already broken once. The bar is
   * three absolutely-positioned pieces inside a relative track; losing any of
   * those declarations empties the cell without changing the markup.
   */
  const rule = (selector) => {
    const match = css.match(new RegExp(`\\${selector}\\s*\\{([^}]*)\\}`));
    return match ? match[1] : "";
  };

  it("gives the track a positioning context", () => {
    expect(rule(".delta-track")).toMatch(/position:\s*relative/);
  });

  it("gives the zero line real dimensions", () => {
    const zero = rule(".delta-zero");
    expect(zero).toMatch(/position:\s*absolute/);
    expect(zero).toMatch(/width:/);
    expect(zero).toMatch(/height:/);
  });

  it("lets the fill be offset from the centre", () => {
    expect(rule(".delta-fill")).toMatch(/position:\s*absolute/);
  });
});

/**
 * The classes this file *cannot* see.
 *
 * `classNamesIn` reads literal strings, so a class built at render time --
 * `finding-${f.severity}`, `dq-${p.color}` -- is invisible to the check above.
 * That is the same blind spot in the other direction: the rule can be missing
 * and nothing complains, exactly as it did for `.settings-grid`.
 *
 * There is no general fix, because the set of values is decided by the backend.
 * What there is: the backend emits three severities and only three, and this
 * asserts a rule exists for each. If a fourth is ever added, the finding renders
 * unstyled and this test is the only thing that would say so.
 */
describe("classes built from backend values", () => {
  const rules = new Set([...css.matchAll(/\.([a-zA-Z0-9_-]+)/g)].map((m) => m[1]));

  // `diagnostics.warnings_for` and `dbadvisor` between them emit exactly these.
  it.each(["high", "medium", "low"])("styles a %s-severity finding", (severity) => {
    expect(rules.has(`finding-${severity}`)).toBe(true);
  });

});

/**
 * Colours that are not classes at all.
 *
 * `DecisionQuality` assigns each outcome a `color` and paints the segment with
 * `background: colors[p.color]`, where `colors` comes from `usePalette()`. A
 * name the palette does not define is not a styling *mistake* -- it is
 * `undefined`, which renders the segment with no background at all. Invisible,
 * in a chart whose whole job is to show the proportions.
 *
 * Both palettes are checked: they are separate objects, and a key present in
 * one and missing from the other would fail only in one colour scheme.
 */
describe("palette keys the components ask for", () => {
  const source = fs.readFileSync(
    path.join(SRC, "components", "DecisionQuality.jsx"),
    "utf8",
  );
  const asked = [...source.matchAll(/color:\s*"([a-z]+)"/g)].map((m) => m[1]);

  const palette = fs.readFileSync(path.join(SRC, "usePalette.js"), "utf8");
  const blocks = palette.split("const DARK");
  const keysIn = (text) =>
    new Set([...text.matchAll(/^\s{2}([a-zA-Z]+):/gm)].map((m) => m[1]));

  it("asks for at least one colour", () => {
    expect(asked.length).toBeGreaterThan(0);
  });

  it.each(["LIGHT", "DARK"])("%s defines every colour asked for", (which) => {
    const keys = keysIn(which === "LIGHT" ? blocks[0] : blocks[1]);
    const missing = [...new Set(asked)].filter((c) => !keys.has(c));

    expect(missing, `palette ${which} has no entry for: ${missing.join(", ")}`).toEqual([]);
  });
});
