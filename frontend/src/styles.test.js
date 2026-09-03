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
