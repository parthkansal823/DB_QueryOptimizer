import "@testing-library/jest-dom/vitest";

// Recharts measures its container to lay a chart out, and jsdom reports every
// element as 0x0 -- so a ResponsiveContainer renders nothing and any assertion
// about a chart's contents fails for a reason that has nothing to do with the
// component. Giving elements a non-zero size is what makes the charts
// renderable under test.
Object.defineProperty(HTMLElement.prototype, "clientWidth", { configurable: true, value: 800 });
Object.defineProperty(HTMLElement.prototype, "clientHeight", { configurable: true, value: 400 });

global.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
};

// jsdom implements no media queries at all, and `usePalette` subscribes to
// prefers-color-scheme -- so every chart component throws on mount without
// this. Reported as light; the palette choice is not what these tests assert.
Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: (query) => ({
    matches: false,
    media: query,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  }),
});
