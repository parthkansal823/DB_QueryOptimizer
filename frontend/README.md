# Dashboard

The React front end for the learned query optimizer. Paste a query, see how
PostgreSQL's plan compares to the one the optimizer picks, and track whether
the system is helping over time.

**To read the panels, see [`../docs/DASHBOARD.md`](../docs/DASHBOARD.md).**
**For how each number is worked out, see [`../docs/METRICS.md`](../docs/METRICS.md).**

## Running it

Normally you get this for free with the whole stack:

```bash
docker compose up --build     # dashboard on :5173, API on :8000
```

To run it on its own against an API that is already up:

```bash
npm install
npm run dev
```

| Command | Does |
|---|---|
| `npm run dev` | Dev server with hot reload |
| `npm run build` | Production build into `dist/` |
| `npm run preview` | Serve the production build |
| `npm run lint` | Oxlint |

The API base URL comes from `VITE_API_BASE_URL` and defaults to
`http://localhost:8000`. The backend only allows requests from the origin in
its `FRONTEND_ORIGIN` variable, so change both together if you move ports.

> **Editing files while running in Docker?** Windows bind mounts do not
> deliver file-change events, so Vite keeps serving the version it cached at
> start-up. Run `docker compose restart frontend` after editing, or run the
> dev server on the host instead.

## Layout

```
src/
├── App.jsx              page layout, data loading, panel order
├── api.js               every backend call
├── usePalette.js        chart colours, light and dark
├── index.css            design tokens
├── App.css              component styles
└── components/
    ├── QueryForm.jsx        query box and sample queries
    ├── OptimizedQuery.jsx   the copyable faster query
    ├── PlanComparison.jsx   plan detail, and why it was chosen
    ├── LatencyChart.jsx     every candidate, measured
    ├── ServedVsNative.jsx   paired totals and the per-query table
    ├── DecisionQuality.jsx  win / wash / backfire / hold / missed
    ├── CumulativeChart.jsx  time saved and regret over the run sequence
    ├── CostModelChart.jsx   estimated cost against measured time
    ├── TrendChart.jsx       day-by-day averages
    ├── ModelHealth.jsx      deployed version, guard, retrain controls
    ├── ProductionRun.jsx    the real serving path
    ├── Recommendations.jsx  index and statistics suggestions
    └── SchemaPanel.jsx      the discovered schema
```

## Chart conventions

Worth keeping to if you add a panel.

**Two colour jobs, kept separate.** Categorical slots carry *identity* — which
plan this is. Status colours carry *polarity* — whether a decision was right.
Mixing them makes orange mean both "the served plan" and "something went
wrong" on the same page.

| Role | Token | Used for |
|---|---|---|
| Identity 1 | `--series-native` | PostgreSQL's own plan |
| Identity 2 | `--series-chosen` | The plan that ran |
| Identity 3 | `--series-oracle` | Best possible |
| Recessive | `--series-candidate` | Measured but not used |
| Neutral | `--series-neutral` | Reference lines, "no change" |
| Good / warning / critical | `--status-*` | Decision outcomes |

Other rules the existing charts follow:

- **One y-axis per chart.** Two scales in one frame invites reading a crossing
  as meaningful when it is an artefact. Where a second measure matters, encode
  it another way — `TrendChart` uses dot size for run count.
- **Status colour always ships with an icon and a label.** Some status steps
  sit below 3:1 contrast on the light surface by design, so colour is never
  the only carrier of meaning.
- **Animation off.** These charts are read, not watched, and a grow-in delays
  the comparison the panel exists to support.
- **A legend whenever there are two or more series.**
- Both themes are defined explicitly in `index.css`. Dark is a selected set of
  steps, not an automatic inversion.

The palette follows a validated categorical scale; if you change it, re-run a
contrast and colour-vision check rather than eyeballing it.

## Adding a panel

1. Add the endpoint to `api.js`.
2. Load it in `App.jsx` and re-fetch it in `handleAnalyze` if analysing a
   query changes it.
3. Build the component against the tokens above.
4. Add a `<section className="card">` with a plain-language heading.

Two things to hold to, because the panels here have been wrong before in
exactly these ways:

- **Compare like with like.** Any "before and after" figure must come from the
  same queries and the same runs. `METRICS.md` §2 shows how two innocent
  averages reported a 97% win that was not real.
- **Say what is excluded.** Panels that filter rows show the counts rather
  than quietly dropping them.
