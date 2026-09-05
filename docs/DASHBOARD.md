# Reading the dashboard

A guide to every panel at `http://localhost:5173`: what it shows, how to read
it, and what a healthy result looks like.

For the exact formula behind each number, see `METRICS.md`. For the research
context and results, see `WRITEUP.md`.

---

## How the page is organised

The first three panels are about **the query you just pasted**. They only
appear after you analyse something. The rest are about **the system as a
whole**, and they are there from the moment you open the page.

| Panel | Scope | Answers |
|---|---|---|
| Faster version of your query | This query | Can I copy something useful right now? |
| Plans considered | This query | What ran, what else was tried, and why |
| Model health | System | What is deployed, and is anything blocked? |
| The production path | This query | What does this cost to run for real? |
| Database fixes | System + query | What should I change in the database? |
| How much time this is saving | System | Is this helping, overall? |
| Were the decisions right? | System | *Why* is it helping, or not? |
| Time saved, run by run | System | Is it still improving? |
| Do PostgreSQL's estimates match reality? | Database | Is there anything here to learn? |
| Day by day | System | Longer-term trend (only shows after day two) |
| What we found in your database | Database | What does it know about my schema? |
| Settings | System | What is it configured to do, and what is actually in force? |

### A control can show one value while another is applied

Two of the settings are not what the optimizer uses. `confidence_z` and
`min_relative_gain` are declared with defaults of 1.0 and 0.05, and the
optimizer overrides both with the measured gate in `models/gate.json` unless
somebody explicitly chooses them &mdash; deferring to a measurement over a default
is the point of calibrating one.

The panel used to show the default and say nothing, so it read **z = 1.0** while
the gate in force was **1.5** (`WRITEUP.md` 2.54). Any control whose value has
been superseded now carries a line under it:

> **In force: 0.02** &mdash; the measured gate in `models/gate.json` is being applied
> instead of this value. Changing this setting overrides the calibration.

Editing the control is still how you override the calibration; the note exists
so you know that is what you are doing.

---

## Faster version of your query

The practical output. If any alternative plan beat PostgreSQL, this hands you
the SQL to copy.

- **PostgreSQL** — how long PostgreSQL's own plan took.
- **Optimized** — how long the fastest alternative took.
- **Speedup** — the ratio between them, with the percentage underneath.

Below that is the query with a `/*+ ... */` hint attached. The hint changes
only the plan, never the results, and PostgreSQL ignores it as a comment if
the `pg_hint_plan` extension is not installed. It is safe to commit.

Two notes appear when they apply:

- **"Why PostgreSQL got this wrong"** — the row count it expected against
  what it actually got. A large gap means its estimate was off, which is
  usually why it picked a poor plan.
- **A cost inversion** — PostgreSQL priced the *faster* plan higher. This is
  the clearest evidence you will see that its cost model was wrong here.

**If the panel is empty**, PostgreSQL's own plan was the fastest thing tried.
That is a real and common answer, not a failure.

> This panel reports what the measurements found *afterwards*. The optimizer
> may not have picked this plan when serving, because it did not know the
> answer in advance. When that happens, the panel says so.

---

## Plans considered

Side-by-side detail for PostgreSQL's plan and the one the optimizer picked:
measured time, estimated cost, top node type, join order, join types, and the
hint that produced it.

Underneath is the decision itself — the policy in use, the speedup the model
predicted, and the bar it had to clear before it was allowed to switch.

If the optimizer stayed with PostgreSQL, a short line explains why:

| Reason | What happened |
|---|---|
| Nothing else was clearly faster | No alternative beat the confidence bar |
| The best alternative is priced far higher | The safety check blocked it as too risky |
| The learned path has been slower before | The regression guard blocked this query |
| No alternative join orders | Single table, or a shape not handled yet |

### Was the pick actually faster?

Directly under the two plan panels, one sentence saying what happened:

> **This pick was slower.** It ran in 12.50 ms against PostgreSQL's 8.64 ms —
> 45% worse.

or

> **Outcome:** the chosen plan and PostgreSQL's ran within 5% of each other,
> which is inside this database's measurement noise. Neither was meaningfully
> faster.

This is a *measured* statement, not a predicted one. The decision note beside it
says what the model expected; this says what the clock said. `/query/analyze`
executes the model's pick alongside a few others, so by the time the panel
renders the answer is known — and it used to present a pick that ran slower as
"LEARNED PICK / CHOSEN" beside a larger number and leave you to compare the two
yourself.

The 5% band matters. Two runs of the *same* plan on this database differ by more
than that, so calling a 4% difference a regression would be as wrong as calling
it a win.

### Every plan we tried

A bar per plan, so you can see the whole spread rather than just the winner.

- **Blue** — PostgreSQL's plan.
- **Orange** — the plan that actually ran.
- **Grey** — measured but not used.
- **Dashed outline** — the model wanted this one, but a safety check blocked
  it. Compare the bar heights to judge whether that was the right call.

---

## Model health

- **Serving** — `learned` if a trained model is deployed, `heuristic` if it is
  falling back to the cost rule (which is what happens before you train).
- **Policy** — how it picks among candidates. See `METRICS.md`.
- **Model version** — the timestamped version currently deployed.
- **New data to learn from** — executions logged since that version trained.
  Only counts rows a retrain would actually use.

**Regression guard** lists queries that are being served PostgreSQL's plan on
purpose, because the learned path has measurably been slower on them. An empty
list is the healthy state. A query needs at least three observations before it
can be blocked, so a query that just regressed will not appear immediately.

**Retrain & gate** trains a challenger and deploys it *only* if it clearly
beats what is running. **Roll back** restores the previous version.

---

### Findings

At the top of Model health, a list of things wrong with the model or its safety
gate, worst first. Each says what was measured and what it means.

Every one of these had actually shipped at some point and was invisible from
this dashboard:

| finding | what it means |
|---|---|
| `gate_ignores_uncertainty` | `confidence_z` is 0, so the model's uncertainty is being multiplied by zero. The "pessimistically" figure in the decision note will equal the prediction exactly — the safety check is decorative. |
| `gate_calibrated_on_too_few_queries` | The threshold was chosen on a handful of queries, so its reported regression rate moves in steps of 1/n and is fitted to whichever queries happened to be held out. |
| `uncertainty_understated` | The eight models disagree by less than the model's actual error, by the factor shown. `docs/CONFIDENCE.md` §3 explains why no arrangement of the ensemble fixes that. |
| `error_comparable_to_headroom` | Predictions are typically off by about as much as the speedups worth chasing. More distinct queries is the lever, not a different model. |
| `gate_promise_not_met` | The calibration promised a regression rate and the system is exceeding it. Either the workload has moved or the gate was chosen on too few queries. Scoped to runs served *since this gate was applied* — before that it inherited the previous gate's verdict and reported a freshly-calibrated gate as already broken (2.49). |
| `optimizer_never_deviates` | Nothing has deviated recently, so the optimizer is currently a no-op. |
| `gate_calibrated_for_another_policy` | The gate's thresholds were swept under one selection policy and the server selects with another. Not a rounding difference: on 15 held-out queries `greedy` landed at 104.1% of native and `pairwise_rank` at 95.1% (`WRITEUP.md` 2.43). Re-run `app.autotune --apply`. |
| `gate_calibrated_for_another_action_space` | Same idea, for the hint families on offer. The action space decides what there is to select from, so a threshold measured against a different one is measuring a different system. |
| `gate_does_not_record_its_policy` | The gate file predates `calibrated_with_policy`, so the two cannot be compared at all. "Cannot tell" rather than "they disagree". |
| `no_usable_coverage_bound` | A conformal bound at this risk level would need so many sigma that `GATE_MODE=conformal` would decline almost every query. Correct, and not currently actionable. |

"No problems found" is a result, not an empty panel.

The same report is `GET /model/diagnostics`, or `python -m app.diagnostics` for
the full version including the drift check.

## The production path

Everything above uses `/query/analyze`, which plans every candidate and executes
a few of them so they can be compared. It is a measuring tool: even executing
four costs four times what serving one does.

It used to execute *all* of them, which on a four-table query meant 19
executions at 48 ms each — 915 ms of a 1.2 s request. Planning a candidate costs
**0.89 ms**, and the model predicts latency from the plan tree rather than
needing it measured, so those executions bought a slower dashboard and nothing
else. `ANALYZE_EXECUTE_LIMIT` (default 4) now bounds them: the plan the model
chose, plus the cheapest by the planner's own estimate. **1.19 s to 0.19 s.**

The consequence is worth knowing when reading the panels above: the comparison
is against four executed alternatives, not every one. `n_candidates_planned` and
`n_candidates_executed` in the response say which numbers are measured and which
are the planner's estimate.

This button uses `/query/optimize`, the real path: it plans every candidate
with `EXPLAIN` (nothing runs) and executes only the one it picks.

- **Time spent choosing** — the planning overhead the optimizer added.
- **Time running the query** — the actual execution.
- **Choosing, as a share** — overhead as a percentage of execution. This is
  the number that decides whether the whole idea pays for itself. On this
  workload it sits near 3%.

---

## Database fixes

A hint fixes one query. These fix *why* the hint was needed, so every query
touching those columns benefits.

- **Correlated columns** — PostgreSQL assumed two filters were unrelated and
  multiplied their odds. `CREATE STATISTICS` teaches it the real relationship.
- **Missing index** — a sequential scan read a large table and threw away
  almost all of it.
- **Unindexed foreign key** — PostgreSQL indexes the referenced key
  automatically, never the referencing column, so joins through it scan.

Each card carries DDL you can copy. Nothing runs automatically: `CREATE INDEX`
takes locks and disk space.

---

## Database health

Findings about the *database* rather than about one query's plan. Every other
panel here is about plan choice; these are things fixed once that then help
every query, including ones this system has never seen.

Four groups, each with the exact statement that would apply it:

- **Correlated columns the planner does not know about.** PostgreSQL estimates
  `WHERE a = x AND b = y` by multiplying the two selectivities, which is only
  right when the columns are unrelated. Where they are related the estimate is
  badly wrong and every plan built from it is built for the wrong number of
  rows. Shown as a measured before-and-after: *"estimate off by 4.9x → 1.0x"*.
- **Server settings.** PostgreSQL's defaults are chosen so it will start
  anywhere, not to suit anything. `shared_buffers` is 128MB whether the machine
  has 1GB or 128GB.
- **Indexes not earning their keep.** Never scanned, duplicated, or a leading
  prefix of a wider index. Each is paid for on every write and appears in no
  slow-query report.
- **Table health.** Statistics describing a table that has since changed, and
  dead row versions that every sequential scan has to read past.
- **An index of the wrong *kind*.** B-tree is not always the answer. A column
  the table is already physically ordered by -- an append-only `created_at` --
  can be served by a BRIN index that summarises min/max per page range, a few
  hundred kilobytes where the B-tree is tens of megabytes. It fires only on
  `pg_stats.correlation` above 0.9, never on a key column (BRIN cannot enforce
  uniqueness or answer a point lookup), and never on a column that has not been
  analyzed -- an unmeasured column gets silence rather than a guess.
- **A statistics target that has run out of room.** PostgreSQL stores at most
  `statistics_target` most-common values per column. When a column comes back
  with *exactly* that many, the list is truncated: the planner wanted to track
  more and could not, and every value it dropped is now estimated from one
  uniform average. On this database `orders.user_id` tracks 100 of roughly
  23,558 distinct values. Reported on saturation, which is an observable fact,
  rather than on "estimates look bad", which is a guess about a cause.

**Nothing here is applied.** Running DDL against your database is your decision;
the panel shows the statement and stops.

The correlated-column check is behind a checkbox because it is the expensive
one — each candidate is *measured* by creating the statistics object, running
`ANALYZE`, re-planning and rolling back. `docs/DATABASE_OPTIMIZATION.md` has
the detail.

## Settle a comparison

Every other number on this page comes from a single run. This panel runs a
proper A/B: two arms interleaved so neither gets the colder cache, a pair
discarded if either half fails, a bootstrap interval around the difference, and
a sign test.

The important output is the word **unresolved**. Native totals on this database
swing about 13% between identical runs, so a mean difference smaller than that
is not a result however good it looks. When the interval spans zero the panel
says so rather than reporting the mean.

It is slow — each paired run is two full benchmark sweeps — and runs in the
background, so the panel can be left and come back to.

This existed as `python -m app.experiment` long before it had a panel, which is
the problem it solves: the rigorous comparison was the inconvenient one, and
`docs/WRITEUP.md` §2.4 records a headline result that rested on three runs
because of it.

## How much time this is saving

The headline panel, and the one to read carefully.

- **Time with PostgreSQL** / **Time with the optimizer** — total time across
  every comparable run.
- **Improvement** — the difference, as a percentage.
- **Changed the plan** — how often the optimizer deviated at all.

Every figure is a **matched pair**: both plans measured for the same query,
seconds apart, in the same run. Runs where the optimizer stayed with
PostgreSQL count as 0%, not as missing data.

> This matters more than it sounds. Comparing the average of the plans it
> chose against the average of all PostgreSQL plans is not a comparison — the
> two averages cover different queries. The optimizer deviates on queries it
> understands, which skew cheap, so that version of the number reports a huge
> win while every expensive query is still being served PostgreSQL's plan.
> `METRICS.md` works through the arithmetic.

### The per-query table

Sorted **worst first**, so a query that got slower leads instead of hiding
inside an average.

| Column | Meaning |
|---|---|
| Query | The SQL, collapsed to one line. Hover for the full text |
| Changed | Runs where the plan was changed, out of total runs |
| PostgreSQL | Average time for PostgreSQL's plan |
| Served | Average time for the plan that actually ran |
| Fastest seen | The quickest plan measured for this query — the ceiling |
| Time saved / lost | Green right of centre is saved, red left is lost |
| Improvement | Percentage, negative when the query got slower |

"Fastest seen" is knowable only because the candidates were executed, which is
why it is a dashboard figure rather than a production one: in production you
never learn what the plans you skipped would have cost.

It is the fastest of the candidates this endpoint *executed*, not of all of
them — four by default. `app.benchmark` executes every candidate, so its oracle
is the real ceiling; this one is a floor under it. Raise
`ANALYZE_EXECUTE_LIMIT` to close the gap, at the cost of the request time the
limit exists to save.

---

## Were the decisions right?

An improvement percentage cannot tell you *why* a system is winning or losing.
This panel splits every decision into what it actually achieved.

**When it changed the plan:**

| Outcome | Meaning |
|---|---|
| Paid off | Deviated and the query got faster |
| No real change | Deviated, but the difference was too small to matter |
| Backfired | Deviated and the query got slower |

**When it kept PostgreSQL's plan:**

| Outcome | Meaning |
|---|---|
| Right to hold | Nothing faster was available |
| Missed a win | A faster plan was available and it did not take it |

Those last two are the reason this panel exists. Both score exactly 0%
improvement, both are invisible in any average, and only one of them is a good
decision.

Two figures sit above:

- **Time added** — total cost of the deviations that backfired.
- **Time left on the table** — total wins it passed up.

"Faster" here means at least 5% *and* at least 2 ms — the same bar the
optimizer applies to itself before it will switch, so it is judged by its own
rules rather than a stricter standard.

### What good looks like

A healthy system has a large "Paid off" segment, few "Backfired", and shrinking
"Missed a win" as it learns. **Backfired is the one to watch**: a diffuse
average win does not compensate for one user-facing query getting three times
slower.

---

## Time saved, run by run

Three running totals of time spent, in decision order. All three only climb —
what matters is the distance between them.

- **PostgreSQL down to served** — the time saved. Shaded green.
- **Served down to best possible** — regret: the time still on the table.

The dashed line is the best plan available on each run. It is a bound, not a
competitor, which is why it is drawn as a reference line rather than a series.

| Shape | Reading |
|---|---|
| Top gap widening | Still finding wins |
| Top gap flat | Stopped finding wins |
| Lines crossing | Losing ground — served is now slower than PostgreSQL |
| Bottom gap growing in a straight line | Not learning; same mistake repeatedly |

**Regret vs PostgreSQL** below 1.00 means the optimizer has cost less than
always trusting PostgreSQL would have. That single number is the honest
summary of whether any of this was worth doing.

---

## Do PostgreSQL's estimates match reality?

This is the premise of the whole project, plotted.

PostgreSQL picks plans by minimising estimated cost. If that estimate ordered
plans the way real time does, a learned optimizer would have nothing to learn.

Each dot is one measured plan: estimated cost across, measured time up. Both
axes are logarithmic, because each spans a thousandfold range.

- **Rank correlation** — how well cost order predicts speed order. 1.00 would
  be perfect. Anything below it is the space this project works in.
- **Vertical spread** — plans PostgreSQL prices identically that run ten times
  apart. That spread is the opening.

Plans switched off by a hint are excluded. PostgreSQL gives those a placeholder
cost of 10¹⁰ rather than a real estimate, so they say nothing about estimate
quality, and including them would both wreck the scale and flatter the
correlation.

---

## Day by day

Only appears once there is more than one day of history — a single point is
not a trend.

Both lines average the *same* runs, so the gap is the optimizer's doing rather
than a difference in which queries happened to run. Bigger dots mean more runs
behind that day's average. Movement in the PostgreSQL line reflects the mix of
queries, not PostgreSQL changing.

---

## What we found in your database

What the optimizer discovered on its own, so "works on any database" is
something you can check rather than take on trust.

- **Tables** and **Join edges** — the join graph it built.
- **Rows (estimated)** — from `pg_class.reltuples`, the statistics the planner
  itself uses. An estimate maintained by autovacuum, not a live `COUNT(*)`.
- **Edges from** — `declared FKs` if the database declares foreign keys,
  `naming` if it had to infer them from column names (`x_id → x.id`).

Inferred edges are guesses. A wrong guess costs speed, never correctness. This
is how the system runs on the JOB/IMDB benchmark, whose schema declares no
foreign keys at all.

---

## Building on your own hint

If the query you paste already carries a `/*+ ... */` comment — which is what
this panel's own output is — a line appears above the SQL naming what was
carried forward.

Two things it is telling you. **The hints were merged, not stacked.**
pg_hint_plan reads only the first hint comment in a statement and silently
ignores any that follow, so adding a second one would have thrown away the one
you pasted in; they go into a single comment instead, and a new join order
replaces an old one rather than contradicting it.

**The baseline is still the unhinted query.** The speedup is measured from where
PostgreSQL started, not from your previous pass. Otherwise a query you had
already optimised would keep reporting "no improvement available" — it would be
competing against itself.

## Was this comparison fair?

Under "Faster version of your query", beside the speedup, is a line about cache
state. It exists because the most expensive measurement mistake this project
made was reporting an 80.5% improvement that was really 26.0%: the two sides ran
at different cache states, and the difference between reading from memory and
reading from disk was attributed to the plan.

**fair** — both plans ran at a comparable cache hit ratio, so the difference in
time is about the plans.

**unfair** — one side read from memory and the other from storage. The gap alone
could explain the result, so it is not evidence about the plans. Run the pair
through *Settle a comparison* instead.

**unknown** — buffer counters were not collected on one side. Not the same as
"fair": silence there is how the original mistake went unnoticed.

Beside it, **work** reports how many 8 kB pages each plan touched. That number
has a spread of **0.0%** across repeated runs where the timing varies by about
5%, which makes it the better tie-breaker when two runtimes are inside the noise
band. It is not a speed verdict: measured here, a parallel plan touched 2.1×
fewer pages and ran 25% slower, because worker startup dominates on a small
query.

## Maintenance

Vacuum and analyze decide what the planner is *allowed* to choose, not only how
much disk is wasted. Four findings, none of which the table-health panel can see:

- **Never analyzed** — a partitioned parent, an inheritance parent or a foreign
  table. Autovacuum never analyzes any of them, so changes to child tables leave
  the parent with no statistics at all, on the largest queries in the system.
- **Approaching a forced vacuum** — how close the oldest unfrozen rows are to
  triggering an anti-wraparound autovacuum, which scans the whole table and runs
  whether or not autovacuum is enabled. Left long enough, PostgreSQL stops
  accepting writes.
- **Updates that touch every index** — the fraction of updates that were *not*
  heap-only. This is what an unused index really costs: not disk, which is a
  fixed cost, but a write into every index on every update.
- **Index-only scan readiness** — what share of a table's pages are marked
  all-visible. Below that, an index-only scan checks the heap anyway, so both an
  `IndexOnlyScan` plan and a covering index buy less than they look like they
  should. The fix is a `VACUUM`, and it is also why one of those candidates lost.

## Across the whole workload

Read from `pg_stat_statements`, so it covers every query the server has run, not
only the one in the box.

**One caveat is repeated on the panel because it changes what the numbers mean.**
pg_hint_plan hints are comments, and pg_stat_statements normalises comments away
— so every candidate plan for a query folds into a single row. "This query wrote
118,000 temp blocks" therefore means *some plan for it did*, not the native one.
Rows generated by this optimizer's own candidate sweep are labelled, because on
a dashboard-driven database the sweep is most of the workload.

## Observability

Settings that decide whether a slow query leaves any evidence behind — all of
which ship off, or effectively off. Each finding says what its absence costs
*and* what turning it on costs, because "enable more logging" is advice that has
taken production systems down.

The one worth reading first is `log_autovacuum_min_duration`, which defaults to
ten minutes and therefore hides nearly every autovacuum — including an
aggressive freeze scan landing in the middle of a benchmark and appearing as
variance nobody can account for.

## How the page is put together

The dashboard was redesigned around one observation: **everything was a card,
every card had the same weight, and so nothing had any.** A panel reading
"nothing to compare yet" occupied the same bordered rectangle, with the same
padding and the same heading size, as one showing a 36× measured speedup. The
page could not tell you where to look because it did not know.

Three rules now run through it, and each replaces something specific.

**Weight follows importance.** Panels come in three weights. The hero — the one
result worth acting on — carries the accent border. Ordinary panels are a
hairline. An empty result is a single line of muted text, not a bordered box the
height of a real one. Section headings went from 12px uppercase grey, which is
the treatment for a table column header, to a real heading with space above it.

**Colour means something or is absent.** One accent, used for the primary action
and for a measured win. Severity is a 2px left rule rather than a tinted box:
ten tinted boxes stacked up read as an alarm, ten rules read as a list you can
work through. Everything else is ink on paper.

**Density is deliberate.** Tables and stat rows tighten; structure gets air.
Both used identical padding before, which is how the page managed to look
uniform and empty at once. The number that is the answer is now set at 32px
against 13px for the sentence qualifying it — they were both 14px.

Both themes are defined in full in `index.css`. No colour is declared only
inside a media query, which is how a page ends up unreadable in the theme its
author was not using.

### Four bugs the redesign surfaced

Rewriting the stylesheet against the real markup found problems that had been
invisible while everything looked uniformly grey:

- **The query editor had no class at all.** `.query-editor` was styled and never
  applied; the textarea had been picking up only the generic input rule, which
  is why it rendered two lines tall.
- **`query-editor` meant two different things.** `QueryManager` used it as the
  name of a *container* holding four inputs. Once the textarea claimed it, the
  container inherited a fixed 9rem height and the saved-query form collapsed on
  top of itself.
- **`DemoPage` wrapped its one-line intro in `app-header`**, giving a sentence
  the site header's border, padding and bottom margin — a stray rule across the
  page and a large gap under it.
- **A table inside a grid cell was unbounded.** The per-run query list ran to a
  hundred rows and made its card twice the height of the two beside it, which
  broke the alignment the grid exists to provide. It scrolls now, with a sticky
  header.

## Empty panels are answers

If a panel is empty, that is usually information rather than breakage:

| You see | It means |
|---|---|
| No faster version | PostgreSQL's plan was already the best tried |
| No alternative join orders | Single-table query, or a shape not handled |
| Nothing to compare yet | No run has measured both plans for one query |
| No queries blocked | The learned path has not measurably regressed |
| No database fixes | Estimates were close enough, no scan wasting work |

To get history quickly, run the benchmark over the built-in workload:

```bash
docker compose exec backend python -m app.benchmark
```
