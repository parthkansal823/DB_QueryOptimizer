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

## The production path

Everything above uses `/query/analyze`, which runs every candidate so they can
be compared. That is a measuring tool — it costs N executions to answer one
question, so it would be slower than having no optimizer at all.

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

"Fastest seen" is knowable only because `/query/analyze` runs every candidate.
In production you never learn what the plans you skipped would have cost.

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
