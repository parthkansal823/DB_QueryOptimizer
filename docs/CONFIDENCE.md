# How the optimizer decides whether to trust itself

This is the part of the system that decides *not* to act. It matters more than
the part that chooses a plan, because a query optimizer that is usually right
and occasionally three times slower is worse than one that does nothing.

This document explains, in order: what the decision looks like, what went wrong
with it, and what the measurements said when each fix was tried. Two of those
fixes were wrong and were removed. That is written down here too, because a
document that only lists what worked is a sales pitch.

---

## 1. What a decision actually is

For every query the optimizer produces two numbers.

**The prediction.** The model does not predict "this plan will take 213 ms". It
predicts a *ratio*: how long this plan will take compared to PostgreSQL's own
plan. `0.70` means "30% faster". `1.20` means "20% slower". This is deliberate —
we never need to know the absolute time, only whether the alternative is better.

**The uncertainty.** The model is not one model but eight, each trained on a
slightly different sample of the data. Where they agree, the prediction rests on
evidence. Where they disagree, it does not. The spread between them is the
uncertainty, usually written as *sigma*.

The gate combines them:

```
prediction = 0.70x        "I think this is 30% faster"
sigma      = 0.05         "and the eight models roughly agree"
                          -> worth acting on

prediction = 0.70x        "I think this is 30% faster"
sigma      = 0.60         "but the eight models disagree wildly"
                          -> not worth acting on
```

That is the whole idea. Everything below is about the ways it failed.

---

## 2. The first failure: the gate demanded the worst case also win

The rule used to be a single condition — the *pessimistic* estimate had to beat
PostgreSQL:

```
prediction + z x sigma  <  1 - required_margin
```

Read out loud: *"even if I am as wrong as I usually am, this still wins."*

That sounds prudent and it throws away the best opportunities the system has.
A real example from the dashboard:

| | |
|---|---|
| model predicted | **0.20x** — five times faster |
| what actually happened | 0.16x — it was right |
| but the eight models disagreed on this query, so the pessimistic figure was | 1.13x |
| the rule required | below 0.80x |
| **result** | **refused** |

It declined a five-fold speedup because the worst case was a 13% loss. Nobody
would take that trade knowingly.

The mistake was asking one question of two different things. Size of opportunity
and amount of risk are not the same quantity and do not belong in the same
inequality. So they are now asked separately:

| question | asked of | condition |
|---|---|---|
| Is there an expected win? | the prediction | `prediction < 1 - required_margin` |
| Is the downside bounded? | the uncertainty | `pessimistic < ceiling` (default 1.25x) |

The second is a **ceiling on how bad it can get**, not a demand that the worst
case also be good. This is also what the Bao paper actually claims a learned
optimizer must guarantee — never *much* worse than the optimizer it replaces.
"Never worse at all" is a different and impossible promise.

Both dashboard cases now come out right. The 5.9x win is served: `0.20` clears
the gain bar and `1.13` is inside the ceiling. A candidate predicted at `0.89x`
whose pessimistic case is `1.32x` is still held, because that one really is a
coin flip on a small edge.

Measured over the whole workload, same model, same harness:

| gate | queries vetoed | share of the available speedup captured |
|---|---|---|
| worst-case-must-win | 20 of 24 | 13.5% |
| no gate at all | 2 of 24 | 26.0% |
| **expected win + bounded downside** | 10 of 24 | **52.7%** |

The gate now beats having no gate, which is the point: it lets the good bets
through *and* blocks the bad ones.

---

## 3. The second failure: sigma was never going to work

Even after that, something was wrong. The uncertainty was strangely small.
Measured on the model's own out-of-bag predictions — that is, predictions made
for rows the individual models had not seen:

```
how much the eight models disagree with each other :  0.026
how wrong the model actually is                    :  0.106
```

The disagreement is **four times smaller than the error**. So "one sigma" was
not one standard deviation of anything; it was about a quarter of one.

### Two wrong explanations, both measured and discarded

**Attempt 1 — "the models are too similar, give them different random seeds."**

Measured: no effect whatsoever, down to the sixteenth decimal place. The reason
is visible in the model configuration: with `subsample=1.0` and
`colsample_bytree=1.0`, LightGBM is fully deterministic. A random seed has
nothing to act on. Reverted.

**Attempt 2 — "then turn on the randomness so the seed matters."**

Measured:

| | before | after |
|---|---|---|
| understated by | 4.04x | **4.41x** (worse) |
| does sigma predict where the error is? | +0.238 | **+0.218** (worse) |
| held-out error | 0.2135 | 0.2157 (slightly worse) |

Reverted.

### The actual explanation

Both attempts assumed the eight models were *too similar*. They are not. They
agree because **they are all making the same mistake**.

This is the distinction between two kinds of being wrong:

- **Variance** — the model would have come out differently with different
  training data. An ensemble can see this: train the members on different
  samples and watch them disagree.
- **Bias** — the whole approach is systematically off, no matter what data you
  train it on. The features do not contain the answer. Every member is wrong in
  the same direction, so they *agree* while all being wrong.

Ensemble spread measures variance only. This model's error is dominated by bias
and by irreducible measurement noise — the same query timed twice on the same
database differs by around 29%. Neither is visible in how much the members
argue, which is why no rearrangement of the ensemble ever helped and why sigma
sat at a quarter of the real error.

**A confidence signal here cannot come from the models' disagreement. It has to
come from the residuals themselves.**

---

## 4. The fix: measure how wrong you have been, not how much you argue

Conformal prediction is the standard tool for this. The idea is almost
embarrassingly simple:

1. Take predictions the model made for data it did not train on.
2. Record how wrong each one was.
3. Sort those errors. Take the 90th percentile.
4. That number is your bound: *"90% of the time, the truth is no worse than my
   prediction plus this."*

It makes no assumption about the shape of the errors, needs no bell curve, and
holds for any model. If the model is bad, the number is large and honest.

### Where it was still going wrong

The textbook version divides each error by that row's sigma before taking the
percentile, so the interval is wider where the model is unsure. That is the
right choice **when sigma carries information**. Here it does not, and the
division was actively destructive: on rows where the eight models happened to
agree exactly, the denominator is near zero, and a single such row hijacks the
whole percentile.

Measured over 2,483 calibration rows, for a candidate predicted at `0.70x`:

| risk level | bound after dividing by sigma | bound without dividing |
|---|---|---|
| 5% | overflow — numerically meaningless | **0.88x** |
| 10% | 1.38x — "not confident" | **0.82x — still a win** |
| 20% | 0.84x | 0.76x |

Dropping the division gives a **constant-width interval**, which is the honest
shape when there is nothing reliable to vary the width by. And it gives a bound
the gate can act on: a plan predicted at `0.70x` is, with 90% confidence, no
worse than `0.82x`.

That is the answer to "why is the model never confident". It could be. The
machinery was dividing by a broken denominator.

---

## 5. What the guard adds on top

Two more layers sit around the gate, both about queries rather than plans.

**A stricter bar for queries never seen before.** The retrospective guard blocks
queries with a *measured history* of getting slower. By definition it cannot
protect a query on its first execution — exactly when the model is most likely
to be extrapolating. So an unseen query gets its safety margin widened, easing
back to normal as evidence arrives.

This is also where a real bug was caught by a test. The widening used to be
applied to *sigma*. Once the bound stopped using sigma, that silently did
nothing — the extra protection for unfamiliar queries disappeared without any
error. It now widens the margin itself, which works under both kinds of bound.

**A deadline at execution time.** Everything above reasons about a plan *before*
it runs. If a prediction turns out wrong, the query used to simply take however
long it took. Now a deviation gets a time limit derived from what PostgreSQL's
own plan costs on that query; past it, the statement is cancelled and
PostgreSQL's plan runs instead. Native never gets a deadline — cancelling it
would leave nothing to fall back to — and neither does a query with no measured
history, since there is nothing to derive one from.

That is the difference between *predicting* that things will not get much worse
and *ensuring* it.

---

## 6. How to see all of this on your own database

```bash
docker compose exec backend python -m app.diagnostics
```

It reports whether the gate is consulting the uncertainty at all, how many
queries it was calibrated on, how far the spread understates the error, and
whether the gate is still delivering what its calibration promised. Every
finding names what was measured and what it means.

The same report is served at `GET /model/diagnostics` and shown in the
dashboard's **Model health** panel.

Two findings are worth recognising:

- **`gate_ignores_uncertainty`** — `confidence_z` is 0, so the uncertainty is
  being multiplied by zero and the "pessimistic" figure on the dashboard is the
  prediction wearing a hat. This had shipped.
- **`uncertainty_understated`** — the spread is smaller than the error by the
  factor shown. Section 3 is what that means.

---

## 7. The short version

- The gate asks two questions: is there a win, and is the downside bounded. It
  used to ask one, and refused a measured 5.9x speedup because of it.
- The ensemble's disagreement is not a usable confidence signal on this model,
  because the error is bias rather than variance. Two attempts to fix that were
  measured and reverted.
- Confidence comes from the model's own past errors instead, without dividing
  by the disagreement. That bound is usable: a predicted `0.70x` is `0.82x` at
  90% confidence.
- An unseen query gets a wider margin; a deviation that overruns gets cancelled.
