# Paper

IEEE conference format only.

| File | Class | Aimed at |
|---|---|---|
| `main.tex` | `IEEEtran` (conference) | ICDE, or any IEEE venue |
| `references.bib` | — | 24 entries, all cited |
| `figures/*.drawio` | draw.io | reference only, see below |

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

**Compile on Overleaf.** Upload `main.tex` and `references.bib`, keep the
default pdfLaTeX compiler, and recompile. Locally, the only TeX image lacks
IEEEtran, makecell and pgfplots, so the paper was checked by compiling a copy
with the `article` class and those packages stubbed: no errors, no undefined
references or citations, and one 0.8 pt overfull line in Table I.

### Figures

Both figures are drawn inside `main.tex` (TikZ and pgfplots), so the paper
builds with no image files. The `.drawio` files are earlier sketches, kept for
reference only.

| Figure | Shows |
|---|---|
| Fig. 1 | offline and online paths, learned steps shaded, the two duplicated steps marked ➊ ➋ |
| Fig. 2 | JOB served vs. oracle totals under the narrow and wide action spaces |

Algorithm 1 gives the paired timing protocol, now with randomized arm order.

---

## Status after the 2026-11-03 revision

### What changed

- **Prose rewritten throughout** in a plainer academic register. No claim was
  added that is not in `docs/WRITEUP.md`, `docs/CONFIDENCE.md` or the code.
- **IEEE structure completed:** a new *Experimental Setup* section (platform,
  workloads, splits) and a formal definition of the capture ratio, Eq. (2), with
  the action space $\mathcal{A}$ explicit in the oracle term. A *Discussion*
  section now holds the reporting recommendations and the tooling list, which
  used to be spread across Results and Conclusion. *Threats to Validity* absorbs
  the reviewer gaps this file used to list (no Bao/Lero baseline, no
  significance tests, the 64.9-point split range).
- **Six audit questions.** Q5, *at what granularity does this rule resolve
  conflicts?* (WRITEUP §2.93), was added; the old Q7 (*which language reads
  this string?*, one regex defect) was dropped as off-thesis.
- **Trimmed to the thesis** (capture-ratio validity and silent measurement
  failures). Removed: the Selection Policies and Cold Start subsections, the
  concurrency paragraph, TPC-H, the Candidate Generation and Features
  subsections (one clause each in §III-A now), the hyperparameter paragraph
  (now only in `ARTIFACTS.md`), the buffer-count noise-floor paragraph (D43),
  the Q1 environment-variable crash story, Q7, the separate Robustness related
  work, the action-space sweep table (its outcome is one paragraph at the end
  of Q5), and the long tool list. The ensemble result is one paragraph in
  §III-B and no longer in the abstract or contributions.
- **Bibliography completed.** Every entry now has a full author list. Checked
  against Crossref and DataCite on 2026-11-03.

### Corrections made to the old draft

| Old draft said | Now says | Why |
|---|---|---|
| Kepler: *"…for **Faster** Parametric Query Optimization"*, SIGMOD booktitle | *"Kepler: Robust Learning for Parametric Query Optimization"*, Proc. ACM Manag. Data 1(1) | the published title drops "Faster"; that is the arXiv title |
| Lero journal, BitE, LEON+, ICDE survey, Redbench, CERT, BayesQO: `and others` or no authors | full author lists | Crossref / DataCite |
| arXiv ids in `eprint` | ids in `howpublished` | IEEEtran.bst ignores `eprint`, so those entries printed with no identifier |
| "rejected per-join-width models … 96 rows across 3 queries" | removed | no source in the repo; the roadmap lists these models as *wanted*, not rejected |
| paired protocol "puts the min-of-draws effect at 66% of headroom" | 66% is the combined effect of drift and oracle bias | the 66% comes from both causes together and cannot be attributed to the min bias alone |
| buffer-count per-query noise floor (§2.88) | removed; only the 18.6% from repeated runs of one plan is used | equal buffer counts do not mean equal CPU work, and `noise.buffer_signature` sums PostgreSQL's per-node counts, which already include their children, so it double-counts |
| "13 survivors … the other seven are these rows plus bushy/memoize" | no count; "every other survivor adds bushy and/or memoize" | see the open item below |

### Still open: check these before submitting

1. **Twelve or thirteen surviving action spaces?** WRITEUP §2.94 says
   "thirteen of the 128" and then lists twelve rows "in full". It also says
   `bushy` is identical "across four pairs", but the twelve rows contain six
   such pairs. The paper now avoids both counts. Re-run
   `python -m app.autotune --policy pairwise_rank` if the count is ever needed
   (the sweep is now one paragraph at the end of §VI-E).
2. **Hardware: confirmed** by the author (Intel Core Ultra 7 155H, 32 GB, NVMe
   SSD, Windows 11, Docker Desktop). The paper cites the tag
   `paper-v2` (the older `icde2027-submission` tag points at the previous
   version). **Push the branch and the tag to GitHub before
   submitting**, or the citation points nowhere.
3. **The 29% figure.** `docs/CONFIDENCE.md` §3 still says a query timed twice
   differs by "around 29%", while everything else, including the paper, uses
   18.6%. The paper does not depend on the 29% figure, but fix the doc.
4. **Redbench** has a published version (doi `10.1145/3735403.3735998`). Its
   venue was not confirmed, so the arXiv version is cited. Swap it in once
   confirmed.
5. **Page budget.** After the trims the body is about 9,100 words, well inside
   a 12-page IEEE limit. CONECCT allows 6–8 pages, so a shorter version is
   still needed there; **check on Overleaf**.
6. **Survey quotes.** The quotes in Table I come from the original reading of
   each paper and were not re-checked in this revision.

### Self-review fixes (2026-11-03, after the two external reviews)

- PostgreSQL version: the early figures (Table II's synthetic and TPC-H
  ceilings, the 18.6% noise floor, the 23/48 candidate counts) predate the
  PG16 → PG18 upgrade (WRITEUP §2.29). The paper now says so.
- JOB and transfer numbers come from logged collection runs (3 repetitions,
  median kept), not from the paired protocol. The paper now says so.
- Paired vs. deploy: the paired protocol also warms caches, so part of the
  difference is cache regime and not only noise.
- Regression-guard concurrency: no longer called "safe". It is a stress test
  with zero errors plus the known window recorded in §2.87.
- The JOB-trained model's 96.5% includes its own training queries, so it is
  marked as not out-of-sample.
- "Dominated by bias" is now stated as an interpretation; conformal coverage is
  below nominal on *most* splits (median 75.5%).
- Speculative phrases removed ("what most reviewers would suggest", "no reason
  to assume it is small", "nothing to learn"); safety-bound values stated (25%,
  1.25×); "published" changed to "reported".

**Not verifiable from this repository:** the quotes in Table I and the
one-line summaries of other papers in §II. Check those against the papers
themselves.

### New experiments (2026-11-03), run on the author's laptop

The database was restored from `backups/lqo_pg18_20260904-1345.dump` into a
fresh PostgreSQL 18.6 container, then analyzed. Raw results are in
`backend/models/`.

- **Paired replication**: `python -m app.benchmark --protocol paired --repeat 5
  --arm-order random --seed 100 --json-out models/paired_5runs_random.json`.
  Mean saving 84 ms (1.4% of native), 95% *t*-interval [−226, +393] ms, so the
  saving is **not distinguishable from zero**. Headroom is stable at 526 ms
  [452, 599]. The guard blocked 0 queries in runs 1–3 and 11–12 in runs 4–5
  (because the runs feed it), and runs 4–5 are where the savings came from.
  The *t*-intervals in the paper were computed from the JSON; the script
  itself prints bootstrap intervals, which are narrower at n = 5.
- **End to end**: `python -m app.e2e_latency --reps 5 --seed 200 --json-out
  models/e2e_latency.json`. Without the cache, +22.5% vs native [+2,544,
  +3,585] ms; with a warm cache, −3.1% [−837, −48] ms. The interval
  resamples queries, not sessions.

### Remaining items from the external reviews

7. **Per-query noise floor from repeated executions** of the same plan. The
   buffer-count floor was removed. If it is ever revived, `buffer_signature`
   should read the root node's counts (which are already cumulative) instead of
   summing every node.
8. **Defect inventory.** Resolved by dropping "Thirty" from the title. The
   abstract still says "more than thirty", which `docs/WRITEUP.md` supports.
9. **Independent replications.** The five paired runs share a guard that learns
   between them. A cleaner design would reset or freeze the guard per run, or
   run with `--no-guard` as a separate arm.

---

## The survey: seven systems read

The paper's Table I came from reading each system for two things: *is there a
best-possible baseline, and how is it computed?* and *how many times is each
plan executed, and which statistic is taken?*

| System | Best-possible baseline | Timing |
|---|---|---|
| Neo | **no**, approximated by "the best query latency seen so far by the system" | median over 50 network initialisations |
| Bao | **yes**, "the optimal hint set … computed by exhaustively executing all query plans" (§5.3) | cold-cache isolation; no per-query repetition stated |
| Balsa | **no**, compared with PostgreSQL and a commercial optimizer | "repeat each experiment 8 times and report the median" |
| Lero | **yes**, "Fastest Found Plan", from exhaustive search | none stated |
| Kepler | **yes, parameterised**: "the optimal plan over some plan set *P*" | "execute the resulting plan three times … take the **minimum**" |
| RELOAD | **no**, the reference is the "expert plan" from the cost-based optimizer | "median as the metric after 6 repetitions", plus a 2σ band |
| LIMAO | **no**, compared with Balsa, Bao and PostgreSQL | none stated, though "execution stability" is one of its metrics |

Two claims did not survive this reading. *The field does not report the
ceiling* is false, because three of seven do. *Nobody states the action space*
is false too, because Kepler writes the optimum over a plan set $P$. What
replaced them is stronger: ceiling discipline and timing discipline move
independently, and no system in the sample has both.

**To extend it to ten:** `references.bib` already has BitE, LEON+ and PARQO.
Add each row with a quote. A row without a quote is an assumption, which is
exactly what this paper argues against.

### The nearest published argument

Lehmann, Sulimov and Stockinger, PVLDB 17(7), 2024, standardise **how the
model is trained and split**. This paper is about **what the result is divided
by**. §II-D handles the distinction explicitly. A reviewer who knows their
paper will ask what is left after it, and the answer is the 99.2% → 91.0%
correction, which a standardised split would not have caught.

Liu, Giridhara and Sabek, PVLDB 18(8), 2025, apply conformal prediction more
thoroughly than this system's gate does. Read it before submitting, because it
may change how §III-D describes the gate.

---

## What was deliberately left out

These parts of the repository were considered and excluded. The reasons are
recorded here so the decision does not have to be made again:

- **The database advisor** (`app/dbadvisor/`, `app/advisor.py`,
  `app/index_advice.py`, `app/hypothetical.py`). A hint fixes one query; it
  does not fix the reason the planner needed one. That is a different
  contribution and **a second paper**.
- **Onboarding onto an arbitrary database** (`app/onboard.py` and friends).
  Interesting engineering, but no measured claim is attached.
- **The dashboard** (`docs/DASHBOARD.md`). Not a research contribution.
- **The full metric catalogue** (`docs/METRICS.md`). Worth one sentence if a
  reviewer asks how outcomes are classified.

If the paper needs to grow for a full-paper venue, grow the survey, not this
list.

## Where to submit (decided: an Indian IEEE venue)

ICDE 2027 was dropped because of cost (full registration about USD 1,040 plus
travel to Copenhagen), so the "[Experiment, Analysis, and Benchmark]" title tag
was removed. Checked 2026-11-03; confirm on each official site.

1. **IEEE CONECCT 2027** (IEEE Bangalore Section). Deadline **15 Dec 2026**;
   Bengaluru, 29–31 Jul 2027. The 2026 rules were 6–8 pages, IEEE template,
   **double-blind** (remove name, university and repository link), and a fee
   of INR 10,000 for IEEE student members (INR 12,000 for non-members) plus 18%
   GST. Needs a shortened, anonymized version.
2. **ICDSE 2027** (CUSAT, Kochi, 22–24 Jul 2027). Full-paper deadline 10 Jun
   2027. The backup if CONECCT rejects.
3. **IEEE HiPC 2027** (Bengaluru, December). The strongest IEEE venue in India;
   the deadline is likely mid-2027.

A paper under review at one venue cannot be submitted to another at the same
time.

**arXiv:** posting a preprint establishes a date and does not count as prior
publication at the venues above. Resolve open items 1 and 2 first, because an
arXiv version is permanent.
