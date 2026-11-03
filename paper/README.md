# Paper

IEEE conference format only.

| File | Class | Aimed at |
|---|---|---|
| `main.tex` | `IEEEtran` (conference) | ICDE, or any IEEE venue |
| `references.bib` | — | 28 entries, all cited |
| `figures/*.drawio` | draw.io | reference only, see below |

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

**Not yet compiled.** There is no LaTeX toolchain on this machine. Citations,
cross-references, environments and brace balance have been checked by script,
but not the layout. Overleaf is the quickest way to compile: upload `main.tex`
and `references.bib`, keep the default pdfLaTeX compiler, and recompile.
Expect to fix a few overfull boxes in the two full-width tables.

### Figures

Both figures are drawn in TikZ inside `main.tex`, so the paper builds
with no image files and no export step. The `.drawio` files were the original
sketches and are kept for reference; editing them no longer changes the paper.

| Figure | Shows |
|---|---|
| Fig. 1 | offline and online paths, learned steps shaded, the two duplicated steps marked ➊ ➋ |
| Fig. 2 | `deploy` vs `paired` arm ordering |

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
- **Seven audit questions, not six.** Q5, *at what granularity does this rule
  resolve conflicts?* (WRITEUP §2.93), was missing from the paper even though
  §VII-C already referred to the defect it found.
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
   `python -m app.autotune --policy pairwise_rank` and put the real number back
   into §VII-C.
2. **Hardware.** §IV-A now lists the machine this repository lives on (Intel
   Core Ultra 7 155H, 32 GB, NVMe SSD, Windows 11, Docker Desktop). **Confirm
   the measurements actually ran on it.** Also commit the working tree, tag a
   release for the paper, and cite the tag in Artifact Availability (there is a
   TODO comment there).
3. **The 29% figure.** `docs/CONFIDENCE.md` §3 still says a query timed twice
   differs by "around 29%", while everything else, including the paper, uses
   18.6%. The paper does not depend on the 29% figure, but fix the doc.
4. **Redbench** has a published version (doi `10.1145/3735403.3735998`). Its
   venue was not confirmed, so the arXiv version is cited. Swap it in once
   confirmed.
5. **Page budget.** The body is about 9,000 words. That is roughly 10–11 IEEE
   two-column pages plus references. It fits ICDE (12 + references) but not a
   6- or 8-page venue. If one is targeted, cut §III-E, §VII-D and the tooling
   list in §VIII-B first.
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

### From the external review (2026-11-03): needs new experiments

The text now claims only what the existing data supports. These need new runs
before the claims can be made stronger:

7. **Repeat the paired benchmark** several times, with randomized or
   counterbalanced arm order (`_paired_medians` always times native, served,
   oracle in that order), and report confidence intervals.
8. **Measure end-to-end latency**, planning included, with and without the
   decision cache. Without it the paper cannot claim a net saving: 215 ms is
   3.8% of native, and uncached planning measured 41–47% of execution time on
   the one query profiled.
9. **Per-query noise floor from repeated executions** of the same plan. The
   buffer-count floor was removed. If it is ever revived, `buffer_signature`
   should read the root node's counts (which are already cumulative) instead of
   summing every node.
10. **Defect inventory.** The reviewer asked for a numbered list backing the
    "thirty" in the title (trigger, consequence, fix, validation for each).
    Either add it to the repo and cite it, or drop the number from the title.

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

## Where to submit (decided: IEEE)

Checked 2026-11-03. Confirm dates on each official site before submitting.

1. **ICDE 2027, Experiment, Analysis & Benchmark (EAB) category.** Round 2
   deadline **Nov 11, 2026, 5 pm Pacific**, via Microsoft CMT. Copenhagen,
   May 17–21, 2027. 12 pages plus references, IEEE format, single-blind, no
   appendix, accept/reject with no revision. EAB papers must start the title
   with "[Experiment, Analysis, and Benchmark]" (already in `main.tex`) and must
   make every artifact public; the repo is public at
   `github.com/parthkansal823/DB_QueryOptimizer`. ICDE's policy also requires
   AI-generated content to be disclosed in the acknowledgments. See
   icde2027.github.io/submission-guidelines.html.
2. **IEEE CONECCT 2027** (IEEE Bangalore Section). Deadline **Dec 15, 2026**;
   Bengaluru, Jul 29–31, 2027. Was 6 pages in 2025, so it would need a
   shortened version.
3. **IEEE HiPC 2027** (Bengaluru, December). The strongest IEEE venue in India;
   expect a deadline around mid-2027.
4. **ISPASS 2027** (IEEE). Fits the measurement-methodology angle. Not yet
   announced; the 2025 deadline was mid-December.

**arXiv:** posting a preprint establishes a date and does not count as prior
publication at the venues above. Resolve open items 1 and 2 first, because an
arXiv version is permanent.
