# Defect inventory

Indexed list of the defects behind the paper's "more than thirty defects".
Each row points to the `docs/WRITEUP.md` section that records the measurement,
the fix and how it was checked. Where a fix does not fit in one line, the row
says "see §" instead of summarising it.

**Kind:** `M` = measurement failure (a reported number described something other
than the deployed system). `F` = functional failure (the system behaved wrongly;
included because the same audit questions found it).

**Question:** the audit question from the paper (Q1 is this code reached? Q2
does this measurement agree with what runs? Q3 is this worth building? Q4 which
population? Q5 at what granularity does this rule resolve conflicts? Q6 is the
tool wrong, or the thing it measures?). An earlier draft had a seventh question,
"which language reads this string?", for D40 alone; the paper dropped it and
D40 is now listed under Q1.
Assignments for §2.36 onward follow the lists in `PENDING.md`. Those marked `*`
were assigned retrospectively while compiling this file.

| ID | § | Q | Kind | Defect | Affected quantity | Fix / check |
|---|---|---|---|---|---|---|
| D01 | 2.0 | Q1* | M | `pg_hint_plan` created with `CREATE EXTENSION` but not preloaded, so every hint was silently ignored | every hinted-candidate timing before the fix | preloaded via `shared_preload_libraries`; integration tests check that hints bind |
| D02 | 2.4.0 | Q2* | M | the rebuilt "hard" schema's seed data contained none of the intended correlations | headroom of the synthetic workload | see §2.4.0 |
| D03 | 2.4.4 | Q2* | M | `confidence_z` could change no decision: sigma and residuals were in mismatched units | gate calibration | see §2.4.4 |
| D04 | 2.4.5 | Q2* | M | ensemble sigma overconfident relative to out-of-bag error (8x at the time) | gate risk term | conformal bound and scale diagnostics added; see §2.4.5, `CONFIDENCE.md` |
| D05 | 2.16 | Q2* | M | diagnostics panel read a fixed results file and described a different model (-5.9% vs +79.2%) | trustworthiness panel | evidence read from the model registry |
| D06 | 2.18 | Q2* | M | training labels from 2 repetitions and from collections taken under different database states | gate firing, model accuracy | see §2.18 |
| D07 | 2.19 | Q6* | M | calibration refusal blamed model accuracy while a different bound was refusing | calibration verdict | see §2.19 |
| D08 | 2.21 | Q2* | M | replayed calibration said 56% deviations, the live endpoint made 0 | gate | see §2.21 |
| D09 | 2.23 | Q1* | F | registry reported promotion decisions while `train()` still overwrote the served model | promotion | `train()` writes to a scratch path; promotion decides |
| D10 | 2.25 | Q2* | M | regression guard blocked on evidence about a previous system | guard blocks | evidence restricted to a window; see §2.25 |
| D11 | 2.28 | Q2* | M | promotion scored models with a hard-coded policy the server did not run | promotion decision | see §2.28 |
| D12 | 2.29 | Q1* | F | the PostgreSQL 18 upgrade silently disabled the `disable_cost`-based safety veto | safety veto | candidates with disabled nodes dropped from the action space |
| D13 | 2.31 | Q2* | M | promotion metric (a mean) blind to one dominating regression | promotion decision | see §2.31 |
| D14 | 2.36 | Q1 | F | `pg_stat_statements` configured but never installed | query statistics | see §2.36 |
| D15 | 2.42 | Q1 | M | conformal gate fitted on every training run, consulted on none | gate | see §2.42, §2.50 |
| D16 | 2.43 | Q2 | M | every calibration measured `greedy` while the server ran `pairwise_rank` | calibration, policy verdict | one default policy; the duplicate setting deleted |
| D17 | 2.45 | Q2 | M | promotion yardstick was the alphabetically last queries | promotion decision | see §2.45 |
| D18 | 2.47 | Q2 | M | replay could not apply the safety veto | calibration, promotion | veto modelled in the replay |
| D19 | 2.49 | Q2 | M | drift check judged a new gate on runs served under its predecessor | gate drift report | `applied_at` cutoff |
| D20 | 2.50 | Q1 | M | training evaluation scored native as a candidate; the API ignored the swept `gate_mode` | evaluation, gate mode | see §2.50 |
| D21 | 2.52 | Q2* | M | tuner recommended a non-served policy on a 0.8-point difference | tuner recommendation | see §2.52 |
| D22 | 2.53 | Q1* | F | `group_by_template` setting had no effect (read once at import) | settings | see §2.53 |
| D23 | 2.54 | Q2* | M | panel showed `confidence_z` 1.0 while the gate applied 1.5 | dashboard | report all three values |
| D24 | 2.56 | Q1 | M | collection narrower than serving: join-order sample of 8, 39% of the action space in a catch-all family dropped from replays | JOB ceiling (99.2%) | widened; re-measured in §2.80 |
| D25 | 2.65 | Q1 | M | benchmark runs wrote rows that blinded the action-space sweep (flat axis) | action-space sweep | see §2.65 |
| D26 | 2.69 | Q3 | M | three query-level features changed 0 of 16 held-out decisions | feature value | `feature_audit` |
| D27 | 2.70 | Q2/Q4 | M | `has_hint` separated baselines from candidates in training, constant at serving | held-out capture (mean +8.3 points when removed) | feature removed; constancy computed over candidates only |
| D28 | 2.73 | Q2 | M | gate applied per serving path rather than per model | gate | see §2.73 |
| D29 | 2.74 | Q1 | F | `INCLUDE_DISABLE_INDEX` generated zero candidates (depended on index pinning) | action space | see §2.74 |
| D30 | 2.75 | Q1 | M | sweep axis omitted `memoize` and `parallel` | action-space sweep | axis derived from the data |
| D31 | 2.76 | Q2/Q6 | M | collection logged provable no-ops; one was the recorded oracle on 11 of 65 queries | oracle, headroom | no-ops filtered from replays |
| D32 | 2.77 | Q2 | F | a guard block could never lift on evidence, only by timeout | guard | shadow sampling |
| D33 | 2.78 | Q2 | M | two definitions of query identity (Python and SQL) disagreed on the default setting | grouping of logged rows | see §2.78 |
| D34 | 2.79 | Q1 | M | recommended `min_relative_gain = 0` made the caution mechanism inert | gate | see §2.79 |
| D35 | 2.81 | Q2 | M | deterministic measurement session pinned settings away from the server's defaults | training labels | see §2.81 |
| D36 | 2.83 | Q2/Q6 | M | benchmark offered plans with disabled nodes; 36 of 43 vetoes | deviation rate | such candidates dropped (vetoes 43 to 30) |
| D37 | 2.86 | Q6 | M | transfer evaluated on a narrower candidate set than the one served | transfer capture (97.0% to -5.3%) | re-measured on the wide space |
| D38 | 2.89 | Q2 | F | two parsers for one boolean environment variable disagreed on 5 of 10 values | action-space flags | one parser module (`app/env.py`) |
| D39 | 2.90 | Q1 | F | a malformed environment variable crashed every entry point at import | availability | `app/env.py`; subprocess test |
| D40 | 2.91 | Q1 | M | regex `\b` became a backspace in Python and is a backspace in PostgreSQL; the filter never fired | workload report | raw string and `\y` |
| D41 | 2.92 | Q4 | M | collection budget spent on 16 queries, of which 4 were held out | sweep population | `collect_data --held-out` |
| D42 | 2.93 | Q5 | M | "latest run wins" per query removed `scan_method` from the held-out queries | sweep axis | collection warns before narrowing |
| D43 | paper review | Q6* | M | buffer-count noise floor assumed equal buffers meant equal work, and summed cumulative per-node counts | per-query noise floor | withdrawn from the paper; `noise.buffer_signature` still sums (see `paper/README.md`) |
| D44 | paper review | Q2* | M | paired protocol timed identical arms separately; the minimum turned noise into headroom | paired headroom, saving | identical arms share one timing (`tests/test_paired_protocol.py`) |
| D45 | paper review | Q2* | M | paired protocol always timed arms in the same order | paired timings | arm order randomized per repetition |
| D46 | paper review | Q4* | M | repeated benchmark runs were not independent: the guard learned from each run | intervals across runs | each run restored to the same database snapshot |
| D47 | paper review | Q2* | M | benchmark's served arm called `select_plan` on its own candidates: no guard caution (2.0 on unseen queries), no candidate budget, and no index-map families; found when five paired runs (−340 ms) and the end-to-end run (+468 ms cached) disagreed in sign | paired served total, saving, capture (Table V) | served decision made by `planner.plan_query`; enumeration passes `cur`; `tests/test_paired_protocol.py::TestTheServedArmIsTheServersDecision`; pre-fix runs kept as `models/paired_snapshot_run*.json` |

47 entries: 39 measurement, 8 functional.
