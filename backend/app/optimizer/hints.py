"""
Turns a list of table aliases into candidate `pg_hint_plan` join orders.

This is the "action space" for the optimizer: each hint is one alternative
plan the learned component can choose between, instead of trusting
Postgres's single default choice.
"""

from __future__ import annotations

import hashlib
import itertools
import re
import random


def _rng_for(tables: list[str]) -> random.Random:
    """
    A generator seeded from the table list itself.

    Sampling used the global `random`, so a query with more tables than the
    candidate budget got a *different* action space on every call. Training
    collection saw one subset, inference saw another, and the model was asked
    to score plans it had never been shown. Two benchmark runs meant to differ
    only in a flag (`--no-guard`, say) were also comparing different candidate
    sets, which quietly undermines every A/B in the writeup.

    This bites well before the "large query" case it was written for: six
    tables is 720 permutations against a budget of 8, and the v2 workload
    (docs/WRITEUP.md 2.9) is built around 5- and 6-way joins.

    Seeding from the tables keeps sampling deterministic per query without a
    global seed, so a process that never calls `random.seed` still reproduces
    its own action space, and two different queries still get different draws.
    """
    # Not a security hash -- just a stable way to turn a table list into a seed.
    digest = hashlib.md5(" ".join(tables).encode(), usedforsecurity=False).hexdigest()
    return random.Random(int(digest[:16], 16))


# Above this many tables, n! is too large to build in memory. The old code
# called `list(itertools.permutations(tables))` *before* sampling it down,
# which meant the factorial explosion it documented as avoided was paid in
# full on every call: 10 tables cost 2.2s and 466MB to return 8 candidates,
# 12 tables exhausts memory outright. JOB's schema has 21 tables, so the
# stretch-goal benchmark was the exact case that could not run.
#
# The cutoff sits at 8 (40320 permutations, ~5MB, ~0.03s) rather than lower
# because staying on the materialising path where it is affordable keeps the
# *sampled orderings themselves* byte-identical to what previous runs drew.
# Changing which orderings get sampled would silently move the action space
# out from under every model already trained against it -- the train/serve
# skew README.md warns about -- so the fix deliberately changes behaviour
# only where the old path could not produce an answer at all.
_MATERIALIZE_MAX_TABLES = 8


def _sample_orders(tables: list[str], k: int, rng: random.Random) -> list[tuple[str, ...]]:
    """
    Up to `k` distinct orderings of `tables`, drawn uniformly.

    Enumerates exhaustively while that is cheap, and falls back to rejection
    sampling of individual shuffles beyond that -- which never materialises
    more than `k` permutations. Both branches consume `rng`, so the result
    stays deterministic per table set either way (see `_rng_for`).
    """
    n = len(tables)

    if n <= _MATERIALIZE_MAX_TABLES:
        all_perms = list(itertools.permutations(tables))
        if len(all_perms) <= k:
            return all_perms
        return rng.sample(all_perms, k)

    # n! here is astronomically larger than k (9 tables is already 362880),
    # so repeated draws practically never collide and the loop terminates in
    # ~k iterations. The `seen` set is what keeps the orderings distinct,
    # which is the property `rng.sample` was providing above.
    seen: set[tuple[str, ...]] = set()
    chosen: list[tuple[str, ...]] = []
    while len(chosen) < k:
        perm = tuple(rng.sample(tables, n))
        if perm not in seen:
            seen.add(perm)
            chosen.append(perm)
    return chosen


def _is_connected_order(order: tuple[str, ...], join_graph: dict[str, list[str]]) -> bool:
    """True if every table after the first joins to one already placed."""
    placed = {order[0]}
    for table in order[1:]:
        if not set(join_graph.get(table, ())) & placed:
            return False
        placed.add(table)
    return True


def _sample_connected_orders(
    tables: list[str], join_graph: dict[str, list[str]], k: int, rng: random.Random
) -> list[tuple[str, ...]]:
    """
    Up to `k` orderings that never force a cartesian product.

    Grows each order one *neighbour* at a time rather than filtering random
    permutations. Above a few tables the connected orders are a vanishing
    fraction of all permutations, so rejection sampling would spend almost all
    its draws on candidates it then discards; building only reachable orders
    means every draw is usable.
    """
    orders: set[tuple[str, ...]] = set()

    # Bounded: a sparse or disconnected graph may not be able to fill the
    # quota, and this must not spin looking for orders that do not exist.
    for _ in range(k * 40):
        if len(orders) >= k:
            break
        remaining = set(tables)
        order = [rng.choice(sorted(remaining))]
        remaining.discard(order[0])
        while remaining:
            reachable = sorted(
                t for t in remaining if set(join_graph.get(t, ())) & set(order)
            )
            if not reachable:
                break  # this draw cannot be completed; start another
            nxt = rng.choice(reachable)
            order.append(nxt)
            remaining.discard(nxt)
        if not remaining:
            orders.add(tuple(order))

    return sorted(orders)


def generate_join_order_candidates(
    tables: list[str],
    max_candidates: int = 8,
    join_graph: dict[str, list[str]] | None = None,
) -> list[str]:
    """
    Generate Leading() hint strings for different join orders of `tables`.

    Given a `join_graph` (from `plan_extractor.extract_join_graph`) only
    **connected** orders are produced. Without one, the previous behaviour
    applies: enumerate while that is affordable, sample beyond it.

    Why connectivity is worth the trouble. A `Leading()` order that introduces
    a table sharing no join predicate with the tables placed so far forces
    Postgres to build the cross product of the two. It prices that at
    `disable_cost` (~1e10), so the plan is never chosen -- the optimizer is
    handed a list of alternatives most of which it cannot use. The waste grows
    with table count: a 6-table chain join has 720 orderings and only 32
    connected ones, so a blind sample of 8 is expected to contain well under
    one usable candidate. This is the "sampling above 5 tables" limitation,
    and constraining the search space is a cheaper fix than learning it.
    """
    if len(tables) <= 1:
        return []

    rng = _rng_for(tables)

    if join_graph:
        if len(tables) <= _MATERIALIZE_MAX_TABLES:
            # Small enough to enumerate: filtering gives *every* connected
            # order, so the budget is spent on a complete set rather than
            # whatever repeated sampling happened to reach.
            connected = [
                p for p in itertools.permutations(tables)
                if _is_connected_order(p, join_graph)
            ]
            chosen = (
                connected
                if len(connected) <= max_candidates
                else rng.sample(connected, max_candidates)
            )
        else:
            chosen = _sample_connected_orders(tables, join_graph, max_candidates, rng)
        # A graph that yields nothing -- a genuine cross join, or conditions the
        # parser could not read -- falls through rather than returning nothing.
        if chosen:
            return [f"/*+ Leading({' '.join(perm)}) */" for perm in chosen]

    chosen = _sample_orders(tables, max_candidates, rng)
    return [f"/*+ Leading({' '.join(perm)}) */" for perm in chosen]


# -- Bushy join trees -------------------------------------------------------
#
# `Leading(a b c d)` names an order, and pg_hint_plan builds it **left-deep**:
# ((a JOIN b) JOIN c) JOIN d. Every candidate this module produced was that
# shape, so the entire action space was one corner of the plan space.
#
# The nested form `Leading(((a b) (c d)))` specifies the tree itself, which
# lets two pairs be built independently and then joined -- a *bushy* plan. That
# matters when both halves filter down hard: a left-deep tree has to carry the
# first table's rows through every subsequent join, while a bushy one builds two
# small intermediates and joins those.
#
# This project already knew the difference mattered. `plan_tree.py` computes
# `tree_bushiness` as a feature and describes it as "0.0 = perfectly left-deep
# (what Leading() produces)" -- the encoder was measuring a property the
# generator could not vary. Neo and Bao both search bushy plans for the same
# reason.
#
# Only balanced splits are produced, and only where both halves are internally
# connected and connected to each other. An unconnected half is a cartesian
# product, which Postgres prices at disable_cost and never picks.


def _is_connected_set(tables: tuple[str, ...], join_graph: dict[str, list[str]]) -> bool:
    """Is this set of tables reachable from any one of them, using only itself?"""
    if len(tables) <= 1:
        return True
    remaining = set(tables)
    frontier = {remaining.pop()}
    while frontier:
        table = frontier.pop()
        neighbours = set(join_graph.get(table, ())) & remaining
        remaining -= neighbours
        frontier |= neighbours
    return not remaining


def _halves_touch(
    left: tuple[str, ...], right: tuple[str, ...], join_graph: dict[str, list[str]]
) -> bool:
    """Do the two halves share a join predicate, so joining them is not a cross product?"""
    right_set = set(right)
    return any(set(join_graph.get(t, ())) & right_set for t in left)


def generate_bushy_candidates(
    tables: list[str],
    join_graph: dict[str, list[str]] | None = None,
    max_candidates: int = 6,
) -> list[str]:
    """
    `Leading(((a b) (c d)))` hints: plan trees no left-deep ordering can express.

    Needs at least four tables -- below that every tree is left-deep -- and a
    join graph, because a bushy split is only worth proposing when both halves
    are genuinely joinable.
    """
    if len(tables) < 4 or not join_graph:
        return []

    rng = _rng_for(tables)
    ordered = sorted(tables)
    half = len(ordered) // 2

    splits: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    seen: set[tuple] = set()
    for left in itertools.combinations(ordered, half):
        right = tuple(t for t in ordered if t not in left)
        # A split and its mirror image describe the same tree.
        key = tuple(sorted((left, right)))
        if key in seen:
            continue
        seen.add(key)
        if not _is_connected_set(left, join_graph):
            continue
        if not _is_connected_set(right, join_graph):
            continue
        if not _halves_touch(left, right, join_graph):
            continue
        splits.append((left, right))

    if not splits:
        return []
    if len(splits) > max_candidates:
        splits = rng.sample(splits, max_candidates)

    return [
        f"/*+ Leading((({' '.join(left)}) ({' '.join(right)}))) */"
        for left, right in sorted(splits)
    ]


# -- Per-table scan methods -------------------------------------------------
#
# `HINT_SETS` below turns operators off for the *whole statement*:
# `Set(enable_seqscan off)` bans sequential scans everywhere, including on the
# small lookup table where one is obviously right. pg_hint_plan can say it per
# relation instead -- `IndexScan(orders)` forces an index scan on `orders` and
# leaves every other table alone.
#
# That is the difference between "never scan anything sequentially" and "stop
# scanning *this* table sequentially", and on a join between a large filtered
# table and a small one it is usually the second that is wanted.
# `IndexOnlyScan` is in the default tuple because it is the one scan type the
# other three cannot substitute for: it answers the query from the index alone
# and never touches the heap, so on a covering index it does a fraction of the
# I/O an index scan does. Postgres only picks it when it believes the visibility
# map is current enough, which is a judgement about vacuum state rather than
# about the query -- exactly the kind of estimate this project exists to
# second-guess.
SCAN_METHODS = ("SeqScan", "IndexScan", "BitmapScan", "IndexOnlyScan")

# TID scans only apply to queries with a `ctid` predicate, which no query in
# this workload has. Kept so the family is complete and so a `ctid`-driven
# workload can switch it on, but out of the default tuple: on every other query
# `TidScan(t)` is a planning round trip that returns the native plan.
ALL_SCAN_METHODS = SCAN_METHODS + ("TidScan",)


def generate_scan_method_candidates(
    tables: list[str], methods: tuple[str, ...] = SCAN_METHODS, max_tables: int = 3
) -> list[str]:
    """One hint per (table, scan method), for the first few tables."""
    if not tables:
        return []
    # Bounded: every table x method is a planning round trip, and the tables a
    # query lists first are the ones its FROM clause leads with.
    chosen = sorted(dict.fromkeys(tables))[:max_tables]
    return [f"/*+ {method}({table}) */" for table in chosen for method in methods]


# -- Negative scan hints ----------------------------------------------------
#
# `IndexScan(orders)` says "use an index on orders". If no index is usable it is
# not a request the planner can honour, so it prices the alternatives at
# disable_cost and hands back a plan chosen under a constraint that was never
# satisfiable -- a wasted candidate that still costs a planning round trip.
#
# `NoSeqScan(orders)` says only "not a sequential scan", and leaves the planner
# free to choose between index, index-only and bitmap on its own. It is a
# strictly weaker constraint than any of the positive forms, which makes it the
# safer way to express the thing that is actually usually wanted: *this
# particular* table is being scanned when it should not be.
#
# The distinction matters most where the positive form is most tempting. On a
# join between a large filtered table and a small lookup table, forcing an index
# scan on the large table is right and forcing one on the lookup table is wrong;
# the negative form on the large table gets the first without risking the
# second.
NEGATIVE_SCAN_METHODS = ("NoSeqScan", "NoIndexScan", "NoBitmapScan")

# `NoIndexScan` already bans index-only scans as well (pg_hint_plan documents it
# as covering both), so `NoIndexOnlyScan` is only distinct when you want to
# forbid the index-only variant while still permitting a plain index scan -- a
# narrow case, hence out of the default tuple. `NoTidScan` is here for
# completeness, and is a no-op on any query without a `ctid` predicate.
ALL_NEGATIVE_SCAN_METHODS = NEGATIVE_SCAN_METHODS + ("NoIndexOnlyScan", "NoTidScan")


def generate_negative_scan_candidates(
    tables: list[str],
    methods: tuple[str, ...] = NEGATIVE_SCAN_METHODS,
    max_tables: int = 3,
) -> list[str]:
    """`NoSeqScan(t)` and friends: ban one scan type on one table."""
    if not tables:
        return []
    chosen = sorted(dict.fromkeys(tables))[:max_tables]
    return [f"/*+ {method}({table}) */" for table in chosen for method in methods]


# -- Index selection --------------------------------------------------------
#
# Every scan hint above chooses an access *method*. When a table carries several
# indexes the harder question is which one, and that is a separate decision the
# planner also makes from estimates: it prices each candidate index by the
# selectivity it expects the predicate to have and picks the cheapest. A
# correlated pair of columns -- the failure this whole project is built around
# -- makes a two-column index look far less selective than it is, so the planner
# reaches for a different index, or for a sequential scan.
#
# `IndexScan(table index)` pins the choice. It is the only family here that
# needs to know something about the schema rather than the query, so callers
# pass a `{table: [index, ...]}` map read from `pg_index`; without one this
# generates nothing rather than guessing at names.
#
# **That map must be keyed by the query's alias, not the relation name.**
# pg_hint_plan addresses relations the way the query spells them, so on
# `FROM orders o` the relation is `o`; `IndexScan(orders orders_pkey)` is not an
# error and not a warning -- it is silently ignored, which looks exactly like a
# hint that applied and changed nothing. Measured here: `Parallel(order_items 4
# hard)` planned at cost 7870 (unchanged), `Parallel(oi 4 hard)` at 1774. Use
# `optimizer/catalog.py::indexes_by_alias`, which rekeys the catalog through the
# baseline plan's own alias mapping.
def generate_index_candidates(
    table_indexes: dict[str, list[str]] | None,
    methods: tuple[str, ...] = ("IndexScan", "BitmapScan", "IndexOnlyScan"),
    max_tables: int = 2,
    max_indexes_per_table: int = 3,
) -> list[str]:
    """`IndexScan(t idx)`: pin the specific index, not just the access method."""
    if not table_indexes:
        return []
    candidates: list[str] = []
    for table in sorted(table_indexes)[:max_tables]:
        for index in sorted(table_indexes[table])[:max_indexes_per_table]:
            for method in methods:
                candidates.append(f"/*+ {method}({table} {index}) */")
    return candidates


# `DisableIndex(t idx)` plans as though the index were not there. Its use here
# is not to speed a query up -- it is the missing half of the unused-index
# finding in `app/dbadvisor/indexes.py`.
#
# That advisor can say an index has zero scans in `pg_stat_user_indexes`, which
# is evidence about the past and says nothing about whether some query would
# have been slower without it. `DisableIndex` answers the question directly:
# plan the workload as if it had already been dropped, and compare. An index
# recommended for deletion on statistics alone is a recommendation that cannot
# be checked before it is irreversible.
#
# **Not supported by the pg_hint_plan on this server, and this is why the family
# is not wired into `generate_candidates`.** The keyword is in the current
# pg_hint_plan documentation; the extension installed here is 1.6.3, which does
# not have it:
#
#     INFO:  pg_hint_plan: hint syntax error at or near "DisableIndex(u users_pkey)"
#     DETAIL:  Unrecognized hint keyword "DisableIndex".
#
# Note the severity: **INFO**. With the default `client_min_messages` that
# message is never delivered, so an unrecognised keyword is not an error, not a
# warning, and not a failure -- the query simply plans unhinted, exactly as if
# the hint had been applied and changed nothing. That is the same silent failure
# as docs/WRITEUP.md 2.0, and it is why every keyword added to this module was
# checked against the running extension with `debug_print` on rather than
# against the documentation. All 31 others were accepted; this one was not.
#
# Kept, because the technique is sound and the keyword exists upstream. Turning
# it on is a matter of upgrading pg_hint_plan and adding an
# `include_disable_index` flag -- not of changing anything here.
def generate_disable_index_candidates(
    table_indexes: dict[str, list[str]] | None,
    max_tables: int = 2,
    max_indexes_per_table: int = 3,
) -> list[str]:
    """
    `DisableIndex(t idx)`: plan as if this index had already been dropped.

    Requires a pg_hint_plan newer than 1.6.3 -- see the note above. Deliberately
    left out of `generate_candidates`.
    """
    if not table_indexes:
        return []
    candidates: list[str] = []
    for table in sorted(table_indexes)[:max_tables]:
        for index in sorted(table_indexes[table])[:max_indexes_per_table]:
            candidates.append(f"/*+ DisableIndex({table} {index}) */")
    return candidates


# The `*Regexp` forms restrict the scan to indexes whose *name* matches a POSIX
# pattern. Where index names are known -- and this system reads them from the
# catalog -- naming them outright is strictly more precise, so this family adds
# nothing the one above cannot already express.
#
# It earns its place in exactly one situation: a naming convention that outlives
# any particular index. `IndexScanRegexp(orders .*_created_at)` keeps meaning
# the right thing after the index is rebuilt under a new name, which a pinned
# name does not. Off by default; supply your own patterns to use it.
def generate_index_regexp_candidates(
    tables: list[str],
    patterns: tuple[str, ...] = (),
    methods: tuple[str, ...] = (
        "IndexScanRegexp",
        "BitmapScanRegexp",
        "IndexOnlyScanRegexp",
    ),
    max_tables: int = 2,
) -> list[str]:
    """`IndexScanRegexp(t pattern)`: restrict to indexes matching a name pattern."""
    if not tables or not patterns:
        return []
    chosen = sorted(dict.fromkeys(tables))[:max_tables]
    return [
        f"/*+ {method}({table} {pattern}) */"
        for table in chosen
        for pattern in patterns
        for method in methods
    ]


# -- Negative join methods --------------------------------------------------
#
# `generate_join_method_candidates` below forces one operator across every join
# in the tree, and to do that it has to pin the join order too -- the method
# hints name the growing prefixes of a `Leading()`. So a candidate meant to
# answer "would a hash join be better here" also silently answers "and in this
# order", and a loss cannot be attributed to either.
#
# The negative forms need no `Leading()` at all. `NoNestLoop(a b c)` bans nested
# loops among those relations however the planner chooses to order them, which
# isolates the operator question from the ordering question. It is also the form
# that matches the actual failure mode: a nested loop chosen because the outer
# side was estimated at 3 rows and returned 30,000 is a disaster, and what is
# wanted is not a specific replacement but simply *not that*.
NEGATIVE_JOIN_METHODS = ("NoNestLoop", "NoHashJoin", "NoMergeJoin")


def generate_negative_join_method_candidates(
    tables: list[str], methods: tuple[str, ...] = NEGATIVE_JOIN_METHODS
) -> list[str]:
    """`NoNestLoop(a b c)`: ban an operator across the whole join, order-free."""
    if len(tables) < 2:
        return []
    relations = " ".join(sorted(dict.fromkeys(tables)))
    return [f"/*+ {method}({relations}) */" for method in methods]


# -- Memoize ----------------------------------------------------------------
#
# Memoize (PG14+) caches the inner side of a nested loop across repeated outer
# values. Whether it pays depends entirely on how many *distinct* values the
# outer side has -- a high-repeat outer side turns N inner scans into a handful,
# and a unique one adds a hash table nobody reads.
#
# The planner decides from `ndistinct`, which is the single statistic most often
# wrong on the correlated columns this project targets, so it is a decision
# worth putting in the action space. `Set(enable_memoize off)` in `HINT_SETS`
# already covers the statement-wide ban; these are the per-join forms, which can
# leave memoization on where it helps and off where it does not.
#
# `Memoize()` is documented as permissive rather than forcing -- it allows the
# topmost join among those relations to memoize, and the planner may still
# decline.
def generate_memoize_candidates(tables: list[str]) -> list[str]:
    """`Memoize(a b)` / `NoMemoize(a b)`: per-join inner-result caching."""
    if len(tables) < 2:
        return []
    relations = " ".join(sorted(dict.fromkeys(tables)))
    return [f"/*+ Memoize({relations}) */", f"/*+ NoMemoize({relations}) */"]


# -- Per-table parallelism --------------------------------------------------
#
# `TUNING_SETTINGS` already varies `max_parallel_workers_per_gather`, but that
# is a statement-wide ceiling, and it is only a ceiling: Postgres still applies
# its own size heuristic and will refuse to parallelise a table it considers too
# small, so `Set(max_parallel_workers_per_gather 4)` on such a table changes
# nothing at all.
#
# `Parallel(table 4 hard)` overrides the heuristic and demands the workers. That
# is the whole point -- the heuristic is a rule about table size, and the cases
# worth attacking are the ones where a small table is expensive (an expensive
# filter, a costly function) or a large one is cheap. `Parallel(table 0)`
# removes a Gather whose startup cost is not repaid.
#
# `soft` respects the planner's own judgement, which makes it the safe default
# and also, mostly, a no-op; `hard` is the one that adds something to the space.
PARALLEL_DEGREES: tuple[str, ...] = ("0", "2", "4")


def generate_parallel_candidates(
    tables: list[str],
    degrees: tuple[str, ...] = PARALLEL_DEGREES,
    max_tables: int = 2,
    mode: str = "hard",
) -> list[str]:
    """`Parallel(t 4 hard)`: force a worker count on one table."""
    if not tables:
        return []
    chosen = sorted(dict.fromkeys(tables))[:max_tables]
    candidates: list[str] = []
    for table in chosen:
        for degree in degrees:
            # Zero workers means "no parallelism at all", where hard/soft has
            # nothing left to qualify -- pg_hint_plan takes the bare form.
            qualifier = "" if degree == "0" else f" {mode}"
            candidates.append(f"/*+ Parallel({table} {degree}{qualifier}) */")
    return candidates


# A leading `/*+ ... */` block, and the directives inside one.
_LEADING_HINT = re.compile(r"\A\s*/\*\+(?P<body>.*?)\*/\s*", re.DOTALL)
# Directive names are matched by regex; their arguments are scanned, because a
# regex cannot count parentheses to arbitrary depth.
#
# A bushy join hint nests two levels -- `Leading(((a b) (c d)))` -- and a pattern
# allowing only one, `\((?:[^()]|\([^()]*\))*\)`, matches nothing at all on it.
# The directive was then dropped from the merge, so chaining a second pass onto
# a bushy result silently discarded the bushy plan. Same failure class as
# everything else in this file: a hint that quietly stops existing.
_DIRECTIVE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _find_directives(text: str) -> list[str]:
    """Every `Name(...)` in `text`, with parentheses balanced to any depth."""
    found: list[str] = []
    position = 0
    while True:
        match = _DIRECTIVE_NAME.search(text, position)
        if not match:
            return found
        cursor = match.end()
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] != "(":
            # A bare word -- not a directive. Step past it and keep looking.
            position = match.end()
            continue

        depth = 0
        end = cursor
        for end in range(cursor, len(text)):
            if text[end] == "(":
                depth += 1
            elif text[end] == ")":
                depth -= 1
                if depth == 0:
                    break
        if depth != 0:
            # Unbalanced: not something to hand to the planner.
            return found
        found.append(match.group() + text[cursor : end + 1])
        position = end + 1


def split_hints(query: str) -> tuple[list[str], str]:
    """
    Peel every leading hint comment off `query`, returning (directives, sql).

    Handles a *stack* of them, not just one, because a stack is exactly what a
    user produces by feeding this system its own output back in.
    """
    directives: list[str] = []
    remaining = query
    while True:
        match = _LEADING_HINT.match(remaining)
        if not match:
            return directives, remaining
        directives.extend(_find_directives(match.group("body")))
        remaining = remaining[match.end():]


def _directive_key(directive: str) -> str:
    """
    What a later directive replaces.

    `SeqScan(a)` and `SeqScan(b)` are different instructions and both survive; a
    second `Leading(...)` replaces the first, because a query has one join order;
    `Set(work_mem ...)` keys on the GUC, so a new value for the same setting
    wins and a different setting is kept alongside.
    """
    name, _, rest = directive.partition("(")
    name = name.strip()
    arguments = rest[:-1].split()
    if name == "Set":
        return f"Set:{arguments[0] if arguments else ''}"
    if name == "Leading":
        return "Leading"
    return f"{name}:{' '.join(arguments)}"


def apply_hint(query: str, hint: str) -> str:
    """
    Attach a hint to a query, **merging** with any hint it already carries.

    This used to prepend unconditionally, and that is wrong in the one workflow
    users actually reach for: optimise a query, copy the optimised SQL, paste it
    back in to improve it further. The output already carries a hint comment, so
    prepending produces two:

        /*+ Set(enable_nestloop off) */
        /*+ Leading(u o) */
        SELECT ...

    pg_hint_plan reads **only the first** such comment and discards the rest --
    not as an error, not as a warning, and not in the "not used" or "error hint"
    sections of its own debug output. The second comment simply does not exist as
    far as it is concerned.

    Measured on this server with `pg_hint_plan.debug_print = on`, one query, one
    join, cost against `used hint:`:

        native                          5332.3
        /*+ Leading(u o) */             5332.3   used: Leading(u o)
        /*+ Set(enable_hashjoin off) */ 14365.1  used: Set(enable_hashjoin off)

        A then B, stacked               5332.3   used: Leading(u o)
        B then A, stacked               14365.1  used: Set(enable_hashjoin off)
        merged into one comment         14365.1  used: Leading(u o) Set(...)

    Whichever comment comes first wins outright. So on the second pass the
    optimizer's new hint was prepended, applied, and **the hint the user pasted
    in was silently thrown away** -- the second pass did not build on the first,
    it replaced it. Same silence as docs/WRITEUP.md 2.0, reached by an obvious
    user action rather than by a configuration mistake.

    (An earlier reading of this said pg_hint_plan rejected the whole thing and
    planned natively. That was wrong: `error hint:` is an empty section header in
    the debug output, printed on every statement, not a message about this one.)

    Merging into a single comment is the form pg_hint_plan documents and
    accepts: `/*+ Leading(u o) Set(enable_nestloop off) */` is one hint carrying
    two directives. Later directives win on conflict, which matches
    pg_hint_plan's own last-one-wins rule for a repeated hint type -- so the
    newly chosen candidate overrides whatever the query arrived with, and
    stacking passes converges instead of accumulating contradictions.
    """
    existing, bare = split_hints(query)
    incoming = _find_directives(hint)
    if not existing and not incoming:
        return bare

    merged: dict[str, str] = {}
    for directive in existing + incoming:
        merged[_directive_key(directive)] = directive

    return "/*+ " + " ".join(merged.values()) + " */\n" + bare


def corrected_cardinality_hint(rows_hints: list[str]) -> str | None:
    """
    Wrap learned `Rows(...)` corrections into a single hint comment.

    Unlike every other candidate this module produces, the result does not
    force a plan. It hands the planner better row estimates and lets it decide
    for itself -- so the plan that comes back can be one no `Leading()` or
    `Set()` candidate is able to express. See `optimizer/cardinality.py`.
    """
    if not rows_hints:
        return None
    return "/*+ " + " ".join(rows_hints) + " */"


# -- Stretch goal: join *method* selection, not just join order -------------
#
# pg_hint_plan's HashJoin()/NestLoop()/MergeJoin() hints force the join
# method used when exactly the given set of relations is joined together.
# `Leading(a b c d)` builds a left-deep tree, so its join nodes are the
# growing prefixes (a b), (a b c), (a b c d) -- forcing one method for all of
# them approximates "use this method throughout the plan" without needing a
# full per-node hint-tree generator.
JOIN_METHODS = ("HashJoin", "NestLoop", "MergeJoin")


def _method_hint_for_order(order: tuple[str, ...], method: str) -> str:
    prefixes = [order[: i + 1] for i in range(1, len(order))]
    method_tokens = " ".join(f"{method}({' '.join(p)})" for p in prefixes)
    leading = f"Leading({' '.join(order)})"
    return f"/*+ {leading} {method_tokens} */"


def generate_join_method_candidates(
    tables: list[str], max_orders: int = 4, methods: tuple[str, ...] = JOIN_METHODS
) -> list[str]:
    """
    Generate `Leading()` + forced-method hint strings: `max_orders` join
    orders, each repeated once per method in `methods`. This is the action
    space for join-method selection -- kept separate from
    `generate_join_order_candidates` so existing callers (order-only) are
    unaffected; opt in via `generate_candidates(..., include_join_methods=True)`.
    """
    if len(tables) <= 1:
        return []

    orders = _sample_orders(tables, max_orders, _rng_for(tables))

    return [_method_hint_for_order(order, method) for order in orders for method in methods]


def generate_candidates(
    tables: list[str],
    max_order_candidates: int = 8,
    include_join_methods: bool = False,
    max_method_orders: int = 4,
    include_hint_sets: bool = True,
    include_tuning: bool = False,
    include_bushy: bool = False,
    include_scan_methods: bool = False,
    include_negative_scans: bool = False,
    include_negative_joins: bool = False,
    include_memoize: bool = False,
    include_parallel: bool = False,
    include_enable_settings: bool = False,
    include_extended_toggles: bool = False,
    include_cost_model: bool = False,
    join_graph: dict[str, list[str]] | None = None,
    table_indexes: dict[str, list[str]] | None = None,
) -> list[str]:
    """
    Order-only candidates, plus (opt-in) order x method, plus hint sets, plus
    (opt-in) resource and cost-model settings.

    `include_tuning` is separate from `include_hint_sets` because the two are
    different kinds of action. A hint set removes an operator from the
    planner's menu; a tuning setting changes what the executor is given to work
    with, and can beat every plan shape on the menu without changing the shape
    at all.
    """
    # Ordered by measured value, because `planner.budget_for` truncates this
    # list from the end and therefore decides what a query never gets to try.
    #
    # It used to run: join orders, join methods, operator toggles, resources.
    # The comment justifying that in `planner.py` argued the resource settings
    # "are precisely the ones that cannot help a small query" -- giving a hash
    # join over two thousand rows more `work_mem` does nothing, while reordering
    # its joins might. Plausible, and measured over 2,364 collected candidates
    # on this database, wrong:
    #
    #     resource / cost model   n=542    mean ratio 0.99   median 0.93
    #     join order              n=486    mean ratio 1.27   median 1.12
    #     operator toggle        n=1328    mean ratio 2.91   median 1.58
    #
    # The resource family is the only one whose median is *below* native, and it
    # was the family being thrown away. Measured across the workload, the budget
    # dropped resource candidates on **every** query -- on seven of eight it was
    # the only thing it dropped.
    #
    # Fast families first, so the budget spends what it has on the candidates
    # most likely to win and truncates the ones least likely to.
    candidates = generate_join_order_candidates(
        tables, max_candidates=max_order_candidates, join_graph=join_graph
    )
    if include_tuning:
        candidates += generate_tuning_candidates()
    if include_join_methods and len(tables) >= 2:
        candidates += generate_join_method_candidates(tables, max_orders=max_method_orders)
    if include_hint_sets:
        candidates += generate_hint_sets()
    if include_bushy:
        candidates += generate_bushy_candidates(tables, join_graph=join_graph)
    if include_scan_methods:
        candidates += generate_scan_method_candidates(tables)
    if include_negative_scans:
        candidates += generate_negative_scan_candidates(tables)
    if include_negative_joins:
        candidates += generate_negative_join_method_candidates(tables)
    if include_memoize:
        candidates += generate_memoize_candidates(tables)
    if include_parallel:
        candidates += generate_parallel_candidates(tables)
    if include_enable_settings:
        candidates += generate_enable_candidates()
    if include_extended_toggles:
        candidates += generate_extended_hint_sets()
    if include_cost_model:
        candidates += generate_cost_model_candidates()
    # Index pinning is last because it is the only family that can be empty for
    # a reason outside the query -- no catalog map was supplied -- and the
    # budget in `planner.py` truncates from the end.
    if table_indexes:
        candidates += generate_index_candidates(table_indexes)
    return candidates


# -- Bao-style operator hint sets ------------------------------------------
#
# Join *order* alone turns out to be a weak action space. For a two-table
# query there are only two orderings and PostgreSQL already picks the better
# one, so every "candidate" comes back as the identical plan with the
# identical cost -- the optimizer appears to run but cannot possibly improve
# anything.
#
# Bao (SIGMOD 2021) takes a different action space: toggle whole *classes of
# operator* off and let the planner re-plan under that restriction. Disabling
# nested loops forces a hash or merge join throughout; disabling sequential
# scans pushes the planner toward indexes. Each toggle produces a genuinely
# different plan with a genuinely different cost, which is what gives the
# model something to choose between.
#
# These are `Set()` hints on planner GUCs, which pg_hint_plan applies for the
# duration of the statement. PostgreSQL never truly *disables* an operator --
# it prices it at disable_cost (~1e10) -- so a plan is always producible even
# if every listed operator is off. That is why no combination here can fail.
HINT_SETS: tuple[tuple[str, ...], ...] = (
    ("enable_nestloop",),
    ("enable_hashjoin",),
    ("enable_mergejoin",),
    ("enable_seqscan",),
    ("enable_indexscan",),
    ("enable_nestloop", "enable_mergejoin"),      # -> hash joins
    ("enable_hashjoin", "enable_mergejoin"),      # -> nested loops
    ("enable_nestloop", "enable_hashjoin"),       # -> merge joins
    ("enable_nestloop", "enable_seqscan"),
    ("enable_hashjoin", "enable_seqscan"),
    ("enable_mergejoin", "enable_seqscan"),
    ("enable_material",),
    # Memoize (PG14+) caches the inner side of a nested loop across repeated
    # parameter values. It is a large win when the outer side repeats and pure
    # overhead when it does not, and the planner decides from the estimate this
    # project exists because it distrusts.
    ("enable_memoize",),
    # Hash aggregation can spill; a sorted group-by sometimes beats it, and
    # this is the only way to find out which.
    ("enable_hashagg",),
    # Incremental sort exploits an already-partially-sorted input. When the
    # planner is wrong about how sorted that input is, it is slower than a
    # plain sort.
    ("enable_incremental_sort",),
)


# -- Resource and cost-model settings --------------------------------------
#
# Every family above changes the *shape* of the plan: which order, which join
# operator, which scan. None of them changes what the executor is given to work
# with, and on this database two of those levers move latency more than plan
# shape does.
#
# **Parallelism.** Postgres will only consider a parallel plan when the table is
# large enough and `max_parallel_workers_per_gather` allows it, and the decision
# is made on estimates this project exists because it distrusts. Forcing workers
# on turns a sequential scan of a large table into a parallel one; forcing them
# off removes Gather nodes whose startup cost is not repaid on a small result.
# Neither is reachable through any `Leading()` or operator toggle.
#
# **work_mem.** A hash join whose hash table does not fit spills to disk in
# batches, and a sort that does not fit becomes an external merge. Both are
# large, cliff-edged latency effects that the planner prices only through its
# estimate of the input size -- the estimate that is wrong. Raising work_mem is
# the single most common fix for a slow analytical query in practice, and the
# planner cannot choose it for itself.
#
# `random_page_cost` retunes the cost model toward index scans (the standard
# adjustment for storage that is not a spinning disk), and the collapse limits
# change how much of the join tree the planner is willing to reorder at all --
# `join_collapse_limit 1` pins the order written in the query, which is a
# genuinely different action from any `Leading()` hint because it also stops
# subquery pull-up.
#
# Values are given as plain numbers so no quoting is involved: `work_mem` is in
# kilobytes, so 65536 is 64MB. These are `Set()` hints, which pg_hint_plan
# applies for the statement only, so nothing here leaks into the session.
TUNING_SETTINGS: tuple[tuple[str, str], ...] = (
    ("max_parallel_workers_per_gather", "4"),
    ("max_parallel_workers_per_gather", "0"),
    ("work_mem", "65536"),
    ("random_page_cost", "1.1"),
    ("join_collapse_limit", "1"),
    # JIT compilation is priced by the planner's cost estimate, and it is a
    # fixed cost paid before the first row. On a query the planner *thinks* is
    # expensive and is not, compiling can take longer than executing -- a
    # common and easily-missed regression on short analytical queries.
    ("jit", "off"),
    # Above `geqo_threshold` joined relations PostgreSQL stops searching
    # exhaustively and runs a genetic algorithm instead, which is fast,
    # approximate, and *non-deterministic* -- the same query can get different
    # plans on different runs. Raising the threshold buys an exhaustive search
    # on the large joins where the plan space is worth searching properly, and
    # is the setting that matters most on JOB's 8-to-17 table queries.
    ("geqo_threshold", "20"),
)


# -- Settings worth turning ON ---------------------------------------------
#
# `HINT_SETS` is built entirely out of flags to switch *off*, which quietly
# assumes every planner feature is on by default. Two are not:
# `enable_partitionwise_join` and `enable_partitionwise_aggregate` both default
# to **off**, so no combination of `off` toggles can reach the plan shapes they
# unlock -- joining matching partitions pairwise instead of appending first, and
# aggregating per partition. On a partitioned schema that is a shape the action
# space simply could not express.
#
# They are off by default for a reason the manual is explicit about: planning
# gets significantly more expensive, and the number of nodes bounded by
# `work_mem` grows with the partition count, so memory use can climb sharply.
# That is an argument for measuring them per query, which is what this project
# is for, rather than for setting them globally.
# Nine of the server's 21 `enable_*` GUCs appear in `HINT_SETS`. These are the
# rest that can change a plan on a non-partitioned schema, kept separate because
# each one is another planning round trip and most are narrow.
#
# `enable_partition_pruning`, `enable_async_append` and `enable_tidscan` are
# deliberately absent: pruning and async append need partitions or foreign
# tables, and a TID scan needs a `ctid` predicate. On this workload all three
# reproduce the native plan, and a candidate that cannot differ is a round trip
# spent to learn nothing.
EXTENDED_HINT_SETS: tuple[tuple[str, ...], ...] = (
    # The middle ground between sequential and index scan. Banning it forces the
    # planner to commit either way, which is exactly the choice it makes badly
    # when a correlated predicate makes an index look less selective than it is.
    ("enable_bitmapscan",),
    # Distinct from `enable_indexscan`: this bans only the index-*only* variant,
    # isolating the visibility-map decision from the access-method decision.
    ("enable_indexonlyscan",),
    # Discourages explicit Sort nodes, pushing the planner toward index order or
    # hash aggregation instead.
    ("enable_sort",),
    # Gather Merge preserves each worker's sort order at a cost; plain Gather
    # reads in whatever order is convenient and destroys it.
    ("enable_gathermerge",),
    # Without parallel hash every worker builds its own full copy of the hash
    # table -- a large memory difference on a big build side, and the reason
    # `NestLoop`/`MergeJoin` candidates and parallel candidates interact badly
    # (manual 15.3.2: inner sides of nested loop and merge join are never
    # parallel).
    ("enable_parallel_hash",),
    # Affects ORDER BY / DISTINCT inside aggregate calls.
    ("enable_presorted_aggregate",),
)


def generate_extended_hint_sets(
    hint_sets: tuple[tuple[str, ...], ...] = EXTENDED_HINT_SETS,
) -> list[str]:
    """Operator toggles beyond the Bao-style core set."""
    return [
        "/*+ " + " ".join(f"Set({flag} off)" for flag in flags) + " */"
        for flags in hint_sets
    ]


ENABLE_SETTINGS: tuple[tuple[str, ...], ...] = (
    ("enable_partitionwise_join",),
    ("enable_partitionwise_aggregate",),
    ("enable_partitionwise_join", "enable_partitionwise_aggregate"),
)


def generate_enable_candidates(
    settings: tuple[tuple[str, ...], ...] = ENABLE_SETTINGS,
) -> list[str]:
    """One candidate per off-by-default planner feature worth switching on."""
    return [
        "/*+ " + " ".join(f"Set({flag} on)" for flag in flags) + " */"
        for flags in settings
    ]


def generate_tuning_candidates(
    settings: tuple[tuple[str, str], ...] = TUNING_SETTINGS,
) -> list[str]:
    """One candidate per resource / cost-model setting worth trying."""
    return [f"/*+ Set({guc} {value}) */" for guc, value in settings]


# -- Cost-model constants ---------------------------------------------------
#
# `TUNING_SETTINGS` varies `random_page_cost` and a parallel worker *ceiling*.
# Neither is what actually decides whether a parallel plan is considered, and
# the manual is direct about it (15.3.5):
#
#     "If a query that is expected to do so does not produce a parallel plan,
#      you can try reducing parallel_setup_cost or parallel_tuple_cost."
#
# `max_parallel_workers_per_gather` is a ceiling; these two are the price the
# planner pays to go near it. And below `min_parallel_table_scan_size` (8MB) it
# will not consider a parallel scan at all, whatever the ceiling says -- which
# is why `Set(max_parallel_workers_per_gather 4)` is a measured no-op on a small
# table and `Parallel(t 4 hard)` is not.
#
# The rest are the cost model itself. Their defaults describe a machine with a
# seek-heavy spinning disk and cheap CPU; this one has 32GB of RAM, an SSD, and
# a database that fits entirely in memory, so every default here is describing
# the wrong machine:
#
#   effective_cache_size  the planner's guess at how much is already cached.
#                         4GB default; the value below is 12GB in 8kB blocks.
#                         Higher makes index scans look cheaper.
#   cpu_tuple_cost        0.01 default. Raising it relative to the page costs
#                         models a machine where CPU, not I/O, is the limit.
#   seq_page_cost         only the *ratio* to random_page_cost matters, so
#                         lowering both says "I/O barely costs anything", which
#                         on a fully-cached database is true.
#
# The manual's warning applies and is worth keeping in view: "there is no
# well-defined method for determining ideal values for the cost variables...
# changing them on the basis of just a few experiments is very risky." Which is
# an argument for measuring them per query -- this project's entire premise --
# rather than for setting them globally on a hunch.
#
# Every setting below was checked against the running server for acceptance;
# see `test_integration_hints.py`.
COST_MODEL_SETTINGS: tuple[tuple[str, str], ...] = (
    ("parallel_setup_cost", "0"),
    ("parallel_tuple_cost", "0.01"),
    ("min_parallel_table_scan_size", "0"),
    ("effective_cache_size", "1572864"),
    ("cpu_tuple_cost", "0.05"),
    ("seq_page_cost", "0.5"),
    # The partner to `join_collapse_limit` in TUNING_SETTINGS. That one flattens
    # explicit JOINs; this one flattens *subqueries*, which is the case a query
    # built from a view actually hits. Lowering it stops the planner pulling a
    # subquery into the parent, trading plan quality for a much smaller search.
    ("from_collapse_limit", "1"),
)

# `plan_cache_mode` is deliberately NOT in the tuple above, and the reason is
# worth writing down because it is not obvious from the GUC list.
#
# It chooses between a generic plan (built once, parameter values ignored,
# reused) and a custom plan (rebuilt per execution). That is the same question
# `app/query_template.py` answers when it gives `city='Mumbai'` and
# `city='London'` a single `query_id` -- both assume one plan serves every
# parameter value -- so it looks like an obvious candidate.
#
# But the manual says it "is considered when a cached plan is to be executed,
# not when it is prepared", and this project plans ad-hoc statements: there is
# no cached plan for it to apply to. Measured: accepted by pg_hint_plan, and the
# cost is unchanged at 7870.2 for both values. A candidate that cannot differ is
# a planning round trip spent to learn nothing -- the same reason
# `enable_partition_pruning` is out of EXTENDED_HINT_SETS.
#
# Where it *would* belong is a diagnostic: if a template's members disagree
# about the best plan, `force_custom_plan` is the fix, and that is a finding
# about the workload rather than a candidate for one query.


def generate_cost_model_candidates(
    settings: tuple[tuple[str, str], ...] = COST_MODEL_SETTINGS,
) -> list[str]:
    """One candidate per cost-model constant worth re-pricing."""
    return [f"/*+ Set({guc} {value}) */" for guc, value in settings]


def generate_hint_sets(hint_sets: tuple[tuple[str, ...], ...] = HINT_SETS) -> list[str]:
    """One hint string per operator combination to disable."""
    return [
        "/*+ " + " ".join(f"Set({flag} off)" for flag in flags) + " */"
        for flags in hint_sets
    ]


def plan_fingerprint(plan: dict) -> tuple:
    """
    A structural identity for a plan, used to drop candidates that came back
    identical to one already seen.

    Without this, most "candidates" for a simple query are byte-identical to
    the native plan -- the optimizer then spends its time choosing between
    copies of the same thing and reports a 0% improvement it never had a
    chance to earn. Deduplicating makes the real size of the action space
    visible instead of hiding it behind a candidate count.
    """
    def walk(node):
        children = tuple(walk(c) for c in node.get("Plans", []))
        return (
            node.get("Node Type"),
            node.get("Relation Name") or node.get("Alias"),
            node.get("Join Type"),
            children,
        )

    return walk(plan.get("raw_plan", plan))
