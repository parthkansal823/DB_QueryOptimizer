-- Extended statistics for the correlated columns this schema is built around.
--
-- ============================================================================
-- Running this makes the learned optimizer nearly useless on this database.
-- That is the point of shipping it, and it is why it is not in schema.sql.
-- ============================================================================
--
-- PostgreSQL assumes filters are independent. `WHERE city = 'Mumbai' AND
-- country = 'IN'` is estimated as P(city) x P(country), and since almost every
-- Mumbai row is already IN, the real selectivity is roughly P(city) alone. The
-- estimate comes out an order of magnitude too small, the row counts multiply
-- wrong across joins, and the planner picks a nested loop where a hash join
-- belongs. `data/schema.sql` seeds exactly that correlation on purpose: it is
-- the failure mode Leis et al. (VLDB 2015) identify as the dominant one, and
-- the headroom this project's whole action space exists to recover.
--
-- `CREATE STATISTICS ... (dependencies)` fixes it at the source. Measured in
-- docs/WRITEUP.md 2.57, three rounds alternating on and off:
--
--     statistics OFF    native 602.0 ms   oracle 481.4 ms   headroom 18.6%
--     statistics ON     native 595.9 ms   oracle 567.4 ms   headroom  0.4%
--
-- Ranges disjoint. **98% of the opportunity, gone.** Note where it went: native
-- is unchanged. PostgreSQL's own plan was already close to the best available;
-- what changed is that the good plans stopped being reachable only by hint,
-- because the planner now finds them itself.
--
-- ## So why is this a file rather than a line in schema.sql
--
-- Because they answer different questions.
--
-- If you have a real database with correlated columns, run this. Three
-- statements and an ANALYZE beat any plan-picking layer on cost, coverage and
-- risk: they fix every query against those columns forever, including ones this
-- optimizer never sees, and they cannot regress a query the way a wrong hint
-- can. Any honest recommendation puts this first. `app.advisor` will suggest
-- them for your schema; this file is the answer for the bundled one.
--
-- If you are here to see whether a learned plan-picker works, the demo needs a
-- database where PostgreSQL is *wrong*, because a layer that corrects the
-- optimizer cannot be evaluated on a workload the optimizer already gets right.
-- Creating these by default would leave 0.4% of headroom and a benchmark
-- measuring noise.
--
-- The dishonest version of this project ships schema.sql without the statistics
-- and never mentions them. This is the same code with the trade-off written
-- down.
--
-- ## After running it
--
-- Every number in README.md and docs/ was measured without these. Re-collect
-- and re-train, and expect the headline capture figure to **fall** -- that is
-- the metric working. `app.benchmark` reports headroom alongside capture for
-- exactly this reason: capture is a fraction of an opportunity, and this file
-- removes the opportunity.
--
--     psql "$DATABASE_URL" -f data/extended_stats.sql
--     python -m app.collect_data && python -m app.train --model-out models/_c.pkl
--
-- To undo: DROP STATISTICS "lqo_stx_products_brand_category", etc.

CREATE STATISTICS IF NOT EXISTS "lqo_stx_products_brand_category"
    (dependencies) ON "brand", "category" FROM "products";

CREATE STATISTICS IF NOT EXISTS "lqo_stx_users_city_language"
    (dependencies) ON "city", "language" FROM "users";

CREATE STATISTICS IF NOT EXISTS "lqo_stx_users_city_country"
    (dependencies) ON "city", "country" FROM "users";

-- Required. CREATE STATISTICS only registers the request; the numbers arrive
-- with the next ANALYZE, and until then nothing has changed. 2.57 records this
-- as its own finding, because a run that skips it measures the "before" arm
-- twice and concludes extended statistics do nothing.
ANALYZE "products";
ANALYZE "users";
