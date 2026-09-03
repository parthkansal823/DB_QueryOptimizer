"""
The benchmark workload, v2 -- written to make PostgreSQL's optimizer wrong.

v1 used independent predicates on a 4-table schema. Measured result: an
oracle ceiling of ~6.5%, only 7 of 25 queries with any real alternative, and
no model able to pick the best plan even once. PostgreSQL was right almost
every time, so there was nothing to learn.

v2 targets the specific failure mode Leis et al. identified -- the
**independence assumption**. PostgreSQL estimates `WHERE a = x AND b = y` as
sel(a) * sel(b), which is only right when the columns are unrelated. The v2
schema (`data/schema.sql`) builds in functional dependencies:

    city    -> country     (Mumbai implies IN)
    brand   -> category    (Voltix implies electronics)
    price_band ~ category  (electronics skew premium)
    channel ~ status       (partner channel carries the cancellations)

so a query filtering on *both* halves of a dependency gets a row estimate
several times too small. That error compounds through joins and produces
genuinely wrong join orders -- which is precisely the case a learned
optimizer should be able to beat.

Each entry records `trap`, naming which estimation error it exercises, so a
result can be read against the mechanism rather than just a number. The
schema is now 6 tables, so the workload reaches 5- and 6-way joins where
ordering matters far more than at 2 or 3.

## v3: shapes, not only estimates

The 24 queries above are all the same *shape* -- a select over joins with
filters. None aggregates, sorts, de-duplicates, or uses a subquery, so no
training plan ever contained a `HashAggregate`, a `WindowAgg`, a `Sort` above a
join, or a semi-join.

That is a hole in the training data, and it is visible in the feature vector:
`optimizer/plan_tree.py` counts `n_aggregate`, `n_sort` and `n_materialize`, and
those columns were near-zero in every row the model had ever seen. A feature
that never varies teaches nothing, and a plan shape that never appears in
training is one the model is guessing about at serving time.

`app.generalisation` measured the consequence: prediction error of **2.93x on
trained queries and 3.25x on held-out ones** -- the model generalises (ratio
1.11) and is uniformly inaccurate. Coverage of the *plan space*, not just more
rows of the same shape, is the thing that has not been tried.

So the `shape_` group below adds aggregation, DISTINCT, top-N sorting, a window
function, and semi- and anti-joins. Every one was measured before being added:
native latency against the best plan the served action space can actually reach,
three executions each, median. Two candidates were rejected -- a top-N showing
39% headroom on a 1.7 ms query (a percentage of nothing) and a five-way join
that linked `suppliers` to `users` on `country`, which ran 1.77 s at 0% headroom
because it is a near-cartesian join with no key path rather than a hard query.
"""

from __future__ import annotations

WORKLOAD = [
    # -- correlation traps: both halves of a functional dependency ----------
    {
        "id": "corr_city_country",
        "sql": """
            SELECT o.id, u.name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            WHERE u.country = 'IN' AND u.city = 'Mumbai'
        """,
        "description": "city implies country; estimate is ~5x too small",
        "join_width": 2,
        "trap": "city->country",
    },
    {
        "id": "corr_city_country_lang",
        "sql": """
            SELECT o.id, u.name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            WHERE u.country = 'DE' AND u.city = 'Berlin' AND u.language = 'de'
        """,
        "description": "three mutually-dependent columns; error compounds",
        "join_width": 2,
        "trap": "city->country->language",
    },
    {
        "id": "corr_brand_category",
        "sql": """
            SELECT oi.id, p.name
            FROM order_items oi
            JOIN products p ON oi.product_id = p.id
            WHERE p.category = 'electronics' AND p.brand = 'Voltix'
        """,
        "description": "brand implies category",
        "join_width": 2,
        "trap": "brand->category",
    },
    {
        "id": "corr_brand_band",
        "sql": """
            SELECT oi.id, p.name
            FROM order_items oi
            JOIN products p ON oi.product_id = p.id
            WHERE p.brand = 'Peakline' AND p.price_band = 'premium' AND p.category = 'sports'
        """,
        "description": "brand, band and category are all one fact",
        "join_width": 2,
        "trap": "brand->category+band",
    },
    {
        "id": "corr_status_channel",
        "sql": """
            SELECT o.id, u.name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            WHERE o.status = 'cancelled' AND o.channel = 'partner'
        """,
        "description": "cancellations cluster in the partner channel",
        "join_width": 2,
        "trap": "status~channel",
    },
    # -- 3-way, correlated ---------------------------------------------------
    {
        "id": "corr_3w_city_electronics",
        "sql": """
            SELECT u.id, o.id, oi.id
            FROM users u
            JOIN orders o ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            WHERE u.country = 'US' AND u.city = 'Austin'
        """,
        "description": "correlated user filter propagated through two joins",
        "join_width": 3,
        "trap": "city->country",
    },
    {
        "id": "corr_3w_brand_chain",
        "sql": """
            SELECT o.id, oi.id, p.name
            FROM orders o
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            WHERE p.category = 'books' AND p.brand = 'PagePress'
        """,
        "description": "correlated product filter through two joins",
        "join_width": 3,
        "trap": "brand->category",
    },
    {
        "id": "corr_3w_supplier_chain",
        "sql": """
            SELECT p.id, ps.id, s.name
            FROM products p
            JOIN product_suppliers ps ON ps.product_id = p.id
            JOIN suppliers s ON s.id = ps.supplier_id
            WHERE p.price_band = 'premium' AND p.category = 'electronics'
        """,
        "description": "price_band correlates with category, over the supplier chain",
        "join_width": 3,
        "trap": "band~category",
    },
    {
        "id": "corr_3w_both_ends",
        "sql": """
            SELECT u.id, o.id, oi.id
            FROM users u
            JOIN orders o ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            WHERE u.city = 'Delhi' AND u.country = 'IN' AND o.status = 'completed' AND o.channel = 'web'
        """,
        "description": "two independent correlation traps in one query",
        "join_width": 3,
        "trap": "city->country + status~channel",
    },
    # -- 4-way ---------------------------------------------------------------
    {
        "id": "corr_4w_full_correlated",
        "sql": """
            SELECT o.id, u.name, p.name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            WHERE u.country = 'IN' AND u.city = 'Bangalore'
              AND p.category = 'electronics' AND p.brand = 'Nexon'
        """,
        "description": "correlated filters at both ends of a 4-way join",
        "join_width": 4,
        "trap": "city->country + brand->category",
    },
    {
        "id": "corr_4w_premium_us",
        "sql": """
            SELECT o.id, u.name, p.name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            WHERE u.city = 'Seattle' AND u.language = 'en' AND p.price_band = 'premium'
        """,
        "description": "correlated user columns plus a category-correlated band",
        "join_width": 4,
        "trap": "city->language + band~category",
    },
    {
        "id": "corr_4w_cancelled_premium",
        "sql": """
            SELECT o.id, u.name, p.name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            WHERE o.status = 'cancelled' AND o.channel = 'partner' AND p.category = 'sports'
        """,
        "description": "rare correlated order state joined to a category filter",
        "join_width": 4,
        "trap": "status~channel",
    },
    {
        "id": "corr_4w_power_users",
        "sql": """
            SELECT o.id, u.name, p.name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            WHERE u.id <= 100 AND p.brand = 'Voltix' AND p.category = 'electronics'
        """,
        "description": "skewed power users plus a correlated product filter",
        "join_width": 4,
        "trap": "skew + brand->category",
    },
    {
        "id": "corr_4w_broad",
        "sql": """
            SELECT o.id, u.name, p.name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            WHERE p.price < 900
        """,
        "description": "near-unselective control: no correlation to exploit",
        "join_width": 4,
        "trap": "none (control)",
    },
    # -- 5- and 6-way: where join ordering really bites ----------------------
    {
        "id": "corr_5w_supplier_chain",
        "sql": """
            SELECT o.id, u.name, p.name, ps.lead_days
            FROM orders o
            JOIN users u ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            JOIN product_suppliers ps ON ps.product_id = p.id
            WHERE u.country = 'IN' AND u.city = 'Mumbai' AND p.category = 'electronics'
        """,
        "description": "5-way with a correlated user filter",
        "join_width": 5,
        "trap": "city->country",
    },
    {
        "id": "corr_5w_premium_supplier",
        "sql": """
            SELECT o.id, p.name, ps.id
            FROM orders o
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            JOIN product_suppliers ps ON ps.product_id = p.id
            JOIN suppliers s ON s.id = ps.supplier_id
            WHERE p.brand = 'Wovenly' AND p.category = 'clothing' AND s.rating >= 4
        """,
        "description": "5-way through the supplier chain, correlated product filter",
        "join_width": 5,
        "trap": "brand->category",
    },
    {
        "id": "corr_6w_everything",
        "sql": """
            SELECT o.id, u.name, p.name, s.name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            JOIN product_suppliers ps ON ps.product_id = p.id
            JOIN suppliers s ON s.id = ps.supplier_id
            WHERE u.city = 'London' AND u.country = 'UK'
              AND p.category = 'home' AND p.brand = 'Hearthware'
        """,
        "description": "all six tables, correlated filters at both ends",
        "join_width": 6,
        "trap": "city->country + brand->category",
    },
    {
        "id": "corr_6w_status_premium",
        "sql": """
            SELECT o.id, u.name, p.name, s.name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            JOIN product_suppliers ps ON ps.product_id = p.id
            JOIN suppliers s ON s.id = ps.supplier_id
            WHERE o.status = 'shipped' AND o.channel = 'mobile' AND p.price_band = 'premium'
        """,
        "description": "all six tables via a correlated order-state filter",
        "join_width": 6,
        "trap": "status~channel + band~category",
    },
    # -- uncorrelated controls, to show the contrast -------------------------
    {
        "id": "ctl_country_only",
        "sql": """
            SELECT o.id, u.name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            WHERE u.country = 'US'
        """,
        "description": "control: single-column filter, nothing to mis-estimate",
        "join_width": 2,
        "trap": "none (control)",
    },
    {
        "id": "ctl_category_only",
        "sql": """
            SELECT oi.id, p.name
            FROM order_items oi
            JOIN products p ON oi.product_id = p.id
            WHERE p.category = 'clothing'
        """,
        "description": "control: single-column product filter",
        "join_width": 2,
        "trap": "none (control)",
    },
    {
        "id": "ctl_3w_recent",
        "sql": """
            SELECT u.id, o.id, oi.id
            FROM users u
            JOIN orders o ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            WHERE o.created_at > now() - interval '30 days'
        """,
        "description": "control: recency filter, no correlation",
        "join_width": 3,
        "trap": "none (control)",
    },
    {
        "id": "ctl_4w_quantity",
        "sql": """
            SELECT o.id, u.name, p.name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            WHERE oi.quantity >= 3
        """,
        "description": "control: 4-way with a single uncorrelated filter",
        "join_width": 4,
        "trap": "none (control)",
    },
    {
        "id": "ctl_5w_signup",
        "sql": """
            SELECT o.id, u.name, p.name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            JOIN product_suppliers ps ON ps.product_id = p.id
            WHERE u.signup_year = 2021
        """,
        "description": "control: 5-way with a single uncorrelated filter",
        "join_width": 5,
        "trap": "none (control)",
    },
    {
        "id": "ctl_supplier_rating",
        "sql": """
            SELECT ps.id, s.name
            FROM product_suppliers ps
            JOIN suppliers s ON s.id = ps.supplier_id
            WHERE s.rating = 5 AND s.country = 'IN'
        """,
        "description": "control: supplier filter, weak correlation",
        "join_width": 2,
        "trap": "none (control)",
    },

    # -- v3: plan shapes the queries above never produce ------------------
    {
        "id": "shape_6w_supplier_reach",
        "sql": """
            SELECT s.name, count(DISTINCT o.id) AS orders
            FROM orders o
            JOIN users u ON u.id = o.user_id
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            JOIN product_suppliers ps ON ps.product_id = p.id
            JOIN suppliers s ON s.id = ps.supplier_id
            WHERE p.category = 'electronics' AND p.brand = 'Voltix'
            GROUP BY s.name
            ORDER BY orders DESC
            LIMIT 10
        """,
        "description": "6-way join under a grouped aggregate; 1754 ms native, 48% headroom",
        "join_width": 6,
        "trap": "brand->category, under aggregation",
    },
    {
        "id": "shape_distinct_brands",
        "sql": """
            SELECT DISTINCT p.brand, p.category
            FROM orders o
            JOIN users u ON u.id = o.user_id
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            WHERE u.city = 'Mumbai' AND u.country = 'IN'
        """,
        "description": "DISTINCT forces a dedup above a 4-way join; 28% headroom",
        "join_width": 4,
        "trap": "city->country, under DISTINCT",
    },
    {
        "id": "shape_group_by_category",
        "sql": """
            SELECT p.category, count(*) AS orders, sum(oi.quantity) AS units
            FROM order_items oi
            JOIN products p ON p.id = oi.product_id
            JOIN orders o ON o.id = oi.order_id
            WHERE p.brand = 'Voltix' AND p.category = 'electronics'
            GROUP BY p.category
        """,
        "description": "grouped aggregate over a correlated filter; 21% headroom",
        "join_width": 3,
        "trap": "brand->category, under GROUP BY",
    },
    {
        "id": "shape_top_n_orders",
        "sql": """
            SELECT o.id, u.name, sum(oi.quantity) AS units
            FROM orders o
            JOIN users u ON u.id = o.user_id
            JOIN order_items oi ON oi.order_id = o.id
            WHERE u.country = 'IN' AND u.city = 'Mumbai'
            GROUP BY o.id, u.name
            ORDER BY units DESC
            LIMIT 20
        """,
        "description": "top-N: a Sort above a join, where startup cost dominates; 20% headroom",
        "join_width": 3,
        "trap": "city->country, under ORDER BY / LIMIT",
    },
    {
        "id": "shape_semi_join_exists",
        "sql": """
            SELECT u.id, u.name
            FROM users u
            WHERE u.city = 'Mumbai' AND u.country = 'IN'
              AND EXISTS (
                  SELECT 1 FROM orders o
                  JOIN order_items oi ON oi.order_id = o.id
                  JOIN products p ON p.id = oi.product_id
                  WHERE o.user_id = u.id AND p.price_band = 'premium'
              )
        """,
        "description": "semi-join: EXISTS stops at the first match, so row estimates behave differently",
        "join_width": 4,
        "trap": "city->country, under a semi-join",
    },
    {
        "id": "shape_window_rank",
        "sql": """
            SELECT p.category, p.name, sum(oi.quantity) AS units,
                   rank() OVER (PARTITION BY p.category ORDER BY sum(oi.quantity) DESC) AS rk
            FROM order_items oi
            JOIN products p ON p.id = oi.product_id
            WHERE p.price_band = 'premium' AND p.category = 'home'
            GROUP BY p.category, p.name
        """,
        "description": "window function over a grouped aggregate: a WindowAgg no other query produces",
        "join_width": 2,
        "trap": "price_band~category, under a window function",
    },
    {
        "id": "shape_anti_join_unsold",
        "sql": """
            SELECT p.id, p.name
            FROM products p
            WHERE p.category = 'electronics' AND p.brand = 'Voltix'
              AND NOT EXISTS (
                  SELECT 1 FROM order_items oi WHERE oi.product_id = p.id
              )
        """,
        "description": "anti-join: NOT EXISTS, the one shape whose output shrinks as its input grows",
        "join_width": 2,
        "trap": "brand->category, under an anti-join",
    },
    {
        "id": "shape_range_recent",
        "sql": """
            SELECT o.id, u.name, u.language
            FROM orders o
            JOIN users u ON u.id = o.user_id
            WHERE u.city = 'London' AND u.country = 'UK' AND u.language = 'en'
              AND o.created_at >= now() - interval '90 days'
        """,
        "description": "three correlated columns plus a range predicate on a naturally-ordered column",
        "join_width": 2,
        "trap": "city->country->language, plus a time range",
    },
]
