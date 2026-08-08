"""
The Phase 1 workload: a spread of queries wide/selective enough that join
order (and, per the stretch goal, join *method*) actually changes latency.

Each entry is `{id, sql, description, join_width, selectivity_tag}`.
`join_width` is the number of tables touched (drives how many join-order
permutations `hints.py` has to consider); `selectivity_tag` is a rough label
(high/medium/low = how much of the base table survives the WHERE clause) used
when reporting results, not by the optimizer itself.

Power users (id <= 100) and popular products (id <= 200) get disproportionate
order/order_item volume -- see `data/schema.sql` -- so queries that touch them
deliberately probe the skew.
"""

from __future__ import annotations

WORKLOAD = [
    # -- 2-way joins --------------------------------------------------------
    {
        "id": "2w_country_in",
        "sql": """
            SELECT o.id, u.name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            WHERE u.country = 'IN'
        """,
        "description": "orders x users, single-country filter",
        "join_width": 2,
        "selectivity_tag": "medium",
    },
    {
        "id": "2w_country_narrow",
        "sql": """
            SELECT o.id, u.name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            WHERE u.country = 'SG'
        """,
        "description": "orders x users, narrower single-country filter",
        "join_width": 2,
        "selectivity_tag": "medium",
    },
    {
        "id": "2w_power_users",
        "sql": """
            SELECT o.id, u.name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            WHERE u.id <= 100
        """,
        "description": "orders x users, restricted to skewed power users",
        "join_width": 2,
        "selectivity_tag": "high",
    },
    {
        "id": "2w_items_expensive_products",
        "sql": """
            SELECT oi.id, p.name
            FROM order_items oi
            JOIN products p ON oi.product_id = p.id
            WHERE p.price > 400
        """,
        "description": "order_items x products, high-price filter",
        "join_width": 2,
        "selectivity_tag": "high",
    },
    {
        "id": "2w_items_cheap_products",
        "sql": """
            SELECT oi.id, p.name
            FROM order_items oi
            JOIN products p ON oi.product_id = p.id
            WHERE p.price < 500
        """,
        "description": "order_items x products, broad price filter",
        "join_width": 2,
        "selectivity_tag": "low",
    },
    {
        "id": "2w_items_popular_products",
        "sql": """
            SELECT oi.id, p.name
            FROM order_items oi
            JOIN products p ON oi.product_id = p.id
            WHERE p.id <= 200
        """,
        "description": "order_items x products, restricted to skewed popular products",
        "join_width": 2,
        "selectivity_tag": "high",
    },
    {
        "id": "2w_recent_orders",
        "sql": """
            SELECT o.id, oi.id AS item_id
            FROM orders o
            JOIN order_items oi ON oi.order_id = o.id
            WHERE o.created_at > now() - interval '30 days'
        """,
        "description": "orders x order_items, recent-orders filter",
        "join_width": 2,
        "selectivity_tag": "high",
    },
    {
        "id": "2w_bulk_items",
        "sql": """
            SELECT o.id, oi.id AS item_id
            FROM orders o
            JOIN order_items oi ON oi.order_id = o.id
            WHERE oi.quantity >= 3
        """,
        "description": "orders x order_items, bulk-quantity filter",
        "join_width": 2,
        "selectivity_tag": "medium",
    },
    # -- 3-way joins ----------------------------------------------------------
    {
        "id": "3w_users_orders_items_country",
        "sql": """
            SELECT u.id, o.id, oi.id
            FROM users u
            JOIN orders o ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            WHERE u.country = 'US'
        """,
        "description": "users x orders x order_items, country filter",
        "join_width": 3,
        "selectivity_tag": "medium",
    },
    {
        "id": "3w_users_orders_items_de",
        "sql": """
            SELECT u.id, o.id, oi.id
            FROM users u
            JOIN orders o ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            WHERE u.country = 'DE'
        """,
        "description": "users x orders x order_items, narrower country filter",
        "join_width": 3,
        "selectivity_tag": "medium",
    },
    {
        "id": "3w_orders_items_products_electronics",
        "sql": """
            SELECT o.id, oi.id, p.name
            FROM orders o
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            WHERE p.category = 'electronics'
        """,
        "description": "orders x order_items x products, category filter",
        "join_width": 3,
        "selectivity_tag": "medium",
    },
    {
        "id": "3w_orders_items_products_books",
        "sql": """
            SELECT o.id, oi.id, p.name
            FROM orders o
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            WHERE p.category = 'books'
        """,
        "description": "orders x order_items x products, alt category filter",
        "join_width": 3,
        "selectivity_tag": "medium",
    },
    {
        "id": "3w_power_users_recent",
        "sql": """
            SELECT u.id, o.id, oi.id
            FROM users u
            JOIN orders o ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            WHERE u.id <= 100 AND o.created_at > now() - interval '90 days'
        """,
        "description": "users x orders x order_items, skewed power users + recency",
        "join_width": 3,
        "selectivity_tag": "high",
    },
    {
        "id": "3w_products_items_orders_expensive",
        "sql": """
            SELECT p.id, oi.id, o.id
            FROM products p
            JOIN order_items oi ON oi.product_id = p.id
            JOIN orders o ON o.id = oi.order_id
            WHERE p.price > 300
        """,
        "description": "products x order_items x orders, high-price filter",
        "join_width": 3,
        "selectivity_tag": "medium",
    },
    {
        "id": "3w_users_orders_items_uk_bulk",
        "sql": """
            SELECT u.id, o.id, oi.id
            FROM users u
            JOIN orders o ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            WHERE u.country = 'UK' AND oi.quantity >= 3
        """,
        "description": "users x orders x order_items, country + bulk filter",
        "join_width": 3,
        "selectivity_tag": "high",
    },
    # -- 4-way joins ----------------------------------------------------------
    {
        "id": "4w_full_country_in",
        "sql": """
            SELECT o.id, u.name, p.name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            WHERE u.country = 'IN'
        """,
        "description": "all four tables, single-country filter",
        "join_width": 4,
        "selectivity_tag": "medium",
    },
    {
        "id": "4w_full_country_us_electronics",
        "sql": """
            SELECT o.id, u.name, p.name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            WHERE u.country = 'US' AND p.category = 'electronics'
        """,
        "description": "all four tables, country + category filter",
        "join_width": 4,
        "selectivity_tag": "high",
    },
    {
        "id": "4w_full_power_users_popular_products",
        "sql": """
            SELECT o.id, u.name, p.name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            WHERE u.id <= 100 AND p.id <= 200
        """,
        "description": "all four tables, skewed power users + popular products",
        "join_width": 4,
        "selectivity_tag": "high",
    },
    {
        "id": "4w_full_recent_expensive",
        "sql": """
            SELECT o.id, u.name, p.name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            WHERE o.created_at > now() - interval '60 days' AND p.price > 200
        """,
        "description": "all four tables, recency + price filter",
        "join_width": 4,
        "selectivity_tag": "high",
    },
    {
        "id": "4w_full_broad",
        "sql": """
            SELECT o.id, u.name, p.name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            WHERE p.price < 500
        """,
        "description": "all four tables, near-unselective filter",
        "join_width": 4,
        "selectivity_tag": "low",
    },
    {
        "id": "4w_full_country_de_bulk",
        "sql": """
            SELECT o.id, u.name, p.name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            WHERE u.country = 'DE' AND oi.quantity >= 3
        """,
        "description": "all four tables, country + bulk-quantity filter",
        "join_width": 4,
        "selectivity_tag": "high",
    },
    {
        "id": "4w_full_sg_books",
        "sql": """
            SELECT o.id, u.name, p.name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            WHERE u.country = 'SG' AND p.category = 'books'
        """,
        "description": "all four tables, narrow country + category filter",
        "join_width": 4,
        "selectivity_tag": "high",
    },
    {
        "id": "4w_full_clothing_recent",
        "sql": """
            SELECT o.id, u.name, p.name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            WHERE p.category = 'clothing' AND o.created_at > now() - interval '180 days'
        """,
        "description": "all four tables, category + wide recency filter",
        "join_width": 4,
        "selectivity_tag": "medium",
    },
    {
        "id": "4w_full_home_power_users",
        "sql": """
            SELECT o.id, u.name, p.name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            WHERE p.category = 'home' AND u.id <= 100
        """,
        "description": "all four tables, category + power-user filter",
        "join_width": 4,
        "selectivity_tag": "high",
    },
    {
        "id": "4w_full_sports_broad_price",
        "sql": """
            SELECT o.id, u.name, p.name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            WHERE p.category = 'sports' AND p.price BETWEEN 5 AND 500
        """,
        "description": "all four tables, category + full price range (near no-op filter)",
        "join_width": 4,
        "selectivity_tag": "low",
    },
]
