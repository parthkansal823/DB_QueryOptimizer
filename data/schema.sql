-- Synthetic e-commerce schema, v2 -- rebuilt to be *hard for the optimizer*.
--
-- ## Why v1 was replaced
--
-- The first version used independent, uniformly-random columns. Measurement
-- showed the consequence: across its 25-query workload the oracle ceiling was
-- only ~6.5%, just 7 of 25 queries had any plan beating PostgreSQL by >5%,
-- and no model class could pick the best plan even once. PostgreSQL's
-- optimizer was simply right almost every time, so there was nothing for a
-- learned optimizer to learn. A benchmark where the baseline is already
-- optimal cannot measure an improvement.
--
-- ## What makes an optimizer fail
--
-- Leis et al. (VLDB 2015) showed plans go wrong because *cardinality
-- estimates* go wrong, and the biggest single cause is PostgreSQL's
-- **independence assumption**: given `WHERE a = 1 AND b = 2` it multiplies
-- the two selectivities, which is only correct when the columns are
-- unrelated. So v2 builds in deliberate correlations:
--
--   * `city` functionally determines `country` (Mumbai is always IN). A
--     filter on both is estimated as sel(city) x sel(country) but really
--     costs sel(city) -- an underestimate of roughly 5x here.
--   * `language` correlates with `country` the same way.
--   * `brand` functionally determines `category`.
--   * `price_band` correlates with `category` (electronics skew expensive).
--
-- Each of these makes PostgreSQL underestimate row counts, which propagates
-- multiplicatively through joins and produces genuinely wrong join orders --
-- which is exactly the situation a learned optimizer exists to fix.
--
-- Two more tables (suppliers, product_suppliers) widen the join graph to 6
-- tables, so the workload can reach 5- and 6-way joins where ordering
-- matters far more than it does at 2 or 3.

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT NOT NULL,
    city TEXT NOT NULL,      -- functionally determines country
    language TEXT NOT NULL,  -- correlated with country
    signup_year INTEGER NOT NULL
);

CREATE TABLE suppliers (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT NOT NULL,
    rating INTEGER NOT NULL
);

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    brand TEXT NOT NULL,       -- functionally determines category
    price NUMERIC(10, 2) NOT NULL,
    price_band TEXT NOT NULL   -- correlated with category
);

CREATE TABLE product_suppliers (
    id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES products(id),
    supplier_id INTEGER REFERENCES suppliers(id),
    lead_days INTEGER NOT NULL
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP NOT NULL DEFAULT now(),
    status TEXT NOT NULL,
    channel TEXT NOT NULL
);

CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL
);

-- ---------------------------------------------------------------- users --
-- city -> country is a strict functional dependency, and language follows
-- country. PostgreSQL has no multi-column statistics here by default, so
-- filtering on two of the three reads as far more selective than it is.
-- NOTE ON THE `pick` SUBQUERIES BELOW
--
-- Each of the three inserts that build a correlated pair used to draw its
-- values like this:
--
--     CROSS JOIN LATERAL (
--         SELECT ... FROM (VALUES (...)) AS v ORDER BY random() LIMIT 1
--     ) AS c
--
-- which reads as "pick a random row for each generated row" and is not. The
-- subquery references nothing from the outer row, so PostgreSQL is free to
-- evaluate it **once** and reuse the result for all of them -- and does.
--
-- Every user got the same city, every product the same brand, every order the
-- same status. `SELECT count(DISTINCT city) FROM users` returned **1** across
-- 50,000 rows. So the correlated columns this schema exists to provide had no
-- correlation in them, because they had no variety at all: a filter on `city`
-- either matched every row or none.
--
-- That is not a cosmetic problem. It is the premise of the whole project --
-- "PostgreSQL assumes filters are unrelated and multiplies their odds
-- together" needs two columns that vary and covary. It also explains the
-- dashboard reporting estimates like "expected 75 rows and got 0": the
-- generated workload filters on `city = 'Mumbai'`, and no such row existed.
--
-- The form below computes an index per generated row in a plain subquery, then
-- subscripts parallel arrays with it. One index shared across the columns is
-- what keeps them correlated; computing it per row is what gives them variety.

INSERT INTO users (name, country, city, language, signup_year)
SELECT
    'user_' || pick.i,
    (ARRAY['IN','IN','IN','US','US','US','UK','UK','DE','DE','SG'])[pick.idx],
    (ARRAY['Mumbai','Delhi','Bangalore','New York','Austin','Seattle',
           'London','Manchester','Berlin','Munich','Singapore'])[pick.idx],
    (ARRAY['hi','hi','en','en','en','en','en','en','de','de','en'])[pick.idx],
    2018 + (random() * 6)::int
FROM (
    -- `power(random(), 2)` skews toward the low indices, so no single city is a
    -- uniform 1/11 slice -- the same skew the previous ORDER BY intended.
    SELECT i, 1 + floor(power(random(), 2.0) * 11)::int AS idx
    FROM generate_series(1, 50000) AS i
) AS pick;

INSERT INTO suppliers (name, country, rating)
SELECT 'supplier_' || i,
       (ARRAY['IN','US','UK','DE','SG','CN'])[1 + floor(random() * 6)::int],
       1 + floor(random() * 5)::int
FROM generate_series(1, 2000) AS i;

-- ------------------------------------------------------------- products --
-- brand -> category is strict, and price follows the category's band.
INSERT INTO products (name, category, brand, price, price_band)
SELECT
    'product_' || pick.i,
    (ARRAY['electronics','electronics','books','books','clothing',
           'clothing','home','home','sports','sports'])[pick.idx],
    (ARRAY['Voltix','Nexon','PagePress','InkHouse','Threadly',
           'Wovenly','Hearthware','Nestly','Peakline','Trailmark'])[pick.idx],
    round(((ARRAY[800.0, 450.0, 25.0, 40.0, 70.0,
                  180.0, 120.0, 35.0, 260.0, 90.0])[pick.idx]
           * (0.6 + random() * 0.8))::numeric, 2),
    (ARRAY['premium','mid','budget','budget','mid',
           'premium','mid','budget','premium','mid'])[pick.idx]
FROM (
    SELECT i, 1 + floor(random() * 10)::int AS idx
    FROM generate_series(1, 20000) AS i
) AS pick;

INSERT INTO product_suppliers (product_id, supplier_id, lead_days)
SELECT 1 + floor(random() * 20000)::int,
       1 + floor(random() * 2000)::int,
       1 + floor(random() * 30)::int
FROM generate_series(1, 60000) AS i;

-- --------------------------------------------------------------- orders --
-- Power users: 100 users place 40% of all orders. status and channel
-- correlate (cancelled orders cluster in one channel), giving another
-- independence-assumption trap.
INSERT INTO orders (user_id, created_at, status, channel)
SELECT
    CASE WHEN random() < 0.4
         THEN 1 + floor(random() * 100)::int
         ELSE 101 + floor(random() * 49900)::int END,
    now() - (random() * interval '365 days'),
    -- cancelled and refunded cluster in 'partner', which is the trap: a filter
    -- on both status and channel is estimated as the product of two
    -- independent selectivities and is really far narrower.
    (ARRAY['completed','completed','completed','shipped','shipped',
           'cancelled','refunded'])[pick.idx],
    (ARRAY['web','web','mobile','web','mobile',
           'partner','partner'])[pick.idx]
FROM (
    SELECT 1 + floor(random() * 7)::int AS idx
    FROM generate_series(1, 200000) AS i
) AS pick;

-- Popular products: 200 products take 50% of all line items.
INSERT INTO order_items (order_id, product_id, quantity)
SELECT
    1 + floor(random() * 200000)::int,
    CASE WHEN random() < 0.5
         THEN 1 + floor(random() * 200)::int
         ELSE 201 + floor(random() * 19800)::int END,
    1 + floor(random() * 4)::int
FROM generate_series(1, 500000) AS i;

-- No indexes on the foreign keys, on purpose: it keeps join order and join
-- method consequential. Adding them is a worthwhile controlled experiment --
-- the learned optimizer's advantage should shrink once PostgreSQL has good
-- access paths to lean on.
--
-- Note there are deliberately NO `CREATE STATISTICS` objects either. Multi-
-- column statistics are precisely how a DBA would *fix* the correlations
-- above; leaving them off is what preserves the estimation errors this
-- benchmark exists to exploit. Adding them is the other half of that
-- experiment: it should shrink the headroom sharply.
ANALYZE;
