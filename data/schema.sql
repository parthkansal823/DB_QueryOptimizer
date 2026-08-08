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
INSERT INTO users (name, country, city, language, signup_year)
SELECT
    'user_' || i,
    c.country,
    c.city,
    c.language,
    2018 + (random() * 6)::int
FROM generate_series(1, 50000) AS i
CROSS JOIN LATERAL (
    SELECT country, city, language
    FROM (VALUES
        ('IN','Mumbai','hi'), ('IN','Delhi','hi'), ('IN','Bangalore','en'),
        ('US','New York','en'), ('US','Austin','en'), ('US','Seattle','en'),
        ('UK','London','en'), ('UK','Manchester','en'),
        ('DE','Berlin','de'), ('DE','Munich','de'),
        ('SG','Singapore','en')
    ) AS v(country, city, language)
    -- Skewed city popularity: the first rows are far more common, so no
    -- single city is a uniform 1/11 slice.
    ORDER BY random() * (CASE WHEN random() < 0.5 THEN 0.2 ELSE 1.0 END)
    LIMIT 1
) AS c;

INSERT INTO suppliers (name, country, rating)
SELECT 'supplier_' || i,
       (ARRAY['IN','US','UK','DE','SG','CN'])[1 + floor(random() * 6)::int],
       1 + floor(random() * 5)::int
FROM generate_series(1, 2000) AS i;

-- ------------------------------------------------------------- products --
-- brand -> category is strict, and price follows the category's band.
INSERT INTO products (name, category, brand, price, price_band)
SELECT
    'product_' || i,
    b.category,
    b.brand,
    round((b.base_price * (0.6 + random() * 0.8))::numeric, 2),
    b.band
FROM generate_series(1, 20000) AS i
CROSS JOIN LATERAL (
    SELECT category, brand, base_price, band
    FROM (VALUES
        ('electronics','Voltix',   800.0, 'premium'),
        ('electronics','Nexon',    450.0, 'mid'),
        ('books','PagePress',       25.0, 'budget'),
        ('books','InkHouse',        40.0, 'budget'),
        ('clothing','Threadly',     70.0, 'mid'),
        ('clothing','Wovenly',     180.0, 'premium'),
        ('home','Hearthware',      120.0, 'mid'),
        ('home','Nestly',           35.0, 'budget'),
        ('sports','Peakline',      260.0, 'premium'),
        ('sports','Trailmark',      90.0, 'mid')
    ) AS v(category, brand, base_price, band)
    ORDER BY random()
    LIMIT 1
) AS b;

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
    s.status,
    s.channel
FROM generate_series(1, 200000) AS i
CROSS JOIN LATERAL (
    SELECT status, channel
    FROM (VALUES
        ('completed','web'), ('completed','web'), ('completed','mobile'),
        ('shipped','web'), ('shipped','mobile'),
        ('cancelled','partner'), ('refunded','partner')
    ) AS v(status, channel)
    ORDER BY random()
    LIMIT 1
) AS s;

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
