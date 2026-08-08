-- Synthetic e-commerce schema sized so join order actually matters.
-- Cardinalities are intentionally skewed (few products, many order_items)
-- so a naive left-to-right join order is NOT always the fastest one --
-- that gap is exactly what the learned optimizer is trying to close.

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT NOT NULL
);

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price NUMERIC(10, 2) NOT NULL
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL
);

-- Seed data. Sized to be big enough that scan/join choice has a real
-- latency cost, but small enough to seed quickly on a laptop.
INSERT INTO users (name, country)
SELECT 'user_' || i, (ARRAY['IN', 'US', 'UK', 'DE', 'SG'])[1 + floor(random() * 5)::int]
FROM generate_series(1, 50000) AS i;

INSERT INTO products (name, category, price)
SELECT 'product_' || i,
       (ARRAY['electronics', 'books', 'clothing', 'home', 'sports'])[1 + floor(random() * 5)::int],
       round((random() * 500 + 5)::numeric, 2)
FROM generate_series(1, 5000) AS i;

INSERT INTO orders (user_id, created_at)
SELECT 1 + floor(random() * 50000)::int,
       now() - (random() * interval '365 days')
FROM generate_series(1, 200000) AS i;

INSERT INTO order_items (order_id, product_id, quantity)
SELECT 1 + floor(random() * 200000)::int,
       1 + floor(random() * 5000)::int,
       1 + floor(random() * 4)::int
FROM generate_series(1, 500000) AS i;

-- No indexes on the foreign keys yet -- on purpose. This makes join order
-- and join method choice actually consequential. Once your benchmark is
-- working, try adding indexes as a controlled experiment: the learned
-- optimizer's advantage should shrink once Postgres has good indexes to
-- lean on. That comparison is worth a paragraph in your evaluation section.
