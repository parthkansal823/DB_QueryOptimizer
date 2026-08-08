-- TPC-H data generation, in pure SQL.
--
-- The official generator (dbgen) is a C program that has to be fetched and
-- compiled; this reproduces its table shapes, cardinality ratios and value
-- distributions with `generate_series`, so a usable TPC-H database exists
-- after one psql run with nothing to download.
--
-- Cardinalities follow the spec's ratios for scale factor :sf --
--   supplier  10k * sf      part      200k * sf
--   customer  150k * sf     orders    1.5M * sf
--   partsupp  800k * sf     lineitem  ~6M * sf (1-7 lines per order)
-- -- so relative table sizes, which are what drive join-order decisions,
-- match the real benchmark even though the literal strings do not.
--
-- Deliberate skew, because a uniform join key makes join order irrelevant
-- (the lesson from data/schema.sql): 20% of customers place 60% of orders,
-- and 15% of parts appear in 50% of line items.

\set sf :sf

INSERT INTO region (r_regionkey, r_name, r_comment)
SELECT i, (ARRAY['AFRICA','AMERICA','ASIA','EUROPE','MIDDLE EAST'])[i + 1], 'region ' || i
FROM generate_series(0, 4) AS i;

INSERT INTO nation (n_nationkey, n_name, n_regionkey, n_comment)
SELECT i, 'NATION_' || i, i % 5, 'nation ' || i
FROM generate_series(0, 24) AS i;

INSERT INTO supplier (s_suppkey, s_name, s_address, s_nationkey, s_phone, s_acctbal, s_comment)
SELECT i,
       'Supplier#' || lpad(i::text, 9, '0'),
       'address ' || i,
       (random() * 24)::int,
       '10-' || lpad((random() * 999)::int::text, 3, '0') || '-' || lpad((random() * 999)::int::text, 3, '0'),
       round((random() * 10000 - 1000)::numeric, 2),
       'supplier comment ' || i
FROM generate_series(1, (10000 * :sf)::int) AS i;

INSERT INTO part (p_partkey, p_name, p_mfgr, p_brand, p_type, p_size, p_container, p_retailprice, p_comment)
SELECT i,
       'part ' || i,
       'Manufacturer#' || (1 + (random() * 4)::int),
       'Brand#' || (10 + (random() * 4)::int) || (1 + (random() * 4)::int),
       (ARRAY['STANDARD','SMALL','MEDIUM','LARGE','ECONOMY','PROMO'])[1 + (random() * 5)::int]
         || ' ' || (ARRAY['ANODIZED','BURNISHED','PLATED','POLISHED','BRUSHED'])[1 + (random() * 4)::int]
         || ' ' || (ARRAY['TIN','NICKEL','BRASS','STEEL','COPPER'])[1 + (random() * 4)::int],
       1 + (random() * 49)::int,
       (ARRAY['SM CASE','LG BOX','MED BAG','JUMBO PKG','WRAP CAN'])[1 + (random() * 4)::int],
       round((900 + random() * 1100)::numeric, 2),
       'part comment ' || i
FROM generate_series(1, (200000 * :sf)::int) AS i;

INSERT INTO partsupp (ps_partkey, ps_suppkey, ps_availqty, ps_supplycost, ps_comment)
SELECT p.i,
       1 + ((p.i * 7 + s.j * 13) % (10000 * :sf)::int),
       (random() * 9999)::int,
       round((random() * 1000)::numeric, 2),
       'partsupp comment'
FROM generate_series(1, (200000 * :sf)::int) AS p(i),
     generate_series(1, 4) AS s(j);

INSERT INTO customer (c_custkey, c_name, c_address, c_nationkey, c_phone, c_acctbal, c_mktsegment, c_comment)
SELECT i,
       'Customer#' || lpad(i::text, 9, '0'),
       'address ' || i,
       (random() * 24)::int,
       '25-' || lpad((random() * 999)::int::text, 3, '0') || '-' || lpad((random() * 999)::int::text, 3, '0'),
       round((random() * 10000 - 1000)::numeric, 2),
       (ARRAY['BUILDING','AUTOMOBILE','MACHINERY','HOUSEHOLD','FURNITURE'])[1 + (random() * 4)::int],
       'customer comment ' || i
FROM generate_series(1, (150000 * :sf)::int) AS i;

-- Skewed: 60% of orders belong to the first 20% of customers.
INSERT INTO orders (o_orderkey, o_custkey, o_orderstatus, o_totalprice, o_orderdate,
                    o_orderpriority, o_clerk, o_shippriority, o_comment)
SELECT i,
       CASE WHEN random() < 0.6
            THEN 1 + (random() * (30000 * :sf)::int)::int
            ELSE 1 + (random() * (150000 * :sf - 1)::int)::int END,
       (ARRAY['O','F','P'])[1 + (random() * 2)::int],
       round((random() * 500000)::numeric, 2),
       DATE '1992-01-01' + (random() * 2405)::int,
       (ARRAY['1-URGENT','2-HIGH','3-MEDIUM','4-NOT SPECIFIED','5-LOW'])[1 + (random() * 4)::int],
       'Clerk#' || lpad((1 + (random() * 999)::int)::text, 9, '0'),
       0,
       'order comment ' || i
FROM generate_series(1, (1500000 * :sf)::int) AS i;

-- 1-7 line items per order, with parts skewed so 15% of parts take ~50% of lines.
INSERT INTO lineitem (l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity,
                      l_extendedprice, l_discount, l_tax, l_returnflag, l_linestatus,
                      l_shipdate, l_commitdate, l_receiptdate, l_shipinstruct, l_shipmode, l_comment)
SELECT o.o_orderkey,
       CASE WHEN random() < 0.5
            THEN 1 + (random() * (30000 * :sf)::int)::int
            ELSE 1 + (random() * (200000 * :sf - 1)::int)::int END,
       1 + (random() * (10000 * :sf - 1)::int)::int,
       ln.n,
       round((1 + random() * 49)::numeric, 2),
       round((random() * 100000)::numeric, 2),
       round((random() * 0.1)::numeric, 2),
       round((random() * 0.08)::numeric, 2),
       (ARRAY['R','A','N'])[1 + (random() * 2)::int],
       (ARRAY['O','F'])[1 + (random() * 1)::int],
       o.o_orderdate + (random() * 120)::int,
       o.o_orderdate + (random() * 90)::int,
       o.o_orderdate + (random() * 150)::int,
       (ARRAY['DELIVER IN PERSON','COLLECT COD','NONE','TAKE BACK RETURN'])[1 + (random() * 3)::int],
       (ARRAY['AIR','RAIL','SHIP','TRUCK','MAIL','FOB','REG AIR'])[1 + (random() * 6)::int],
       'lineitem comment'
FROM orders o,
     LATERAL generate_series(1, 1 + (random() * 6)::int) AS ln(n);

ANALYZE;
