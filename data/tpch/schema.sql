-- TPC-H schema (the standard decision-support benchmark).
--
-- Eight tables in a snowflake around lineitem, which is by far the largest.
-- Foreign keys are declared here (unlike JOB), so schema_graph.discover finds
-- the join graph without needing naming-convention inference.
--
-- No indexes beyond the primary keys, deliberately: the point is to leave
-- join order and join method consequential, exactly as data/schema.sql does
-- for the synthetic dataset.

CREATE TABLE region (
    r_regionkey  INTEGER PRIMARY KEY,
    r_name       CHAR(25) NOT NULL,
    r_comment    VARCHAR(152)
);

CREATE TABLE nation (
    n_nationkey  INTEGER PRIMARY KEY,
    n_name       CHAR(25) NOT NULL,
    n_regionkey  INTEGER NOT NULL REFERENCES region(r_regionkey),
    n_comment    VARCHAR(152)
);

CREATE TABLE supplier (
    s_suppkey    INTEGER PRIMARY KEY,
    s_name       CHAR(25) NOT NULL,
    s_address    VARCHAR(40) NOT NULL,
    s_nationkey  INTEGER NOT NULL REFERENCES nation(n_nationkey),
    s_phone      CHAR(15) NOT NULL,
    s_acctbal    NUMERIC(12,2) NOT NULL,
    s_comment    VARCHAR(101) NOT NULL
);

CREATE TABLE part (
    p_partkey    INTEGER PRIMARY KEY,
    p_name       VARCHAR(55) NOT NULL,
    p_mfgr       CHAR(25) NOT NULL,
    p_brand      CHAR(10) NOT NULL,
    p_type       VARCHAR(25) NOT NULL,
    p_size       INTEGER NOT NULL,
    p_container  CHAR(10) NOT NULL,
    p_retailprice NUMERIC(12,2) NOT NULL,
    p_comment    VARCHAR(23) NOT NULL
);

CREATE TABLE partsupp (
    ps_partkey   INTEGER NOT NULL REFERENCES part(p_partkey),
    ps_suppkey   INTEGER NOT NULL REFERENCES supplier(s_suppkey),
    ps_availqty  INTEGER NOT NULL,
    ps_supplycost NUMERIC(12,2) NOT NULL,
    ps_comment   VARCHAR(199) NOT NULL,
    PRIMARY KEY (ps_partkey, ps_suppkey)
);

CREATE TABLE customer (
    c_custkey    INTEGER PRIMARY KEY,
    c_name       VARCHAR(25) NOT NULL,
    c_address    VARCHAR(40) NOT NULL,
    c_nationkey  INTEGER NOT NULL REFERENCES nation(n_nationkey),
    c_phone      CHAR(15) NOT NULL,
    c_acctbal    NUMERIC(12,2) NOT NULL,
    c_mktsegment CHAR(10) NOT NULL,
    c_comment    VARCHAR(117) NOT NULL
);

CREATE TABLE orders (
    o_orderkey   INTEGER PRIMARY KEY,
    o_custkey    INTEGER NOT NULL REFERENCES customer(c_custkey),
    o_orderstatus CHAR(1) NOT NULL,
    o_totalprice NUMERIC(12,2) NOT NULL,
    o_orderdate  DATE NOT NULL,
    o_orderpriority CHAR(15) NOT NULL,
    o_clerk      CHAR(15) NOT NULL,
    o_shippriority INTEGER NOT NULL,
    o_comment    VARCHAR(79) NOT NULL
);

CREATE TABLE lineitem (
    l_orderkey   INTEGER NOT NULL REFERENCES orders(o_orderkey),
    l_partkey    INTEGER NOT NULL REFERENCES part(p_partkey),
    l_suppkey    INTEGER NOT NULL REFERENCES supplier(s_suppkey),
    l_linenumber INTEGER NOT NULL,
    l_quantity   NUMERIC(12,2) NOT NULL,
    l_extendedprice NUMERIC(12,2) NOT NULL,
    l_discount   NUMERIC(12,2) NOT NULL,
    l_tax        NUMERIC(12,2) NOT NULL,
    l_returnflag CHAR(1) NOT NULL,
    l_linestatus CHAR(1) NOT NULL,
    l_shipdate   DATE NOT NULL,
    l_commitdate DATE NOT NULL,
    l_receiptdate DATE NOT NULL,
    l_shipinstruct CHAR(25) NOT NULL,
    l_shipmode   CHAR(10) NOT NULL,
    l_comment    VARCHAR(44) NOT NULL,
    PRIMARY KEY (l_orderkey, l_linenumber)
);
