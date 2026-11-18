-- Step 2: roll product lines up to one row per order (invoice).
-- Output: orders     (purchases only)
--         returns    (returned value per customer per day)
--         customers  (one row per customer: home country, first order)

CREATE OR REPLACE TABLE orders AS
SELECT
    invoice_id,
    customer_id,
    min(invoice_ts)              AS order_ts,
    min(invoice_date)            AS order_date,
    sum(line_amount)             AS order_value,
    sum(quantity)                AS units,
    count(DISTINCT product_id)   AS distinct_products
FROM transactions
WHERE NOT is_return
GROUP BY invoice_id, customer_id;

CREATE OR REPLACE TABLE returns AS
SELECT customer_id, invoice_date AS return_date, -sum(line_amount) AS return_value
FROM transactions
WHERE is_return
GROUP BY customer_id, invoice_date;

CREATE OR REPLACE TABLE customers AS
SELECT
    customer_id,
    -- a few customers order from more than one country: use the one they order from most
    mode(country)        AS country,
    min(invoice_date)    AS first_order_date
FROM transactions
WHERE NOT is_return
GROUP BY customer_id;
