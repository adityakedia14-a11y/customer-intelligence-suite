-- Step 3: customer features "as of" a cutoff date, plus a churn label.
--
-- Everything on the feature side uses ONLY orders before the cutoff, so a model trained
-- on these features never sees the future. The label looks at the 180 days after the cutoff.
--
-- Set before running:   SET VARIABLE cutoff = DATE '2011-06-09';
-- Output: features      (one row per customer who had ordered before the cutoff)

CREATE OR REPLACE TABLE features AS
WITH params AS (
    SELECT getvariable('cutoff') AS cutoff
),
hist AS (               -- orders before the cutoff
    SELECT o.* FROM orders o, params p WHERE o.order_date < p.cutoff
),
per_customer AS (
    SELECT
        h.customer_id,
        count(*)                                                    AS frequency,
        sum(order_value)                                            AS monetary_total,
        avg(order_value)                                            AS avg_order_value,
        avg(units)                                                  AS avg_units_per_order,
        min(order_date)                                             AS first_order,
        max(order_date)                                             AS last_order,
        count(*) FILTER (WHERE order_date >= p.cutoff - INTERVAL 90 DAY)            AS orders_last_90d,
        count(*) FILTER (WHERE order_date >= p.cutoff - INTERVAL 180 DAY
                           AND order_date <  p.cutoff - INTERVAL 90 DAY)             AS orders_prev_90d,
        coalesce(sum(order_value) FILTER (WHERE order_date >= p.cutoff - INTERVAL 90 DAY), 0)
                                                                    AS revenue_last_90d
    FROM hist h, params p
    GROUP BY h.customer_id
),
product_breadth AS (
    SELECT t.customer_id, count(DISTINCT t.product_id) AS distinct_products
    FROM transactions t, params p
    WHERE NOT t.is_return AND t.invoice_date < p.cutoff
    GROUP BY t.customer_id
),
ret AS (
    SELECT r.customer_id, sum(return_value) AS return_value
    FROM returns r, params p WHERE r.return_date < p.cutoff
    GROUP BY r.customer_id
),
future AS (             -- what happened AFTER the cutoff (only used for the label)
    SELECT o.customer_id, count(*) AS future_orders, sum(order_value) AS future_revenue
    FROM orders o, params p
    WHERE o.order_date >= p.cutoff AND o.order_date < p.cutoff + INTERVAL 180 DAY
    GROUP BY o.customer_id
)
SELECT
    c.customer_id,
    c.country,
    (c.country = 'United Kingdom')::INT                             AS is_uk,
    p.cutoff                                                        AS cutoff_date,
    date_diff('day', pc.last_order, p.cutoff)                       AS recency_days,
    date_diff('day', pc.first_order, p.cutoff)                      AS tenure_days,
    pc.frequency,
    pc.monetary_total,
    pc.avg_order_value,
    pc.avg_units_per_order,
    pb.distinct_products,
    -- average gap between orders; NULL for one-time buyers
    CASE WHEN pc.frequency > 1
         THEN date_diff('day', pc.first_order, pc.last_order) / (pc.frequency - 1) END
                                                                    AS avg_days_between_orders,
    pc.orders_last_90d,
    pc.orders_prev_90d,
    pc.revenue_last_90d,
    coalesce(r.return_value, 0) / pc.monetary_total                 AS return_rate,
    coalesce(f.future_orders, 0)                                    AS future_orders_180d,
    coalesce(f.future_revenue, 0)                                   AS future_revenue_180d,
    (coalesce(f.future_orders, 0) = 0)::INT                         AS churned   -- no order in next 180 days
FROM per_customer pc
JOIN customers c        USING (customer_id)
JOIN product_breadth pb USING (customer_id)
LEFT JOIN ret r         USING (customer_id)
LEFT JOIN future f      USING (customer_id)
CROSS JOIN params p;
