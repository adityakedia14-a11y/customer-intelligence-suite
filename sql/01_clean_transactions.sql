-- Step 1: clean line-level transactions.
-- Input : raw            (view over data/raw/online_retail_ii.csv)
-- Output: transactions   (one row per product line, sales and returns, customers only)

CREATE OR REPLACE TABLE transactions AS
WITH deduped AS (
    -- the two yearly Excel sheets overlap in Dec 2010, which creates exact duplicate rows
    SELECT DISTINCT * FROM raw
)
SELECT
    Invoice                                   AS invoice_id,
    StockCode                                 AS product_id,
    trim(Description)                         AS product_name,
    CAST("Customer ID" AS INTEGER)            AS customer_id,
    Country                                   AS country,
    InvoiceDate                               AS invoice_ts,
    CAST(InvoiceDate AS DATE)                 AS invoice_date,
    Quantity                                  AS quantity,
    Price                                     AS unit_price,
    Quantity * Price                          AS line_amount,     -- negative for returns
    Invoice LIKE 'C%'                         AS is_return
FROM deduped
WHERE "Customer ID" IS NOT NULL                                   -- guest checkouts can't be tracked
  AND Price > 0
  AND (regexp_matches(StockCode, '^[0-9]{5}') OR StockCode LIKE 'DCGS%')   -- real products only
  AND (Invoice LIKE 'C%') = (Quantity < 0);                       -- sales have qty > 0, returns qty < 0
