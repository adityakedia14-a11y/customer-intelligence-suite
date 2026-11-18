"""
Runs the SQL layer in DuckDB and writes the tables the models use.

  sql/01_clean_transactions.sql  raw lines -> clean transactions
  sql/02_orders.sql              transactions -> orders, returns, customers
  sql/03_customer_features.sql   customer features as of a cutoff date + churn label

We build features at three cutoffs so models are always tested on the future:
  2010-12-09  TRAIN  features from the first year, label = next 6 months
  2011-06-09  TEST   features up to June 2011, label = next 6 months (unseen by training)
  2011-12-10  SCORE  features from all data -> today's predictions (no label yet)

Run: python src/build_features.py
"""
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
SQL = ROOT / "sql"
PROC = ROOT / "data" / "processed"
CUTOFFS = {"train": "2010-12-09", "test": "2011-06-09", "score": "2011-12-10"}


def run_sql(con, name):
    con.execute((SQL / name).read_text())


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(PROC / "retail.duckdb"))
    con.execute(f"""CREATE OR REPLACE VIEW raw AS SELECT * FROM read_csv_auto(
        '{ROOT / "data" / "raw" / "online_retail_ii.csv"}',
        types={{'Invoice': 'VARCHAR', 'StockCode': 'VARCHAR'}})""")

    run_sql(con, "01_clean_transactions.sql")
    run_sql(con, "02_orders.sql")
    print(con.sql("""SELECT (SELECT count(*) FROM raw) AS raw_rows,
                            (SELECT count(*) FROM transactions) AS clean_rows,
                            (SELECT count(*) FROM orders) AS orders,
                            (SELECT count(*) FROM customers) AS customers,
                            (SELECT round(sum(order_value)) FROM orders) AS revenue_gbp"""))

    for split, date in CUTOFFS.items():
        con.execute(f"SET VARIABLE cutoff = DATE '{date}'")
        run_sql(con, "03_customer_features.sql")
        df = con.sql("SELECT * FROM features ORDER BY customer_id").df()
        if split == "score":            # the future hasn't happened yet: no label
            df = df.drop(columns=["future_orders_180d", "future_revenue_180d", "churned"])
            print(f"{split:5s} cutoff {date}: {len(df):,} customers")
        else:
            print(f"{split:5s} cutoff {date}: {len(df):,} customers, churn rate {df.churned.mean():.1%}")
        df.to_csv(PROC / f"features_{split}.csv", index=False)

    con.sql("SELECT * FROM orders ORDER BY order_ts").df().to_csv(PROC / "orders.csv", index=False)
    con.sql("SELECT * FROM customers ORDER BY customer_id").df().to_csv(PROC / "customers.csv", index=False)
    con.close()


if __name__ == "__main__":
    main()
