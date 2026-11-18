"""
Step 7: EXPORT FOR POWER BI.

Joins every model's output into a small star schema and writes it as one Excel workbook
(easiest to load: one "Get data" step) plus the same tables as CSVs.

  Retail (UCI Online Retail II)
    Customers         one row per customer: segment, RFM, CLV, churn risk + reasons, priority
    Orders            one row per order (fact table)
    Segments          lookup: sort order, description, colour, suggested action
    ModelPerformance  churn model comparison on the unseen period
    ChurnDrivers      SHAP feature importance
    CLVValidation     lifetime value model: predicted vs actual by decile (holdout)
  Campaign (Hillstrom email experiment)
    CampaignCustomers one row per customer: predicted uplift, uplift group, decile
    UpliftDeciles     actual vs predicted uplift by decile (test set)
    TargetingCurve    share of customers emailed -> share of extra visits captured

Run: python src/export_powerbi.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import clv as clv_mod

ROOT = Path(__file__).resolve().parents[1]
PROC, OUT, PBI = ROOT / "data" / "processed", ROOT / "outputs", ROOT / "powerbi"

SEGMENTS = pd.DataFrame({
    "segment": ["Champions", "Loyal", "New / Promising", "At Risk", "Lost"],
    "segment_order": [1, 2, 3, 4, 5],
    "description": ["Ordered very recently, very often, spend the most",
                    "Regular customers with solid spend",
                    "Recent but few orders so far",
                    "Used to order regularly, gone quiet for months",
                    "One-off buyers from long ago"],
    "suggested_action": ["Reward and protect: VIP perks, early access",
                         "Upsell and cross-sell",
                         "Onboarding series to drive the 2nd and 3rd order",
                         "Win-back offer now, before they become Lost",
                         "Low-cost reactivation only, or exclude"],
    "colour": ["#1D4ED8", "#16A34A", "#A855F7", "#F59E0B", "#9CA3AF"],
})


def priority(row, clv_cut):
    high_value = row.clv_12m >= clv_cut
    high_risk = row.churn_probability >= 0.5
    if high_value and high_risk:
        return "1. Protect (high value, high risk)"
    if high_value:
        return "2. Nurture (high value, low risk)"
    if high_risk:
        return "3. Low-cost win-back (low value, high risk)"
    return "4. Maintain (low value, low risk)"


def clv_validation_deciles(orders):
    cal = clv_mod.rfm_summary(orders, clv_mod.CALIBRATION_END)
    bg, gg = clv_mod.MBGNBD().fit(cal), clv_mod.GammaGamma().fit(cal)
    pred = bg.expected_purchases(clv_mod.HOLDOUT_WEEKS, cal.x.values, cal.t_x.values, cal["T"].values) \
        * gg.expected_value(cal.x.values, cal.monetary.values)
    hold = orders[orders.order_date >= clv_mod.CALIBRATION_END]
    actual = hold.groupby("customer_id").order_value.sum().reindex(cal.index).fillna(0)
    dec = pd.qcut(pd.Series(pred, cal.index).rank(method="first"), 10, labels=range(1, 11)).astype(int)
    t = pd.DataFrame({"decile": dec, "predicted_revenue": pred, "actual_revenue": actual}) \
          .groupby("decile").sum().reset_index()
    return t


def main():
    PBI.mkdir(exist_ok=True)
    feats = pd.read_csv(PROC / "features_score.csv")
    seg = pd.read_csv(PROC / "segments.csv")
    clv = pd.read_csv(PROC / "clv.csv")
    churn = pd.read_csv(PROC / "churn_scores.csv")
    cust_info = pd.read_csv(PROC / "customers.csv", parse_dates=["first_order_date"])
    orders = pd.read_csv(PROC / "orders.csv", parse_dates=["order_date"])

    c = (feats[["customer_id", "recency_days", "frequency", "monetary_total", "avg_order_value",
                "distinct_products", "return_rate"]]
         .merge(cust_info, on="customer_id")
         .merge(seg, on="customer_id")
         .merge(clv, on="customer_id")
         .merge(churn, on="customer_id"))
    c["last_order_date"] = pd.Timestamp("2011-12-10") - pd.to_timedelta(c.recency_days, unit="D")
    c["value_at_risk"] = c.clv_12m * c.churn_probability
    clv_cut = c.clv_12m.quantile(0.7)                 # top 30% by predicted value = "high value"
    c["priority"] = c.apply(priority, axis=1, clv_cut=clv_cut)
    c = c.merge(SEGMENTS[["segment", "suggested_action"]], on="segment")
    for col in ["reason_1", "reason_2", "reason_3"]:
        c[col] = c[col].fillna("")
    c = c[["customer_id", "country", "first_order_date", "last_order_date", "segment", "r_score", "f_score",
           "m_score", "rfm_score", "recency_days", "frequency", "monetary_total", "avg_order_value",
           "distinct_products", "return_rate", "p_alive", "expected_orders_12m", "expected_order_value",
           "clv_12m", "clv_decile", "churn_probability", "churn_risk_band", "reason_1", "reason_2",
           "reason_3", "value_at_risk", "priority", "suggested_action"]]

    o = orders[["invoice_id", "customer_id", "order_date", "order_value", "units", "distinct_products"]]

    perf = pd.read_csv(OUT / "churn_model_comparison.csv", index_col=0)
    perf = perf.rename_axis("model").reset_index()
    perf["model_order"] = range(1, len(perf) + 1)
    drivers = pd.read_csv(OUT / "churn_feature_importance.csv")
    drivers.columns = ["driver", "mean_abs_shap"]

    camp = pd.read_csv(PROC / "uplift_scores.csv")
    deciles = pd.read_csv(OUT / "uplift_by_decile.csv")
    curve = pd.read_csv(OUT / "uplift_targeting_curve.csv")
    curve["share_emailed_pct"] = (curve.share_emailed * 100).round().astype(int)

    tables = {"Customers": c, "Orders": o, "Segments": SEGMENTS, "ModelPerformance": perf,
              "ChurnDrivers": drivers, "CLVValidation": clv_validation_deciles(orders),
              "CampaignCustomers": camp, "UpliftDeciles": deciles, "TargetingCurve": curve}

    csv_dir = PBI / "data"
    csv_dir.mkdir(exist_ok=True)
    with pd.ExcelWriter(PBI / "customer_intelligence.xlsx", engine="openpyxl") as xl:
        for name, df in tables.items():
            df = df.copy()
            for col in df.select_dtypes("float").columns:
                df[col] = df[col].round(4)
            df.to_excel(xl, sheet_name=name, index=False)
            df.to_csv(csv_dir / f"{name}.csv", index=False)
            print(f"{name:18s} {len(df):>6,} rows  {df.shape[1]:>2} cols")

    # Numbers quoted in the build guide, so it can be checked against the data
    check = {
        "customers": int(len(c)),
        "orders": int(len(o)),
        "revenue": float(o.order_value.sum()),
        "predicted_clv_12m_total": float(c.clv_12m.sum()),
        "value_at_risk_total": float(c.value_at_risk.sum()),
        "high_risk_customers": int((c.churn_risk_band == "High").sum()),
        "protect_customers": int((c.priority.str.startswith("1.")).sum()),
        "protect_value_at_risk": float(c.loc[c.priority.str.startswith("1."), "value_at_risk"].sum()),
        "high_value_clv_cutoff": float(clv_cut),
        "campaign_customers": int(len(camp)),
        "persuadables": int((camp.uplift_group == "Persuadables").sum()),
    }
    json.dump(check, open(PBI / "expected_numbers.json", "w"), indent=2)
    print(json.dumps(check, indent=2))


if __name__ == "__main__":
    main()
