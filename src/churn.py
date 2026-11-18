"""
Step 5: CHURN MODEL.  Churn = no order in the 180 days after the snapshot date.

Out-of-time validation (the realistic test - you always predict the future):
  TRAIN  features as of 2010-12-09, label from Dec 2010 - Jun 2011
  TEST   features as of 2011-06-09, label from Jun 2011 - Dec 2011 (never seen in training)

Four models, from simplest to most complex, so we can see if complexity pays off:
  1. Recency rule       - churn score = days since last order
  2. MBG/NBD            - score = -(expected orders in next 26 weeks), no ML
  3. Logistic regression on all features
  4. XGBoost, explained with SHAP values (XGBoost computes them natively: pred_contribs=True)
     Early stopping uses log-loss: AUC on a small validation set was too noisy and stopped
     after 2 trees. We also tried adding MBG/NBD outputs as features - no gain, so left out.

Result: every reasonable model lands near AUC 0.80, which looks like the limit of what
purchase history alone can tell us. XGBoost is kept because it gives calibrated
probabilities and per-customer reasons, not because it is much more accurate.

Final model: XGBoost refit on TRAIN + TEST, then used to score every customer today.
Each customer also gets their top 3 SHAP "reasons" so the dashboard can say WHY.

Run: python src/churn.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import clv

ROOT = Path(__file__).resolve().parents[1]
PROC, OUT = ROOT / "data" / "processed", ROOT / "outputs"
FEATURES = ["recency_days", "tenure_days", "frequency", "monetary_total", "avg_order_value",
            "avg_units_per_order", "distinct_products", "avg_days_between_orders",
            "orders_last_90d", "orders_prev_90d", "revenue_last_90d", "return_rate", "is_uk"]
NICE = {"recency_days": "Days since last order", "tenure_days": "Days as a customer",
        "frequency": "Number of orders", "monetary_total": "Total spend",
        "avg_order_value": "Average order value", "avg_units_per_order": "Units per order",
        "distinct_products": "Distinct products bought", "avg_days_between_orders": "Avg days between orders",
        "orders_last_90d": "Orders in last 90 days", "orders_prev_90d": "Orders 90-180 days ago",
        "revenue_last_90d": "Spend in last 90 days", "return_rate": "Return rate", "is_uk": "UK customer"}
XGB_PARAMS = dict(n_estimators=2000, learning_rate=0.02, max_depth=3, min_child_weight=5,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, eval_metric="logloss",
                  early_stopping_rounds=100, random_state=42)


def logistic_model():
    # skewed counts and money -> log; one-time buyers have no "avg days between orders" -> fill
    def prep(X):
        X = X.copy()
        X["avg_days_between_orders"] = X["avg_days_between_orders"].fillna(X["tenure_days"])
        for c in FEATURES:
            if c not in ("is_uk", "return_rate"):
                X[c] = np.log1p(X[c].clip(lower=0))
        return X
    return make_pipeline(FunctionTransformer(prep), StandardScaler(), LogisticRegression(max_iter=2000))


def fit_xgb(X, y, seed=42):
    """Hold back 20% of the training rows only to decide when to stop adding trees."""
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    m = xgb.XGBClassifier(**XGB_PARAMS)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return m


def scores(y, p, k=0.2):
    """Ranking quality + how good the top-k% list is (what a retention team would actually call)."""
    top = pd.Series(p).rank(ascending=False, method="first").to_numpy() <= k * len(p)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "top20_precision": y[top].mean(), "top20_lift": y[top].mean() / y.mean()}


def mbgnbd_score(orders, cutoff, customer_ids):
    cal = clv.rfm_summary(orders, pd.Timestamp(cutoff))
    model = clv.MBGNBD().fit(cal)
    exp = pd.Series(model.expected_purchases(26, cal.x.values, cal.t_x.values, cal["T"].values), cal.index)
    return -exp.reindex(customer_ids).to_numpy()


def shap_reasons(model, X: pd.DataFrame, n=3) -> pd.DataFrame:
    """Per-customer SHAP values -> the n features pushing churn risk UP the most."""
    contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]  # last col = bias
    contrib = pd.DataFrame(contrib, columns=X.columns, index=X.index)
    reasons = []
    for _, row in contrib.iterrows():
        top = row[row > 0].sort_values(ascending=False).head(n)
        reasons.append([NICE[c] for c in top.index] + [""] * (n - len(top)))
    return contrib, pd.DataFrame(reasons, columns=[f"reason_{i + 1}" for i in range(n)], index=X.index)


def main():
    OUT.mkdir(exist_ok=True)
    train = pd.read_csv(PROC / "features_train.csv")
    test = pd.read_csv(PROC / "features_test.csv")
    score = pd.read_csv(PROC / "features_score.csv")
    orders = pd.read_csv(PROC / "orders.csv", parse_dates=["order_date"])
    Xtr, ytr, Xte, yte = train[FEATURES], train.churned.to_numpy(), test[FEATURES], test.churned.to_numpy()

    # ---------------- compare models on the unseen period ----------------
    preds = {
        "Recency rule": test.recency_days.to_numpy(),
        "MBG/NBD (no ML)": mbgnbd_score(orders, "2011-06-09", test.customer_id),
    }
    logit = logistic_model().fit(Xtr, ytr)
    preds["Logistic regression"] = logit.predict_proba(Xte)[:, 1]
    xgb_model = fit_xgb(Xtr, ytr)
    preds["XGBoost"] = xgb_model.predict_proba(Xte)[:, 1]

    comparison = pd.DataFrame({name: scores(yte, p) for name, p in preds.items()}).T
    comparison.to_csv(OUT / "churn_model_comparison.csv")
    print("Test period (unseen):\n", comparison.round(3).to_string())

    p = preds["XGBoost"]
    extra = {"test_churn_rate": float(yte.mean()), "best_iteration": int(xgb_model.best_iteration),
             "brier": float(brier_score_loss(yte, p)),
             "precision_at_0.5": float(precision_score(yte, p > 0.5)),
             "recall_at_0.5": float(recall_score(yte, p > 0.5))}
    print(extra)

    # ---------------- SHAP on the test period ----------------
    contrib, _ = shap_reasons(xgb_model, Xte)
    importance = contrib.abs().mean().sort_values(ascending=False)
    importance.rename(index=NICE).to_csv(OUT / "churn_feature_importance.csv", header=["mean_abs_shap"])

    # ---------------- plots ----------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for name, s in preds.items():
        fpr, tpr, _ = roc_curve(yte, s)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(yte, s):.3f})")
    ax[0].plot([0, 1], [0, 1], color="grey", ls=":")
    ax[0].set(title="ROC on unseen period (Jun-Dec 2011)", xlabel="false positive rate", ylabel="true positive rate")
    ax[0].legend(fontsize=8)
    imp = importance.rename(index=NICE).iloc[::-1]
    ax[1].barh(imp.index, imp.values, color="#1d4ed8")
    ax[1].set(title="What drives churn risk (mean |SHAP|)")
    bins = pd.cut(p, np.linspace(0, 1, 11), include_lowest=True)
    cal_tbl = pd.DataFrame({"bin": bins, "p": p, "y": yte}).groupby("bin", observed=True).mean()
    ax[2].plot(cal_tbl.p, cal_tbl.y, "o-", color="#1d4ed8")
    ax[2].plot([0, 1], [0, 1], color="grey", ls=":")
    ax[2].set(title="Calibration: predicted vs actual churn", xlabel="predicted probability", ylabel="actual churn rate")
    fig.tight_layout(); fig.savefig(OUT / "churn_model.png", dpi=150); plt.close(fig)

    # ---------------- final model: refit on train + test, score everyone today ----------------
    both = pd.concat([train, test], ignore_index=True)
    final = fit_xgb(both[FEATURES], both.churned.to_numpy())
    Xs = score[FEATURES]
    _, reasons = shap_reasons(final, Xs)
    out = pd.DataFrame({"customer_id": score.customer_id, "churn_probability": final.predict_proba(Xs)[:, 1]})
    out["churn_risk_band"] = pd.cut(out.churn_probability, [0, 0.3, 0.6, 1.0],
                                    labels=["Low", "Medium", "High"], include_lowest=True).astype(str)
    out = pd.concat([out, reasons.reset_index(drop=True)], axis=1)
    out.to_csv(PROC / "churn_scores.csv", index=False)
    print(out.churn_risk_band.value_counts().to_string())

    json.dump({"comparison": comparison.to_dict(orient="index"), **extra,
               "feature_importance": importance.rename(index=NICE).to_dict()},
              open(OUT / "churn_metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
