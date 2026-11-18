"""
Step 6: UPLIFT MODELLING on a randomised email experiment (Hillstrom, MineThatData 2008).

64,000 customers were randomly split into three groups: men's email, women's email, no email.
We use WOMEN'S EMAIL vs NO EMAIL, outcome "visited the website within two weeks"
(purchases are under 1%, too rare to model).

Why the women's email? The men's email has the bigger average effect (+7.7 points), but
that effect is almost the same for every type of customer, so targeting can't improve on
emailing everyone. The women's email works for customers who have bought women's products
(+7.3 points) and barely at all for others (+1.1) - that variation is what uplift models need.

Why uplift and not a normal response model? A response model finds people likely to visit.
Many of them would visit anyway, so emailing them is wasted. Uplift estimates the CHANGE
the email causes for each customer:
        uplift(x) = P(visit | email, x) - P(visit | no email, x)
That's only possible because the email was randomly assigned.

Models (70/30 split, stratified on treatment and outcome):
  * T-learner : one model on emailed customers, one on control; uplift = difference.
  * S-learner : one model with "was emailed" as a feature; uplift = predict(1) - predict(0).
Evaluated with the Qini curve: target customers in order of predicted uplift and count the
EXTRA visits the email produces, compared with targeting at random.

Run: python src/uplift.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "outputs"
TREATMENT, CONTROL, OUTCOME = "Womens E-Mail", "No E-Mail", "visit"
XGB = dict(n_estimators=300, learning_rate=0.05, max_depth=3, min_child_weight=20,
           subsample=0.8, colsample_bytree=0.8, random_state=42)


def load():
    d = pd.read_csv(RAW / "hillstrom.csv")
    d = d[d.segment.isin([TREATMENT, CONTROL])].reset_index(drop=True)
    d.insert(0, "customer_id", np.arange(1, len(d) + 1))
    d["treated"] = (d.segment == TREATMENT).astype(int)
    X = pd.get_dummies(d[["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]],
                       columns=["zip_code", "channel"], dtype=int)
    return d, X


def t_learner(X, t, y, X_new):
    m1 = xgb.XGBClassifier(**XGB).fit(X[t == 1], y[t == 1])
    m0 = xgb.XGBClassifier(**XGB).fit(X[t == 0], y[t == 0])
    p1, p0 = m1.predict_proba(X_new)[:, 1], m0.predict_proba(X_new)[:, 1]
    return p1 - p0, p1, p0


def s_learner(X, t, y, X_new):
    m = xgb.XGBClassifier(**XGB).fit(X.assign(treated=t), y)
    p1 = m.predict_proba(X_new.assign(treated=1))[:, 1]
    p0 = m.predict_proba(X_new.assign(treated=0))[:, 1]
    return p1 - p0, p1, p0


def qini(uplift, t, y, steps=100):
    """Extra visits gained when targeting the top fraction (by predicted uplift)."""
    order = np.argsort(-uplift)
    t, y = t[order], y[order]
    n = len(t)
    frac = np.linspace(0, 1, steps + 1)
    gains = []
    for f in frac:
        k = int(round(f * n))
        nt, nc = t[:k].sum(), (1 - t[:k]).sum()
        yt, yc = y[:k][t[:k] == 1].sum(), y[:k][t[:k] == 0].sum()
        gains.append(yt - yc * nt / nc if nc > 0 else 0.0)
    gains = np.array(gains)
    random_line = frac * gains[-1]
    return frac, gains, np.trapezoid(gains - random_line, frac)


def by_decile(uplift, t, y):
    """Actual visit rate, emailed vs not, within each decile of predicted uplift."""
    dec = pd.qcut(pd.Series(uplift).rank(method="first", ascending=False), 10, labels=range(1, 11))
    d = pd.DataFrame({"decile": dec, "t": t, "y": y})
    g = d.groupby(["decile", "t"], observed=True).y.mean().unstack()
    g.columns = ["control_rate", "treated_rate"]
    g["actual_uplift"] = g.treated_rate - g.control_rate
    g["predicted_uplift"] = pd.Series(uplift).groupby(dec, observed=True).mean()
    return g


def main():
    OUT.mkdir(exist_ok=True)
    d, X = load()
    t, y = d.treated.to_numpy(), d[OUTCOME].to_numpy()
    idx_tr, idx_te = train_test_split(np.arange(len(d)), test_size=0.3, random_state=42,
                                      stratify=t * 2 + y)
    Xtr, Xte = X.iloc[idx_tr], X.iloc[idx_te]

    u_t, _, _ = t_learner(Xtr, t[idx_tr], y[idx_tr], Xte)
    u_s, _, _ = s_learner(Xtr, t[idx_tr], y[idx_tr], Xte)
    results = {}
    curves = {}
    for name, u in {"T-learner": u_t, "S-learner": u_s}.items():
        frac, gains, area = qini(u, t[idx_te], y[idx_te])
        curves[name] = (frac, gains)
        results[name] = {"qini_area": float(area)}
    best = max(results, key=lambda k: results[k]["qini_area"])
    u_best = u_t if best == "T-learner" else u_s
    print("Qini area (higher = better):", {k: round(v["qini_area"], 1) for k, v in results.items()}, "->", best)

    dec = by_decile(u_best, t[idx_te], y[idx_te])
    dec.to_csv(OUT / "uplift_by_decile.csv")
    print(dec.round(4).to_string())

    # Business view: email only the top X% by predicted uplift vs emailing everyone
    frac, gains = curves[best]
    total = gains[-1]
    targeting = pd.DataFrame({"share_emailed": frac, "extra_visits": gains,
                              "share_of_all_extra_visits": gains / total})
    targeting.to_csv(OUT / "uplift_targeting_curve.csv", index=False)
    at = {f"top_{int(p * 100)}pct_captures": float(np.interp(p, frac, gains / total)) for p in (0.3, 0.5)}
    ate = float(y[idx_te][t[idx_te] == 1].mean() - y[idx_te][t[idx_te] == 0].mean())
    print("Average effect of the email (test set):", round(ate, 4), at)

    # Plots
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, (f, g) in curves.items():
        ax[0].plot(f, g, label=f"{name} (Qini area {results[name]['qini_area']:.0f})")
    ax[0].plot([0, 1], [0, total], color="grey", ls=":", label="random targeting")
    ax[0].set(title="Qini curve (test set)", xlabel="share of customers emailed (by predicted uplift)",
              ylabel="extra visits caused by the email")
    ax[0].legend(fontsize=8)
    ax[1].bar(dec.index.astype(int), dec.actual_uplift * 100, color="#1d4ed8", label="actual")
    ax[1].plot(dec.index.astype(int), dec.predicted_uplift * 100, "o-", color="#f59e0b", label="predicted")
    ax[1].axhline(ate * 100, color="grey", ls="--", label="average effect")
    ax[1].set(title="Uplift by predicted-uplift decile (test set)", xlabel="decile (1 = highest predicted uplift)",
              ylabel="visit-rate uplift (points)")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "uplift.png", dpi=150); plt.close(fig)

    # Score every customer with a model trained on all data -> Power BI campaign page
    learner = t_learner if best == "T-learner" else s_learner
    u_all, p1, p0 = learner(X, t, y, X)
    scored = d[["customer_id", "recency", "history", "history_segment", "zip_code", "channel", "newbie",
                "segment", OUTCOME]].copy()
    scored["predicted_uplift"] = u_all
    scored["uplift_decile"] = pd.qcut(pd.Series(u_all).rank(method="first", ascending=False), 10,
                                      labels=range(1, 11)).astype(int)
    scored["p_visit_if_emailed"], scored["p_visit_if_not"] = p1, p0
    # Four classic groups. Persuadables = top 30% by uplift; the rest split by how likely
    # they are to visit anyway.
    hi_base = p0 >= np.median(p0)
    scored["uplift_group"] = np.select(
        [u_all < 0, u_all >= np.quantile(u_all, 0.7), hi_base],
        ["Sleeping dogs", "Persuadables", "Sure things"], default="Lost causes")
    scored.to_csv(PROC / "uplift_scores.csv", index=False)
    print(scored.groupby("uplift_group").agg(customers=("customer_id", "size"),
          avg_uplift=("predicted_uplift", "mean"), p_visit_if_not=("p_visit_if_not", "mean")).round(3))

    json.dump({"treatment": TREATMENT, "control": CONTROL, "outcome": OUTCOME,
               "test_average_effect": ate, "models": results, "best_model": best, **at},
              open(OUT / "uplift_metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
