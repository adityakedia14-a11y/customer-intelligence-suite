"""
Step 4: CUSTOMER LIFETIME VALUE with (M)BG/NBD + Gamma-Gamma, written with scipy.

BG/NBD (Fader, Hardie & Lee, 2005) - "how many more orders?"
  * While a customer is active they order at their own rate  lambda ~ Gamma(r, alpha).
  * After each order they may drop out for good with probability  p ~ Beta(a, b).
  * Inputs per customer (time in weeks):
        x   = number of REPEAT purchase days (first purchase not counted)
        t_x = time from first to last purchase
        T   = time from first purchase to the end of the observation period
  * Outputs: P(still active) and expected orders in the next t weeks.
  * We use the MODIFIED version (MBG/NBD) - see the note above the class.

Gamma-Gamma (Fader, Hardie & Lee, 2005) - "how much per order?"
  * Each customer's order values vary around their own mean; those means vary across
    customers. Result: a customer's average is shrunk towards the population average,
    more strongly when they have few orders.
  * Fit on customers with at least one repeat purchase, using repeat-order values.

CLV (revenue, next 12 months) = expected orders x expected order value.

Validation: fit on orders before 2011-06-09, predict the following 26 weeks, compare
with what actually happened. Then refit on everything and predict the next 52 weeks.

Run: python src/clv.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, hyp2f1, logsumexp
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
PROC, OUT = ROOT / "data" / "processed", ROOT / "outputs"
CALIBRATION_END = pd.Timestamp("2011-06-09")
DATA_END = pd.Timestamp("2011-12-10")            # day after the last order in the data
HOLDOUT_WEEKS = (DATA_END - CALIBRATION_END).days / 7
FORECAST_WEEKS = 52


# --------------------------------------------------------------------------- #
# Data: orders -> (x, t_x, T, monetary) per customer
# --------------------------------------------------------------------------- #

def rfm_summary(orders: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    o = orders[orders.order_date < end]
    # several orders on the same day count as one purchase occasion (standard for BG/NBD)
    daily = o.groupby(["customer_id", "order_date"], as_index=False).order_value.sum()
    g = daily.groupby("customer_id")
    first, last, n = g.order_date.min(), g.order_date.max(), g.size()
    out = pd.DataFrame({
        "x": n - 1,
        "t_x": (last - first).dt.days / 7,
        "T": (end - first).dt.days / 7,
    })
    # Gamma-Gamma uses the average value of REPEAT purchases (drop each customer's first day)
    daily["rank"] = g.cumcount()
    repeat = daily[daily["rank"] > 0].groupby("customer_id").order_value.mean()
    out["monetary"] = repeat.reindex(out.index).fillna(0.0)
    return out


# --------------------------------------------------------------------------- #
# MBG/NBD (modified BG/NBD, Batislam et al. 2007)
#   Plain BG/NBD assumes a customer can only drop out AFTER a repeat purchase, so every
#   one-time buyer has P(alive) = 1 - even someone who bought once, 400 days ago.
#   MBG/NBD also allows dropping out straight after the first purchase. Same accuracy on
#   our holdout, but sensible P(alive) for one-time buyers.
# --------------------------------------------------------------------------- #

def multistart_minimize(neg_ll, starts):
    """Nelder-Mead from several starting points; keep the best. A single start can get stuck."""
    best = None
    for x0 in starts:
        res = minimize(neg_ll, np.asarray(x0, float), method="Nelder-Mead",
                       options={"maxiter": 8000, "xatol": 1e-9, "fatol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    return best


class MBGNBD:
    def _loglik(self, params, x, t_x, T):
        r, alpha, a, b = np.exp(params)          # optimise on the log scale so all stay > 0
        A = gammaln(r + x) - gammaln(r) + r * np.log(alpha) + betaln(a, b + x + 1) - betaln(a, b)
        still_alive = -(r + x) * np.log(alpha + T)
        dropped_out = np.log(a) - np.log(b + x) - (r + x) * np.log(alpha + t_x)
        return (A + logsumexp(np.vstack([still_alive, dropped_out]), axis=0)).sum()

    def fit(self, df: pd.DataFrame):
        x, t_x, T = df.x.to_numpy(), df.t_x.to_numpy(), df["T"].to_numpy()
        res = multistart_minimize(lambda p: -self._loglik(p, x, t_x, T),
                                  [np.zeros(4), np.log([0.5, 5, 0.5, 3]), np.log([1, 20, 1, 5])])
        self.r, self.alpha, self.a, self.b = np.exp(res.x)
        self.converged = bool(res.success)
        return self

    def _odds_dead(self, x, t_x, T):
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        return (a / (b + x)) * ((alpha + T) / (alpha + t_x)) ** (r + x)

    def p_alive(self, x, t_x, T):
        return 1.0 / (1.0 + self._odds_dead(x, t_x, T))

    def expected_purchases(self, t, x, t_x, T):
        """Expected number of purchases in the next t weeks, given each customer's history."""
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        z = t / (alpha + T + t)
        num = (a + b + x) / (a - 1) * (
            1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp2f1(r + x, b + x + 1, a + b + x, z))
        return num / (1.0 + self._odds_dead(x, t_x, T))


# --------------------------------------------------------------------------- #
# Gamma-Gamma
# --------------------------------------------------------------------------- #

class GammaGamma:
    def _loglik(self, params, x, m):
        p, q, v = np.exp(params)
        return (gammaln(p * x + q) - gammaln(p * x) - gammaln(q) + q * np.log(v)
                + (p * x - 1) * np.log(m) + p * x * np.log(x) - (p * x + q) * np.log(v + m * x)).sum()

    def fit(self, df: pd.DataFrame):
        d = df[(df.x > 0) & (df.monetary > 0)]
        x, m = d.x.to_numpy(), d.monetary.to_numpy()
        res = multistart_minimize(lambda p: -self._loglik(p, x, m),
                                  [np.zeros(3), np.log([2, 3, 300]), np.log([5, 5, 1000])])
        self.p, self.q, self.v = np.exp(res.x)
        self.converged = bool(res.success)
        self.population_mean = self.p * self.v / (self.q - 1)
        return self

    def expected_value(self, x, m):
        """Expected average order value; customers with no repeat purchases get the population mean."""
        p, q, v = self.p, self.q, self.v
        shrunk = p * (v + x * m) / (p * x + q - 1)
        return np.where(x > 0, shrunk, self.population_mean)


# --------------------------------------------------------------------------- #

def validate(orders):
    cal = rfm_summary(orders, CALIBRATION_END)
    bg, gg = MBGNBD().fit(cal), GammaGamma().fit(cal)

    hold = orders[(orders.order_date >= CALIBRATION_END) & (orders.order_date < DATA_END)]
    hold_daily = hold.groupby(["customer_id", "order_date"]).order_value.sum().reset_index()
    actual_n = hold_daily.groupby("customer_id").size().reindex(cal.index).fillna(0)
    actual_rev = hold_daily.groupby("customer_id").order_value.sum().reindex(cal.index).fillna(0)

    pred_n = bg.expected_purchases(HOLDOUT_WEEKS, cal.x, cal.t_x, cal["T"])
    pred_rev = pred_n * gg.expected_value(cal.x.to_numpy(), cal.monetary.to_numpy())

    # how much of the actual holdout revenue sits in the customers we predicted as top 20%?
    top = pred_rev.rank(ascending=False, method="first") <= 0.2 * len(cal)
    m = {
        "calibration_end": str(CALIBRATION_END.date()),
        "holdout_weeks": round(HOLDOUT_WEEKS, 1),
        "customers": int(len(cal)),
        "purchases_actual": float(actual_n.sum()),
        "purchases_predicted": float(pred_n.sum()),
        "purchases_error_pct": float((pred_n.sum() / actual_n.sum() - 1) * 100),
        "purchases_mae_per_customer": float(np.abs(pred_n - actual_n).mean()),
        "revenue_actual": float(actual_rev.sum()),
        "revenue_predicted": float(pred_rev.sum()),
        "revenue_error_pct": float((pred_rev.sum() / actual_rev.sum() - 1) * 100),
        "revenue_spearman": float(spearmanr(pred_rev, actual_rev).statistic),
        "top20_revenue_capture_pct": float(actual_rev[top].sum() / actual_rev.sum() * 100),
    }

    # classic check: average predicted vs actual holdout purchases, by calibration frequency
    by_x = pd.DataFrame({"x": cal.x.clip(upper=10), "actual": actual_n, "predicted": pred_n}) \
             .groupby("x").mean()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(by_x.index, by_x.actual, "o-", color="black", label="actual")
    ax[0].plot(by_x.index, by_x.predicted, "s--", color="#1d4ed8", label="MBG/NBD predicted")
    ax[0].set(title=f"Holdout purchases ({HOLDOUT_WEEKS:.0f} weeks) by past repeat purchases",
              xlabel="repeat purchases before Jun 2011 (10 = 10+)", ylabel="avg purchases per customer")
    ax[0].legend()
    dec = pd.qcut(pred_rev.rank(method="first"), 10, labels=range(1, 11))
    by_dec = pd.DataFrame({"dec": dec, "actual": actual_rev, "predicted": pred_rev}).groupby("dec").sum()
    w = 0.4
    ax[1].bar(by_dec.index.astype(int) - w / 2, by_dec.actual / 1e3, w, color="black", label="actual")
    ax[1].bar(by_dec.index.astype(int) + w / 2, by_dec.predicted / 1e3, w, color="#1d4ed8", label="predicted")
    ax[1].set(title="Holdout revenue by predicted-value decile", xlabel="decile (10 = highest predicted)",
              ylabel="GBP thousands")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "clv_holdout.png", dpi=150); plt.close(fig)
    return m, bg, gg


def final_forecast(orders):
    full = rfm_summary(orders, DATA_END)
    bg, gg = MBGNBD().fit(full), GammaGamma().fit(full)
    x, t_x, T, mon = full.x.to_numpy(), full.t_x.to_numpy(), full["T"].to_numpy(), full.monetary.to_numpy()
    out = pd.DataFrame({
        "customer_id": full.index,
        "p_alive": bg.p_alive(x, t_x, T),
        "expected_orders_12m": bg.expected_purchases(FORECAST_WEEKS, x, t_x, T),
        "expected_order_value": gg.expected_value(x, mon),
    })
    out["clv_12m"] = out.expected_orders_12m * out.expected_order_value
    out["clv_decile"] = pd.qcut(out.clv_12m.rank(method="first"), 10, labels=range(1, 11)).astype(int)
    params = {"mbgnbd": {"r": bg.r, "alpha": bg.alpha, "a": bg.a, "b": bg.b, "converged": bg.converged},
              "gamma_gamma": {"p": gg.p, "q": gg.q, "v": gg.v, "converged": gg.converged,
                              "population_mean_order_value": gg.population_mean},
              "freq_monetary_corr": float(np.corrcoef(x[x > 0], mon[x > 0])[0, 1])}
    return out, params


def main():
    import json
    OUT.mkdir(exist_ok=True)
    orders = pd.read_csv(PROC / "orders.csv", parse_dates=["order_date"])

    metrics, bg, gg = validate(orders)
    print("Holdout validation:")
    for k, v in metrics.items():
        print(f"  {k:28s} {v:,.3f}" if isinstance(v, float) else f"  {k:28s} {v}")

    clv, params = final_forecast(orders)
    clv.to_csv(PROC / "clv.csv", index=False)
    json.dump({"holdout": metrics, "final_params": params}, open(OUT / "clv_metrics.json", "w"), indent=2)
    print("\nFinal model params:", json.dumps(params, indent=2))
    print(clv.describe().round(2).to_string())


if __name__ == "__main__":
    main()
