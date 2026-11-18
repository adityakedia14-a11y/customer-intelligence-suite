"""
Step 3: CUSTOMER SEGMENTATION (RFM + K-Means).

  * RFM = Recency (days since last order), Frequency (number of orders), Monetary (total spend).
  * All three are very skewed (a few customers spend 100x the median), so we log them
    and standardise before clustering - otherwise K-Means would only "see" the big spenders.
  * k is chosen with the silhouette score and the elbow in inertia. k=4 and k=5 separate
    almost equally well (0.37 vs 0.35); we use k=5 because it isolates the small group of
    top customers that drives most revenue, which k=4 merges with ordinary loyal customers.
  * Clusters are named by rule from their profile, so names stay the same on every run.
  * We also add classic 1-5 RFM scores (quintiles), which business users often expect.

Run: python src/segment.py     (needs data/processed/features_score.csv)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC, OUT = ROOT / "data" / "processed", ROOT / "outputs"
RFM = ["recency_days", "frequency", "monetary_total"]
K = 5
SEGMENT_ORDER = ["Champions", "Loyal", "New / Promising", "At Risk", "Lost"]


def name_clusters(profile: pd.DataFrame) -> dict:
    """profile: median R, F, M per cluster -> {cluster: name}."""
    names = {}
    recent = profile[profile.recency_days < 90].sort_values("frequency", ascending=False)
    lapsed = profile[profile.recency_days >= 90].sort_values("frequency", ascending=False)
    for cl, name in zip(recent.index, ["Champions", "Loyal", "New / Promising"]):
        names[cl] = name
    for cl, name in zip(lapsed.index, ["At Risk", "Lost"]):
        names[cl] = name
    if len(names) != K:
        raise ValueError(f"Unexpected cluster shapes, check the profile:\n{profile}")
    return names


def rfm_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Classic 1-5 scores. Recency is reversed: more recent = higher score."""
    rank = lambda s: s.rank(method="first")
    df["r_score"] = pd.qcut(rank(-df.recency_days), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df["f_score"] = pd.qcut(rank(df.frequency), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df["m_score"] = pd.qcut(rank(df.monetary_total), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df["rfm_score"] = df.r_score.astype(str) + df.f_score.astype(str) + df.m_score.astype(str)
    return df


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(PROC / "features_score.csv")
    X = StandardScaler().fit_transform(np.log1p(df[RFM].clip(lower=0)))

    # ---- choose k ----
    rows = []
    for k in range(2, 9):
        km = KMeans(k, n_init=10, random_state=42).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_)})
    k_table = pd.DataFrame(rows)
    k_table.to_csv(OUT / "segment_k_selection.csv", index=False)
    print(k_table.round(3).to_string(index=False))

    # ---- final clustering ----
    km = KMeans(K, n_init=10, random_state=42).fit(X)
    df["cluster"] = km.labels_
    names = name_clusters(df.groupby("cluster")[RFM].median())
    df["segment"] = df.cluster.map(names)
    df = rfm_scores(df)

    summary = (df.groupby("segment")
                 .agg(customers=("customer_id", "size"),
                      median_recency_days=("recency_days", "median"),
                      median_orders=("frequency", "median"),
                      median_spend=("monetary_total", "median"),
                      total_revenue=("monetary_total", "sum"))
                 .reindex(SEGMENT_ORDER))
    summary["customer_share"] = summary.customers / summary.customers.sum()
    summary["revenue_share"] = summary.total_revenue / summary.total_revenue.sum()
    summary.to_csv(OUT / "segment_summary.csv")
    print("\n", summary.round(3).to_string())

    df[["customer_id", "segment", "r_score", "f_score", "m_score", "rfm_score"]].to_csv(
        PROC / "segments.csv", index=False)

    # ---- plots ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(k_table.k, k_table.silhouette, "o-")
    ax[0].axvline(K, color="grey", ls="--")
    ax[0].set(title="Choosing k: silhouette score", xlabel="k", ylabel="silhouette")
    colors = dict(zip(SEGMENT_ORDER, ["#1d4ed8", "#16a34a", "#a855f7", "#f59e0b", "#9ca3af"]))
    for seg in SEGMENT_ORDER:
        d = df[df.segment == seg]
        ax[1].scatter(d.recency_days, d.monetary_total, s=6, alpha=0.5, label=seg, color=colors[seg])
    ax[1].set(yscale="log", title="Segments: recency vs spend", xlabel="days since last order",
              ylabel="total spend (GBP, log)")
    ax[1].legend(fontsize=8, markerscale=2)
    fig.tight_layout(); fig.savefig(OUT / "segments.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
