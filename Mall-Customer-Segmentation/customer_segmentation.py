import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score

RANDOM_STATE = 42          
DATA_PATH = "Mall_Customers.csv"
FIG_DIR = "figures"        
os.makedirs(FIG_DIR, exist_ok=True)

INCOME = "Annual Income (k$)"
SCORE = "Spending Score (1-100)"


df = pd.read_csv(DATA_PATH)

print("First 5 rows:")
print(df.head())

print("\nShape (rows, columns):", df.shape)

print("\nColumn names and data types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nNumber of duplicate rows:", df.duplicated().sum())
print("Duplicate CustomerIDs:", df["CustomerID"].duplicated().sum())

print("\nDescriptive statistics:")
print(df.describe().round(2))

print("\nGender counts:")
print(df["Gender"].value_counts())


X = df[[INCOME, SCORE]]

def iqr_outliers(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return series[(series < lower) | (series > upper)], lower, upper

for col in [INCOME, SCORE]:
    outliers, lower, upper = iqr_outliers(df[col])
    print(f"{col}: acceptable range [{lower:.1f}, {upper:.1f}] -> {len(outliers)} outlier(s)")
    if len(outliers) > 0:
        print(df.loc[outliers.index, ["CustomerID", INCOME, SCORE]].to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].boxplot(df[INCOME], vert=True)
axes[0].set_title("Annual Income (k$)")
axes[1].boxplot(df[SCORE], vert=True)
axes[1].set_title("Spending Score (1-100)")
for ax in axes:
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Outlier check")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_outlier_boxplots.png", dpi=150)
plt.show()


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaled feature means:", X_scaled.mean(axis=0).round(3))
print("Scaled feature std  :", X_scaled.std(axis=0).round(3))


plt.figure(figsize=(7, 4))
plt.hist(df[INCOME], bins=15, color="steelblue", edgecolor="white")
plt.title("Distribution of Annual Income")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Number of customers")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_income_distribution.png", dpi=150)
plt.show()

plt.figure(figsize=(7, 4))
plt.hist(df[SCORE], bins=15, color="darkorange", edgecolor="white")
plt.title("Distribution of Spending Score")
plt.xlabel("Spending Score (1-100)")
plt.ylabel("Number of customers")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_spending_score_distribution.png", dpi=150)
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(df[INCOME], df[SCORE], color="teal", alpha=0.7, edgecolor="white")
plt.title("Annual Income vs Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_income_vs_score_scatter.png", dpi=150)
plt.show()



k_values = range(2, 11)
rows = []

for k in k_values:
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = model.fit_predict(X_scaled)
    rows.append({
        "K": k,
        "Inertia": model.inertia_,
        "Silhouette": silhouette_score(X_scaled, labels),
    })

k_results = pd.DataFrame(rows)
k_results["Drop vs K-1 (%)"] = -k_results["Inertia"].pct_change() * 100
print(k_results.round(3).to_string(index=False))


x_norm = (k_results["K"] - k_results["K"].min()) / (k_results["K"].max() - k_results["K"].min())
y_norm = (k_results["Inertia"] - k_results["Inertia"].min()) / (k_results["Inertia"].max() - k_results["Inertia"].min())
distance_to_line = np.abs(x_norm + y_norm - 1) / np.sqrt(2)

best_k = int(k_results.loc[distance_to_line.idxmax(), "K"])
best_k_silhouette = int(k_results.loc[k_results["Silhouette"].idxmax(), "K"])
print("K at the elbow (max distance to line):", best_k)
print("K with the highest silhouette score  :", best_k_silhouette)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(k_results["K"], k_results["Inertia"], marker="o", color="navy")
axes[0].axvline(best_k, color="red", linestyle="--", label=f"Selected K = {best_k}")
axes[0].set_title("Elbow Method: Inertia vs K")
axes[0].set_xlabel("Number of clusters (K)")
axes[0].set_ylabel("Inertia")
axes[0].set_xticks(list(k_values))
axes[0].grid(alpha=0.3)
axes[0].legend()

axes[1].plot(k_results["K"], k_results["Silhouette"], marker="s", color="darkgreen")
axes[1].axvline(best_k, color="red", linestyle="--", label=f"Selected K = {best_k}")
axes[1].set_title("Silhouette Score vs K (cross-check)")
axes[1].set_xlabel("Number of clusters (K)")
axes[1].set_ylabel("Silhouette score")
axes[1].set_xticks(list(k_values))
axes[1].grid(alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_elbow_and_silhouette.png", dpi=150)
plt.show()


final_model = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE)
df["Cluster"] = final_model.fit_predict(X_scaled)

print(f"Final model: K = {best_k}, inertia = {final_model.inertia_:.2f}, "
      f"silhouette = {silhouette_score(X_scaled, df['Cluster']):.3f}")

print("\nCustomers per cluster:")
print(df["Cluster"].value_counts().sort_index())

print("\nDataframe with cluster labels:")
print(df.head(10))


centroids = scaler.inverse_transform(final_model.cluster_centers_)
colors = plt.cm.tab10.colors

plt.figure(figsize=(9, 6))
for c in range(best_k):
    members = df[df["Cluster"] == c]
    plt.scatter(members[INCOME], members[SCORE], s=60, color=colors[c],
                edgecolor="white", alpha=0.85, label=f"Cluster {c}")

plt.scatter(centroids[:, 0], centroids[:, 1], s=250, c="black", marker="X",
            edgecolor="white", linewidth=1.5, label="Centroids")

plt.title(f"K-Means Customer Clusters (K = {best_k})")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_kmeans_clusters.png", dpi=150)
plt.show()


summary = df.groupby("Cluster").agg(
    Customers=("CustomerID", "count"),
    Avg_Income=(INCOME, "mean"),
    Avg_Spending_Score=(SCORE, "mean"),
    Avg_Age=("Age", "mean"),
    Pct_Female=("Gender", lambda s: (s == "Female").mean() * 100),
).round(1)
summary["Pct_of_Customers"] = (summary["Customers"] / len(df) * 100).round(1)

def level(value, series):
    if value < series.quantile(0.25):
        return "Low"
    if value > series.quantile(0.75):
        return "High"
    return "Moderate"

names = {
    c: f"{level(row.Avg_Income, df[INCOME])} Income / {level(row.Avg_Spending_Score, df[SCORE])} Spending"
    for c, row in summary.iterrows()
}
assert len(set(names.values())) == len(names), "Two clusters got the same name - review the naming rule."

summary.insert(0, "Segment", pd.Series(names))
df["Segment"] = df["Cluster"].map(names)

print(summary.sort_values("Avg_Spending_Score", ascending=False).to_string())

plt.figure(figsize=(9.5, 6))
for c in range(best_k):
    members = df[df["Cluster"] == c]
    plt.scatter(members[INCOME], members[SCORE], s=60, color=colors[c],
                edgecolor="white", alpha=0.85, label=f"{names[c]} (n={len(members)})")
plt.scatter(centroids[:, 0], centroids[:, 1], s=250, c="black", marker="X",
            edgecolor="white", linewidth=1.5, label="Centroids")

plt.title("Customer Segments (K-Means)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.grid(alpha=0.3)
plt.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/07_named_segments.png", dpi=150, bbox_inches="tight")
plt.show()


avg_spend = summary.set_index("Segment")["Avg_Spending_Score"].sort_values(ascending=False)
print(avg_spend.round(1))
print(f"\nOverall average spending score: {df[SCORE].mean():.1f}")

plt.figure(figsize=(9, 5))
bars = plt.bar(avg_spend.index, avg_spend.values, color=plt.cm.viridis(np.linspace(0.85, 0.15, len(avg_spend))))
plt.axhline(df[SCORE].mean(), color="red", linestyle="--", label="Overall average")
for bar, value in zip(bars, avg_spend.values):
    plt.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}", ha="center", fontweight="bold")

plt.title("Average Spending Score by Customer Segment")
plt.xlabel("Segment")
plt.ylabel("Average Spending Score (1-100)")
plt.xticks(rotation=20, ha="right")
plt.ylim(0, 100)
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/08_avg_spending_by_segment.png", dpi=150)
plt.show()


grid_rows = []
for eps in [0.25, 0.30, 0.35, 0.40, 0.50]:
    for min_samples in [3, 5, 7]:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
        grid_rows.append({
            "eps": eps,
            "min_samples": min_samples,
            "Clusters": len(set(labels)) - (1 if -1 in labels else 0),
            "Noise points": int((labels == -1).sum()),
        })

print(pd.DataFrame(grid_rows).to_string(index=False))

EPS, MIN_SAMPLES = 0.35, 5
df["DBSCAN_Cluster"] = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(X_scaled)

n_clusters = df["DBSCAN_Cluster"].nunique() - (1 if -1 in df["DBSCAN_Cluster"].values else 0)
n_noise = (df["DBSCAN_Cluster"] == -1).sum()
print(f"DBSCAN (eps={EPS}, min_samples={MIN_SAMPLES}): {n_clusters} clusters, "
      f"{n_noise} noise points ({n_noise / len(df) * 100:.1f}%)")

print("\nDBSCAN cluster profile (-1 = noise):")
print(df.groupby("DBSCAN_Cluster").agg(
    Customers=("CustomerID", "count"),
    Avg_Income=(INCOME, "mean"),
    Avg_Spending_Score=(SCORE, "mean"),
).round(1))

plt.figure(figsize=(9, 6))
for label in sorted(df["DBSCAN_Cluster"].unique()):
    members = df[df["DBSCAN_Cluster"] == label]
    if label == -1:
        plt.scatter(members[INCOME], members[SCORE], color="black", marker="x", s=60,
                    label=f"Noise (n={len(members)})")
    else:
        plt.scatter(members[INCOME], members[SCORE], s=60, color=colors[label % 10],
                    edgecolor="white", alpha=0.85, label=f"Cluster {label} (n={len(members)})")

plt.title(f"DBSCAN Clusters (eps = {EPS}, min_samples = {MIN_SAMPLES})")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.grid(alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/09_dbscan_clusters.png", dpi=150)
plt.show()

print("Cross-tab: K-Means cluster (rows) vs DBSCAN cluster (columns, -1 = noise)")
print(pd.crosstab(df["Segment"], df["DBSCAN_Cluster"]))

ari = adjusted_rand_score(df["Cluster"], df["DBSCAN_Cluster"])
print(f"\nAdjusted Rand Index between the two labelings: {ari:.3f} (1.0 = identical)")


df.to_csv("Mall_Customers_clustered.csv", index=False)
print("Saved: Mall_Customers_clustered.csv")
