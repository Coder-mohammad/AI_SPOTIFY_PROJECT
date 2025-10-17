# Spotify Genre Segmentation Project
# Requirements: pandas numpy scikit-learn matplotlib
# Run: python spotify_genre_segmentation.py

import math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

# ---------- config ----------
DATA_PATH = "spotify_songs.csv"   # <-- set your CSV path
FIG_DIR = Path("figures"); FIG_DIR.mkdir(exist_ok=True)

# ---------- helpers ----------
def new_fig(title=None):
    plt.figure(figsize=(8,5))
    if title: plt.title(title)

def scatter_2d(emb, labels, title, fname):
    new_fig(title)
    plt.scatter(emb[:,0], emb[:,1], s=8, c=labels)
    plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
    plt.tight_layout(); plt.savefig(FIG_DIR / fname); plt.show()

# ---------- load ----------
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Columns:", list(df.columns))

# identifiers and numeric features
id_cols = [c for c in ["track_id","track_name","artist_name","playlist_name","playlist_genre","playlist_subgenre"] if c in df.columns]
candidate_feats = [
    "danceability","energy","key","loudness","mode","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo","duration_ms","time_signature","popularity"
]
num_cols = [c for c in candidate_feats if c in df.columns]
if not num_cols:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_like = {"year","month","day","id","ids"}
    num_cols = [c for c in num_cols if not any(k in c.lower() for k in drop_like)]
print("Using numeric features:", num_cols)

# ---------- clean ----------
df = df.dropna(axis=0, subset=num_cols, how="all").copy()
if "loudness" in df: df["loudness"] = df["loudness"].clip(-60, 5)
if "duration_ms" in df: df["duration_ms"] = df["duration_ms"].clip(30_000, 1_200_000)
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
print("Final shape:", df.shape)

# ---------- EDA ----------
print(df[id_cols + num_cols].describe(include='all') if id_cols else df[num_cols].describe())

for c in num_cols:
    new_fig(f"Histogram: {c}")
    plt.hist(df[c].values, bins=40)
    plt.xlabel(c); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(FIG_DIR / f"hist_{c}.png"); plt.show()

corr = df[num_cols].corr()
new_fig("Correlation matrix")
im = plt.imshow(corr.values, aspect='auto', interpolation='nearest')
plt.xticks(range(len(num_cols)), num_cols, rotation=90)
plt.yticks(range(len(num_cols)), num_cols)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout(); plt.savefig(FIG_DIR / "correlation_matrix.png"); plt.show()

# ---------- scale + reduce ----------
scaler = StandardScaler()
X = scaler.fit_transform(df[num_cols].values)

pca2 = PCA(n_components=2, random_state=42)
X_pca2 = pca2.fit_transform(X)
print("PCA2 variance:", pca2.explained_variance_ratio_.sum())

pca10 = PCA(n_components=min(10, X.shape[1]), random_state=42)
X_pca10 = pca10.fit_transform(X)
print("PCA10 variance:", pca10.explained_variance_ratio_.sum())

tsne = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_pca10)

# ---------- clustering ----------
def cluster_and_score(emb, label_tag):
    results = []
    for k in [4,6,8,10]:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        kl = km.fit_predict(emb)
        sil = silhouette_score(emb, kl) if len(set(kl))>1 else np.nan
        db  = davies_bouldin_score(emb, kl) if len(set(kl))>1 else np.nan
        results.append(("KMeans", k, sil, db, kl))
    for eps in [0.8, 1.0, 1.2]:
        dbs = DBSCAN(eps=eps, min_samples=10)
        dl = dbs.fit_predict(emb)
        if len(set(dl))>1:
            sil = silhouette_score(emb, dl); db = davies_bouldin_score(emb, dl)
        else:
            sil, db = np.nan, np.nan
        results.append(("DBSCAN", eps, sil, db, dl))
    for k in [4,6,8]:
        agg = AgglomerativeClustering(n_clusters=k)
        al = agg.fit_predict(emb)
        sil = silhouette_score(emb, al) if len(set(al))>1 else np.nan
        db  = davies_bouldin_score(emb, al) if len(set(al))>1 else np.nan
        results.append(("Agglo", k, sil, db, al))
    best = sorted(results, key=lambda x: (-(x[2] if not math.isnan(x[2]) else -999), (x[3] if not math.isnan(x[3]) else 999)))[0]
    algo, param, sil, db, labels = best
    print(f"Best on {label_tag}: {algo} param={param} silhouette={sil:.3f} DB={db:.3f}")
    return algo, param, labels

algo_pca2, param_pca2, labels_pca2 = cluster_and_score(X_pca2, "PCA-2D")
algo_tsne, param_tsne, labels_tsne = cluster_and_score(X_tsne, "t-SNE")

df["cluster_pca2"] = labels_pca2
df["cluster_tsne"] = labels_tsne

# ---------- visuals ----------
scatter_2d(X_pca2, df["cluster_pca2"].values, "Clusters on PCA-2D", "clusters_pca2.png")
scatter_2d(X_tsne, df["cluster_tsne"].values, "Clusters on t-SNE", "clusters_tsne.png")

if "playlist_genre" in df:
    print("Top genres:\n", df["playlist_genre"].value_counts().head(10))
    ct = pd.crosstab(df["cluster_tsne"], df["playlist_genre"])
    print(ct.head(20))
    new_fig("Cluster vs Playlist Genre (top 10)")
    sub = ct.iloc[:, :min(10, ct.shape[1])].values
    im = plt.imshow(sub, aspect='auto', interpolation='nearest')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("Playlist Genre (top 10)"); plt.ylabel("Cluster")
    plt.tight_layout(); plt.savefig(FIG_DIR / "cluster_vs_genre.png"); plt.show()

# ---------- centroid interpretation ----------
k = param_pca2 if isinstance(param_pca2, int) else 6
km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_pca2)
centroids = km.cluster_centers_
approx_centers = scaler.inverse_transform(pca2.inverse_transform(centroids))
centroid_df = pd.DataFrame(approx_centers, columns=num_cols)
print("\nApprox cluster centers in original feature space:")
print(centroid_df.round(3))

# ---------- simple recommender ----------
nn = NearestNeighbors(n_neighbors=11, metric="euclidean").fit(pca10.transform(X))
def recommend_by_track_name(query_name: str, top_k: int = 10):
    if "track_name" not in df: raise ValueError("track_name column not found")
    idx = df[df["track_name"].str.lower()==query_name.lower()].index
    if len(idx)==0:
        idx = df[df["track_name"].str.lower().str.contains(query_name.lower(), na=False)].index
    if len(idx)==0: raise ValueError("Track not found")
    i = idx[0]
    distances, indices = nn.kneighbors(pca10.transform(X)[i].reshape(1, -1), return_distance=True)
    cols = id_cols + num_cols if id_cols else num_cols
    recs = df.iloc[indices[0][1:]][cols].copy()
    recs["distance"] = distances[0][1:]
    return recs

# Example:
# print(recommend_by_track_name("Shape of You"))

# ---------- save ----------
OUT_PATH = "spotify_with_clusters.csv"
df.to_csv(OUT_PATH, index=False)
print("Wrote:", OUT_PATH)
