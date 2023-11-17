import numpy as np
import pandas as pd
import plotly.express as px
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from umap.umap_ import UMAP

from utils.datasets import datasets

newsgroups = datasets["20 Newsgroups Dirty"].get_corpus()
newsgroups = [" ".join(tokens) for tokens in newsgroups]


hdbscan_args = dict(
    # min_cluster_size=10,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)
umap_args = dict(
    # n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
)

trf = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = np.stack([trf.encode(text) for text in tqdm(newsgroups)])

neighbor_counts = [5, 10, 15, 20, 25, 30, 35, 40, 45]
min_cluster_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45]
records = []
for n_neighbors in tqdm(neighbor_counts):
    umap = UMAP(**umap_args, n_neighbors=n_neighbors)
    reduced = umap.fit_transform(embeddings)
    for min_cluster_size in min_cluster_sizes:
        hdbscan = HDBSCAN(**hdbscan_args, min_cluster_size=min_cluster_size)
        hdbscan.fit(reduced)
        uniques = np.unique(hdbscan.labels_)
        records.append(
            dict(
                n_neighbors=n_neighbors,
                min_cluster_size=min_cluster_size,
                n_topics=len(uniques),
            )
        )
stability = pd.DataFrame.from_records(records)

stability.to_csv("stability.csv")

fig = px.scatter(
    stability,
    x="n_neighbors",
    y="min_cluster_size",
    color="n_topics",
    size="n_topics",
    template="plotly_white",
    size_max=60,
    height=800,
    width=800,
)
fig.write_image("figures/umap_hdbscan_stability.png", scale=2)
