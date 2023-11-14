import pandas as pd
import plotly.express as px
from tabulate import tabulate

dat = pd.read_csv("evaluation.csv", index_col=0)
dat["Mean Coherence"] = (
    dat["Coherence"] + dat["Centroid Word Embedding Coherence"]
) / 2

fig = px.scatter(
    dat,
    x="Mean Coherence",
    y="Diversity",
    color="model",
    facet_col="dataset",
    template="plotly_white",
    width=1600,
    height=600,
)
fig.update_traces(marker=dict(size=12, line=dict(width=2, color="black")))
fig.write_image("figures/coherence_diversity_scatter.png", scale=2)


mean_score = (
    dat.drop(columns="dataset")
    .groupby("model")
    .mean()[["Diversity", "Mean Coherence"]]
)
mean_score["Average"] = mean_score.mean(axis=1)
mean_score = mean_score.sort_values("Average", ascending=False)
print(
    tabulate(
        mean_score,
        headers="keys",
        tablefmt="psql",
    )
)
