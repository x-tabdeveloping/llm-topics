import pickle

import numpy as np
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
from umap.umap_ import UMAP

my_topics = [
    "Computer Graphics, Images, Files, Graphics Card, Monitor",
    "Operating Systems, Windows, Widgets, X Server, Linux, Unix",
    "Computer Hardware, IBM, etc.",
    "For Sale, Sales",
    "Cars, Motorcycles, Vehichles",
    "Hockey, Baseball, Sports, Games, Team, Players",
    "Cryptography, Spying, NSA, Espionage",
    "Electronics, Nuclear Power",
    "Networks, Internet, Telecommunications",
    "Medicine, Health, Obesity",
    "Education, Academia, Universities, Degree Programme",
    "Space, NASA",
    "Islam, Christianity, Atheism, Religion",
    "Guns, Firearms, Legislation",
    "Israeli-Palestinian Conflict, Zionism",
    "Yugoslav War, Armenian Genocide",
    "Sexuality, Homosexuality, Homophobia",
    "Court, Law, Judge, Jury",
    "Stop words: the, that, of, to, for, he, him, not, you, is etc.",
    "Meaningless acronyms: em, max, ax, ey, pl, bhj, etc.",
]

with open("joint_results.pkl", "rb") as in_file:
    results = pickle.load(in_file)

records = []
for run in results:
    if (run["dataset"] == "20 Newsgroups Dirty") and (run["n_topics"] == 20):
        topics = run["model_output"]["topics"][:20]
        for topic in topics:
            records.append(
                dict(
                    topic=", ".join(topic), model=run["model"], assigned=False
                )
            )

data = pd.DataFrame.from_records(records)
newsgroup_data = pd.DataFrame(
    dict(model="Assigned", topic=my_topics, assigned=True)
)
data = pd.concat((data, newsgroup_data))
model = SentenceTransformer("all-MiniLM-L6-v2")
data["embedding"] = data["topic"].map(model.encode)
embeddings = np.stack(data["embedding"])
x, y = (
    UMAP(n_components=2, n_neighbors=20, metric="cosine")
    .fit_transform(embeddings)
    .T
)
data["x"] = x
data["y"] = y
data["short_topic"] = data["topic"].map(lambda s: " ".join(s.split(", ")[:4]))
fig = px.scatter(
    data[~data.assigned],
    x="x",
    y="y",
    color="model",
    text="short_topic",
    template="plotly_white",
    hover_name="topic",
)
fig = fig.update_traces(
    textfont=dict(size=12, color="rgba(0,0,0,0.6)"), marker_size=16
)
for index, row in data[data.assigned].iterrows():
    fig = fig.add_annotation(
        x=row["x"],
        y=row["y"],
        text="<b>" + row["topic"],
        showarrow=False,
        font=dict(size=16),
    )
fig.show()

fig.write_html("qualitative_topics.html")
