import pickle

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from utils.datasets import datasets

with open("joint_results.pkl", "rb") as in_file:
    results = pickle.load(in_file)

records = []
for run in results:
    if run["dataset"] == "20 Newsgroups Dirty":
        n_stopwords = 0
        n_total = 0
        for topic in run["model_output"]["topics"]:
            for word in topic:
                n_total += 1
                if word in ENGLISH_STOP_WORDS:
                    n_stopwords += 1
        records.append(
            dict(
                n_topics=run["n_topics"],
                model=run["model"],
                stopword_frequency=n_stopwords / n_total,
            )
        )

data = pd.DataFrame.from_records(records)

fig = px.line(
    data,
    x="n_topics",
    y="stopword_frequency",
    color="model",
    markers=True,
    template="plotly_white",
    width=800,
    height=800,
)
fig = fig.update_traces(
    marker=dict(size=12, line=dict(width=2, color="black")), line=dict(width=2)
)
fig = fig.update_xaxes(title="Number of Topics")
fig = fig.update_yaxes(title="Frequency of Stop Words in Topics")
fig.write_image("figures/stop_words.png", scale=2)

dataset = datasets["20 Newsgroups Dirty"].get_corpus()
corpus = list(map(" ".join, dataset))
vectorizer = CountVectorizer()
freq = vectorizer.fit_transform(corpus).sum(axis=0)
freq = np.squeeze(np.asarray(freq))
freq_mapping = {
    word: frequency
    for word, frequency in zip(vectorizer.get_feature_names_out(), freq)
}
records = []
for run in results:
    if run["dataset"] == "20 Newsgroups Dirty":
        for topic in run["model_output"]["topics"]:
            for word in topic:
                records.append(
                    dict(
                        model=run["model"],
                        frequency=freq_mapping[word],
                    )
                )

data = pd.DataFrame(records)
mean_freq = data.groupby("model").mean().sort_values("frequency").reset_index()
fig = px.bar(
    mean_freq,
    x="frequency",
    y="model",
    template="plotly_white",
    width=800,
    height=800,
    orientation="h",
)
fig = fig.update_xaxes(title="Mean Frequency of Words in Topics")
fig = fig.update_yaxes(title="Model")
fig.write_image("figures/word_frequency.png", scale=2)
