from pathlib import Path

import joblib
import topicwizard.figures as figs
from embetter.text import SentenceEncoder
from sklearn.datasets import fetch_20newsgroups
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline

from utils.representation_based import LeakyCountVectorizer, MixtureTopicModel

print("Loading data")
corpus = fetch_20newsgroups(
    remove=("headers", "footers", "quotes"),
    categories=["alt.atheism", "sci.space"],
).data

print("Fitting topic model...")
representation = SentenceEncoder("all-MiniLM-L6-v2")
mixture = GaussianMixture(10)
model = MixtureTopicModel(representation, mixture)
pipe = make_pipeline(LeakyCountVectorizer(stop_words="english"), model)
pipe.fit(corpus)


print("Saving model.")
out_dir = Path("topic_models")
out_dir.mkdir(exist_ok=True)
joblib.dump(pipe, out_dir.joinpath("trf_gmm.joblib"))

print("Producing figures.")
figures_dir = Path("figures/trf_gmm")
figures_dir.mkdir(exist_ok=True, parents=True)
topic_map = figs.topic_map(corpus, pipeline=pipe)
topic_map.write_html(figures_dir.joinpath("topic_map.html"))
bars = figs.topic_barcharts(corpus, pipeline=pipe)
bars.write_html(figures_dir.joinpath("topic_barcharts.html"))
topic_wordclouds = figs.topic_wordclouds(corpus, pipeline=pipe)
topic_wordclouds.write_html(figures_dir.joinpath("topic_wordclouds.html"))
word_map = figs.word_map(corpus, pipeline=pipe)
word_map.write_html(figures_dir.joinpath("word_map.html"))

print("DONE")
