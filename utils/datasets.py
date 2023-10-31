import numpy as np
from gensim.utils import tokenize
from octis.dataset.dataset import Dataset
from sklearn.datasets import fetch_20newsgroups

datasets = dict()

datasets["BBC News"] = Dataset()
datasets["BBC News"].fetch_dataset("BBC_News")

datasets["20 Newsgroups Clean"] = Dataset()
datasets["20 Newsgroups Clean"].fetch_dataset("20NewsGroup")

newsgroups = fetch_20newsgroups(subset="all")
corpus = [
    list(tokenize(text, lowercase=True, deacc=True))
    for text in newsgroups.data
]
vocabulary = set([token for text in corpus for token in text])
datasets["20 Newsgroups Dirty"] = Dataset(
    corpus=corpus,
    vocabulary=list(vocabulary),
    labels=list(np.array(newsgroups.target_names)[newsgroups.target]),
)
