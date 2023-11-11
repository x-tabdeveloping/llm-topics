from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from octis.dataset.dataset import Dataset
from octis.models.model import AbstractModel
from sentence_transformers import SentenceTransformer
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.mixture import GaussianMixture

from utils.soft_ctf_idf import soft_ctf_idf


class GMMTopicModel(TransformerMixin, AbstractModel):
    def __init__(
        self,
        n_components: int,
        sentence_transformer_name: str = "all-MiniLM-L6-v2",
        vectorizer_args: Optional[Dict[str, Any]] = None,
    ):
        self.n_components = n_components
        self.sentence_transformer_name = sentence_transformer_name
        self.vectorizer_args = vectorizer_args

    def fit_transform(self, X: Iterable[str], y=None):
        vectorizer_args = self.vectorizer_args or dict(
            stop_words="english", max_features=8_000
        )
        self.vectorizer = CountVectorizer(**vectorizer_args)
        document_term_matrix = self.vectorizer.fit_transform(X)
        self.sentence_transformer = SentenceTransformer(
            self.sentence_transformer_name
        )
        embeddings = np.stack(self.sentence_transformer.encode(X))
        self.mixture = GaussianMixture(20)
        self.mixture.fit(embeddings)
        document_topic_matrix = self.mixture.predict_proba(embeddings)
        self.components_ = soft_ctf_idf(
            document_topic_matrix, document_term_matrix
        )
        return document_topic_matrix

    def fit(self, X: Iterable[str], y=None):
        self.fit_transform(X, y)
        return self

    def transform(self, corpus: Iterable[str]):
        embeddings = np.stack(self.sentence_transformer.encode(corpus))
        return self.mixture.predict_proba(embeddings)

    def get_top_k(self, top_n: int = 15) -> List[List[str]]:
        highest = np.argpartition(-self.components_, top_n)[:, :top_n]
        vocab = self.vectorizer.get_feature_names_out()
        top = vocab[highest]
        topics = []
        for words in top:
            topics.append(list(words))
        return topics

    def train_model(
        self, dataset: Dataset, hyperparams=None, top_words=10
    ) -> Dict:
        results = dict()
        corpus = dataset.get_corpus()
        texts = [" ".join(words) for words in corpus]
        results["topic-document-matrix"] = self.fit_transform(texts).T
        if hyperparams is None:
            hyperparams = dict()
        results["topic-word-matrix"] = self.components_
        results["topics"] = self.get_top_k(top_words, **hyperparams)
        return results
