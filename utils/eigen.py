from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from octis.dataset.dataset import Dataset
from octis.models.model import AbstractModel
from sentence_transformers import SentenceTransformer
from sklearn.base import TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def maximum_marginal_relevance(
    component, word_embeddings, top_n: int, diversity: float
):
    results = [np.argmax(component)]
    n_candidates = min(5 * top_n, word_embeddings.shape[0])
    candidates = list(np.argpartition(-component, n_candidates)[:n_candidates])
    candidates.remove(results[0])
    for _ in range(top_n):
        candidate_sim = component[candidates]
        target_sim = cosine_similarity(
            word_embeddings[candidates], word_embeddings[results]
        )
        marginal_relevance = (1 - diversity) * candidate_sim - (
            diversity
        ) * np.max(target_sim, axis=1)
        best = candidates[np.argmax(marginal_relevance)]
        results.append(best)
        candidates.remove(best)
    return results


class EigenModel(TransformerMixin, AbstractModel):
    def __init__(
        self,
        n_components: int,
        sentence_transformer_name: str = "all-MiniLM-L6-v2",
        vectorizer_args: Optional[Dict[str, Any]] = None,
        specificity: float = 1,
    ):
        self.n_components = n_components
        self.sentence_transformer_name = sentence_transformer_name
        self.vectorizer_args = vectorizer_args
        self.specificity = specificity

    def fit_transform(self, X: Iterable[str], y=None):
        self.svd = TruncatedSVD(self.n_components)
        vectorizer_args = self.vectorizer_args or dict(
            stop_words="english", max_features=8_000
        )
        self.vectorizer = CountVectorizer(**vectorizer_args)
        self.sentence_transformer = SentenceTransformer(
            self.sentence_transformer_name
        )
        embeddings = np.stack(self.sentence_transformer.encode(X))
        doc_topic_matrix = self.svd.fit_transform(embeddings)
        self.eigenvectors = self.svd.components_
        self.vocab = self.vectorizer.fit(X).get_feature_names_out()
        self.freq = np.squeeze(
            np.asarray(self.vectorizer.transform(X).sum(axis=0))
        )
        self.word_embeddings = np.stack(
            self.sentence_transformer.encode(self.vocab)
        )
        self.components_ = cosine_similarity(
            self.eigenvectors, self.word_embeddings
        )
        return doc_topic_matrix

    def fit(self, X: Iterable[str], y=None):
        self.fit_transform(X, y)
        return self

    def transform(self, corpus: Iterable[str]):
        embeddings = np.stack(self.sentence_transformer.encode(corpus))
        return self.svd.transform(embeddings)

    def get_top_k(
        self, top_n: int = 15, diversity: float = 0.2
    ) -> List[List[str]]:
        results = []
        for component in self.components_:
            top_words = maximum_marginal_relevance(
                component,
                self.word_embeddings,
                top_n=top_n,
                diversity=diversity,
            )
            results.append(list(self.vocab[top_words]))
        return results

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
