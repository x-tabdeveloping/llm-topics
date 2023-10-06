from typing import Iterable, Optional

import scipy.sparse as spr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import minmax_scale


class SparseWithText(spr.csr_array):
    """Compressed Sparse Row sparse array with a text attribute,
    this way the textual content of the sparse array can be
    passed down in a pipeline."""

    def __init__(self, *args, texts: Optional[Iterable[str]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if texts is None:
            self.texts = None
        else:
            self.texts = list(texts)


class LeakyCountVectorizer(CountVectorizer):
    """Leaky CountVectorizer class, that does essentially the exact same
    thing as scikit-learn's CountVectorizer, but returns a sparse
    array with the text attribute attached. (see SparseWithText)"""

    def fit_transform(self, raw_documents, y=None):
        res = super().fit_transform(raw_documents, y=y)
        return SparseWithText(res, texts=list(raw_documents))

    def transform(self, raw_documents):
        res = super().transform(raw_documents)
        return SparseWithText(res, texts=raw_documents)


class LeakyTfidf(TfidfVectorizer):
    """Leaky TfidfVectorizer class, that does essentially the exact same
    thing as scikit-learn's TfidfVectorizer, but returns a sparse
    array with the text attribute attached. (see SparseWithText)"""

    def fit_transform(self, raw_documents, y=None):
        res = super().fit_transform(raw_documents, y=y)
        return SparseWithText(res, texts=list(raw_documents))

    def transform(self, raw_documents):
        res = super().transform(raw_documents)
        return SparseWithText(res, texts=raw_documents)


class DecomposingTopicModel(BaseEstimator, TransformerMixin):
    def __init__(
        self, representation: BaseEstimator, decomposition: BaseEstimator
    ):
        self.representation = representation
        self.decomposition = decomposition

    def fit_transform(self, X: SparseWithText, y=None):
        vectors = self.representation.fit_transform(X.texts, y)
        topics = self.decomposition.fit_transform(vectors, y)
        self.components_ = LinearRegression().fit(X, topics).coef_
        self.components_ = minmax_scale(self.components_, axis=1)
        return topics

    def fit(self, X: SparseWithText, y=None):
        self.fit_transform(X, y)
        return self

    def transform(self, X: SparseWithText, y=None):
        vectors = self.representation.transform(X.texts)
        topics = self.decomposition.transform(vectors)
        return topics


class MixtureTopicModel(BaseEstimator, TransformerMixin):
    def __init__(self, representation: BaseEstimator, mixture: BaseEstimator):
        self.representation = representation
        self.mixture = mixture

    def fit_transform(self, X: SparseWithText, y=None):
        vectors = self.representation.fit_transform(X.texts, y)
        self.mixture.fit(vectors)
        topics = self.mixture.predict_proba(vectors)
        self.components_ = LinearRegression().fit(X, topics).coef_
        self.components_ = minmax_scale(self.components_, axis=1)
        return topics

    def fit(self, X: SparseWithText, y=None):
        self.fit_transform(X, y)
        return self

    def transform(self, X: SparseWithText, y=None):
        vectors = self.representation.transform(X.texts)
        topics = self.mixture.predict_proba(vectors)
        return topics
