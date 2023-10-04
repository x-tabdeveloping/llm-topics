from typing import Iterable

import numpy as np
from keybert import KeyBERT
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from tqdm import tqdm


class KeyBertVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        top_n: int = 25,
        zero_cutoff: bool = True,
    ):
        self.model = model
        self.top_n = top_n
        self.keybert = KeyBERT(model)
        self.vectorizer = DictVectorizer()
        self.zero_cutoff = zero_cutoff

    def stream_keywords(self, X: Iterable[str]) -> Iterable[dict[str, float]]:
        for text in X:
            yield dict(self.keybert.extract_keywords(text, top_n=self.top_n))

    def fit(self, X, y=None):
        self.vectorizer.fit(self.stream_keywords(tqdm(X)))
        return self

    def transform(self, X: Iterable[str]):
        dtm = self.vectorizer.transform(self.stream_keywords(tqdm(X)))
        if self.zero_cutoff:
            dtm[dtm < 0] = 0
        return dtm

    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()
