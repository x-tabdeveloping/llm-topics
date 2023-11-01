import numpy as np
from bertopic import BERTopic
from octis.dataset.dataset import Dataset
from octis.models.model import AbstractModel
from sklearn.base import BaseEstimator, TransformerMixin


class SklearnModel(AbstractModel):
    def __init__(
        self,
        vectorizer: BaseEstimator,
        model: TransformerMixin,
    ):
        self.vectorizer = vectorizer
        self.model = model

    def train_model(
        self, dataset: Dataset, hyperparams=None, top_words=10
    ) -> dict:
        results = dict()
        corpus = dataset.get_corpus()
        texts = [" ".join(words) for words in corpus]
        embeddings = self.vectorizer.fit_transform(texts)
        results["topic-document-matrix"] = self.model.fit_transform(
            embeddings
        ).T
        results["topic-word-matrix"] = self.model.components_  # type: ignore
        vocab = self.vectorizer.get_feature_names_out()
        highest = np.argpartition(-self.model.components_, top_words)[
            :, :top_words
        ]
        top = vocab[highest]
        topics = []
        for _, words in enumerate(top):
            topics.append(list(words))
        results["topics"] = topics
        return results


class BERTopicModel(AbstractModel):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def train_model(
        self, dataset: Dataset, hyperparams=None, top_words=10
    ) -> dict:
        self.model = BERTopic(top_n_words=top_words, **self.kwargs)
        results = dict()
        corpus = dataset.get_corpus()
        texts = [" ".join(words) for words in corpus]
        self.model.fit(texts)
        results["topic-word-matrix"] = self.model.c_tf_idf_.todense()
        topics = []
        for topic_id in range(len(set(self.model.topics_)) - 1):
            topics.append([term for term, _ in self.model.get_topic(topic_id)])
        results["topics"] = topics
        results["topic-document-matrix"] = self.model.approximate_distribution(
            texts
        )
        return results
