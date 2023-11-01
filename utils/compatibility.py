import numpy as np
from bertopic import BERTopic
from contextualized_topic_models.models.ctm import CombinedTM, ZeroShotTM
from contextualized_topic_models.utils.data_preparation import (
    TopicModelDataPreparation,
)
from gensim.utils import tokenize
from octis.dataset.dataset import Dataset
from octis.models.model import AbstractModel
from sentence_transformers.models import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


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


class ContextualizedTopicModel(AbstractModel):
    def __init__(
        self, nr_topics: int, sentence_transformer_name: str, kind: str
    ):
        self.nr_topics = nr_topics
        self.sentence_transformer_name = sentence_transformer_name
        self.kind = kind
        if self.kind not in ("zeroshot", "combined"):
            raise ValueError(f"CTM has to be zeroshot or combined, not {kind}")

    def preprocess(self, document: str) -> str:
        tokens = tokenize(document, lower=True, deacc=True)
        tokens = filter(lambda token: token not in ENGLISH_STOP_WORDS, tokens)
        return " ".join(tokens)

    def train_model(self, dataset: Dataset, hyperparams=None, top_words=10):
        results = dict()
        corpus = dataset.get_corpus()
        texts = [" ".join(words) for words in corpus]
        preprocessed_texts = [self.preprocess(text) for text in texts]
        qt = TopicModelDataPreparation(self.sentence_transformer_name)
        training_dataset = qt.fit(
            text_for_contextual=texts, text_for_bow=preprocessed_texts
        )
        transformer = SentenceTransformer(self.sentence_transformer_name)
        embedding_dimensionality = transformer.modules[
            -1
        ].word_embedding_dimension
        model_class = ZeroShotTM if self.kind == "zeroshot" else CombinedTM
        ctm = model_class(
            bow_size=len(qt.vocab),
            contextual_size=embedding_dimensionality,
            n_components=self.nr_topics,
        )
        ctm.fit(training_dataset)
        results["topics"] = ctm.get_topic_lists(top_words)
        results["topic-document-matrix"] = ctm.get_doc_topic_distribution(
            training_dataset
        )
        results["topic-word-matrix"] = ctm.get_topic_word_matrix(
            training_dataset
        )
        return results
