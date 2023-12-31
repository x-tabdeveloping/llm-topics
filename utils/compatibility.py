import numpy as np
from bertopic import BERTopic
from contextualized_topic_models.models.ctm import CombinedTM, ZeroShotTM
from contextualized_topic_models.utils.data_preparation import (
    TopicModelDataPreparation,
)
from gensim.utils import tokenize
from hdbscan import HDBSCAN
from octis.dataset.dataset import Dataset
from octis.models.model import AbstractModel
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from top2vec import Top2Vec
from umap.umap_ import UMAP


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
    def __init__(self, hdbscan_args, umap_args, **kwargs):
        self.kwargs = kwargs
        self.hdbscan_args = hdbscan_args
        self.umap_args = umap_args

    def train_model(
        self, dataset: Dataset, hyperparams=None, top_words=10
    ) -> dict:
        umap_model = UMAP(**self.umap_args)
        hdbscan_model = HDBSCAN(**self.hdbscan_args)
        self.model = BERTopic(
            top_n_words=top_words,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            **self.kwargs,
        )
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
        self,
        nr_topics: int,
        sentence_transformer_name: str,
        kind: str,
        vectorizer_args,
    ):
        self.nr_topics = nr_topics
        self.sentence_transformer_name = sentence_transformer_name
        self.kind = kind
        self.vectorizer_args = vectorizer_args
        if self.kind not in ("zeroshot", "combined"):
            raise ValueError(f"CTM has to be zeroshot or combined, not {kind}")

    def preprocess(self, document: str, vocab) -> str:
        tokens = tokenize(document, lower=True, deacc=True)
        tokens = (token for token in tokens if token in vocab)
        return " ".join(tokens)

    def train_model(self, dataset: Dataset, hyperparams=None, top_words=10):
        results = dict()
        corpus = dataset.get_corpus()
        texts = [" ".join(words) for words in corpus]
        vocab = (
            CountVectorizer(**self.vectorizer_args)
            .fit(texts)
            .get_feature_names_out()
        )
        vocab = set(vocab)
        preprocessed_texts = [self.preprocess(text, vocab) for text in texts]
        qt = TopicModelDataPreparation(self.sentence_transformer_name)
        training_dataset = qt.fit(
            text_for_contextual=texts, text_for_bow=preprocessed_texts
        )
        transformer = SentenceTransformer(self.sentence_transformer_name)
        embedding_dimensionality = transformer.encode("something").shape[0]
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
        results["topic-word-matrix"] = ctm.get_topic_word_matrix()
        return results


class Top2VecModel(AbstractModel):
    def __init__(
        self, nr_topics: int, sentence_transformer_name: str, **kwargs
    ):
        self.nr_topics = nr_topics
        self.sentence_transformer_name = sentence_transformer_name
        self.model_kwargs = kwargs

    def train_model(self, dataset: Dataset, hyperparams=None, top_words=10):
        results = dict()
        corpus = dataset.get_corpus()
        texts = [" ".join(words) for words in corpus]
        model = Top2Vec(
            texts,
            embedding_model=self.sentence_transformer_name,
            **self.model_kwargs,
        )
        try:
            model.hierarchical_topic_reduction(self.nr_topics)
            reduced = True
        except Exception:
            print("Couldn't reduce number of topics.")
            reduced = False

        topic_words, _, _ = model.get_topics(reduced=reduced)
        topics = []
        for topic in topic_words:
            topics.append(topic[:top_words])
        results["topics"] = topics
        return results
