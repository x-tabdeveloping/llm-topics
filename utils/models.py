from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from utils.compatibility import (BERTopicModel, ContextualizedTopicModel,
                                 SklearnModel, Top2VecModel)
from utils.eigen import EigenModel
from utils.keybert import KeyBertVectorizer

models = dict()
models["KeyBert"] = lambda n_topics: SklearnModel(
    KeyBertVectorizer(model="all-MiniLM-L6-v2"), NMF(n_topics)
)
models["Nmf"] = lambda n_topics: SklearnModel(CountVectorizer(), NMF(n_topics))
models["Lda"] = lambda n_topics: SklearnModel(
    CountVectorizer(),
    LatentDirichletAllocation(n_topics),
)
models["EigenModel"] = lambda n_topics: EigenModel(
    n_topics,
    sentence_transformer_name="all-MiniLM-L6-v2",
    vectorizer_args=dict(),
)
models["BERTopic"] = lambda n_topics: BERTopicModel(
    nr_topics=n_topics, embedding_model="all-MiniLM-L6-v2"
)
models["ZeroShotTM"] = lambda n_topics: ContextualizedTopicModel(
    nr_topics=n_topics,
    sentence_transformer_name="all-MiniLM-L6-v2",
    kind="zeroshot",
)
models["CombinedTM"] = lambda n_topics: ContextualizedTopicModel(
    nr_topics=n_topics,
    sentence_transformer_name="all-MiniLM-L6-v2",
    kind="combined",
)
models["Top2Vec"] = lambda n_topics: Top2VecModel(
    nr_topics=n_topics, sentence_transformer_name="all-MiniLM-L6-v2"
)
