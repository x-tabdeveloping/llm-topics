from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from utils.compatibility import (
    BERTopicModel,
    ContextualizedTopicModel,
    SklearnModel,
    Top2VecModel,
)
from utils.eigen import EigenModel
from utils.gaussian_mixture import GMMTopicModel
from utils.keybert import KeyBertVectorizer

models = dict()
models["KeyBert"] = lambda n_topics: SklearnModel(
    KeyBertVectorizer(model="all-MiniLM-L6-v2"), NMF(n_topics)
)
models["Nmf"] = lambda n_topics: SklearnModel(
    CountVectorizer(min_df=10), NMF(n_topics)
)
models["Lda"] = lambda n_topics: SklearnModel(
    CountVectorizer(min_df=10),
    LatentDirichletAllocation(n_topics),
)
models["EigenModel"] = lambda n_topics: EigenModel(
    n_topics,
    sentence_transformer_name="all-MiniLM-L6-v2",
    vectorizer_args=dict(min_df=10),
)
models["BERTopic"] = lambda n_topics: BERTopicModel(
    nr_topics=n_topics,
    embedding_model="all-MiniLM-L6-v2",
    vectorizer_model=CountVectorizer(min_df=10),
    hdbscan_args=dict(
        min_cluster_size=10,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    ),
    umap_args=dict(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
    ),
)
models["ZeroShotTM"] = lambda n_topics: ContextualizedTopicModel(
    nr_topics=n_topics,
    sentence_transformer_name="all-MiniLM-L6-v2",
    kind="zeroshot",
    vectorizer_args=dict(min_df=10),
)
models["Top2Vec"] = lambda n_topics: Top2VecModel(
    nr_topics=n_topics,
    sentence_transformer_name="all-MiniLM-L6-v2",
    min_count=10,
    hdbscan_args=dict(
        min_cluster_size=10,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    ),
    umap_args=dict(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
    ),
)
models["GMM"] = lambda n_topics: GMMTopicModel(
    n_topics,
    sentence_transformer_name="all-MiniLM-L6-v2",
    vectorizer_args=dict(min_df=10),
)
