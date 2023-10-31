from octis.models.CTM import CTM
from octis.models.ETM import ETM
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from utils.compatibility import BERTopicModel, SklearnModel
from utils.eigen import EigenModel
from utils.keybert import KeyBertVectorizer

models = dict()
models["KeyBert"] = lambda n_topics: SklearnModel(
    KeyBertVectorizer(), NMF(n_topics)
)
models["NMF"] = lambda n_topics: SklearnModel(
    CountVectorizer(stop_words="english", max_features=8000), NMF(n_topics)
)
models["LDA"] = lambda n_topics: SklearnModel(
    CountVectorizer(stop_words="english", max_features=8000),
    LatentDirichletAllocation(n_topics),
)
models["Eigen"] = lambda n_topics: EigenModel(n_topics)
models["BERTopic"] = lambda n_topics: BERTopicModel(nr_topics=n_topics)
models["CTM"] = lambda n_topics: CTM(num_topics=n_topics)
models["ETM"] = lambda n_topics: ETM(num_topics=n_topics)
