from octis.evaluation_metrics.coherence_metrics import (Coherence,
                                                        WECoherenceCentroid)
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.topic_significance_metrics import KL_uniform

metrics = {
    "Diversity": lambda dataset: TopicDiversity(),
    "Coherence": lambda dataset: Coherence(texts=dataset.get_corpus()),
    "KL Uniform Significance": lambda dataset: KL_uniform(),
    "Centroid Word Embedding Coherence": lambda dataset: WECoherenceCentroid(),
}
