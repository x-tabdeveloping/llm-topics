import numpy as np


def most_important_words(
    pipeline, top_n: int = 15
) -> list[list[tuple[str, float]]]:
    _, vectorizer = pipeline.steps[0]
    _, topic_model = pipeline.steps[-1]
    components = topic_model.components_
    vocab = vectorizer.get_feature_names_out()
    highest = np.argpartition(-components, top_n)[:, :top_n]
    top_words = vocab[highest]
    importances = components[highest]
    res = []
    for top, imp in zip(top_words, importances):
        res.append(list(zip(top, imp)))
    return res


def infer_topic_names(pipeline, top_n: int = 4) -> list[str]:
    _, vectorizer = pipeline.steps[0]
    _, topic_model = pipeline.steps[-1]
    components = topic_model.components_
    vocab = vectorizer.get_feature_names_out()
    highest = np.argpartition(-components, top_n)[:, :top_n]
    top_words = vocab[highest]
    topic_names = []
    for i_topic, words in enumerate(top_words):
        name = "_".join(words)
        topic_names.append(f"{i_topic}_{name}")
    return topic_names
