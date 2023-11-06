from random import sample

import numpy as np
from ctransformers import AutoModelForCausalLM
from gensim.utils import tokenize
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from utils.datasets import datasets

bbc = datasets["BBC News"].get_corpus()
corpus = list(map(" ".join, bbc))

tfidf = TfidfVectorizer(stop_words="english")
dtm = tfidf.fit_transform(corpus)

importance = np.squeeze(np.asarray(dtm.sum(axis=0)))
highest = np.argsort(-importance)[:40]

for word in tfidf.get_feature_names_out()[highest]:
    print(word)


prompt = """
# Keywords: 'government', 'organisation', 'progress',
'list', 'service', 'parliament'
'manifesto'
# Topic:
Governance
# Keywords: {keywords}
# Topic:
"""

llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-v0.1-GGUF",
    model_file="mistral-7b-v0.1.Q2_K.gguf",
)

extractor = KeyBERT("all-MiniLM-L6-v2")
keywords = [extractor.extract_keywords(text, top_n=6) for text in tqdm(corpus)]

topic_labels = []
for _key in tqdm(keywords):
    keyword_str = ", ".join([f"'{keyword}'" for keyword, _ in _key])
    p = prompt.format(keywords=keyword_str)
    topic = llm(p, stop=["# Keywords:"]).removesuffix("# Keywords:")
    topic_labels.append(topic)

labels = [list(tokenize(topics, lower=True)) for topics in topic_labels]

for key, topic in zip(keywords, topic_labels):
    print("Keywords: ", key)
    print("Topics: ", topic)
    print("--------------------------------")
