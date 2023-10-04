from pathlib import Path

import joblib
import topicwizard.figures as figs
from llama_cpp import Llama
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

print("Loading data")
corpus = fetch_20newsgroups(
    remove=("headers", "footers", "quotes"),
    categories=["alt.atheism", "sci.space"],
).data
corpus = [text[:1000] for text in corpus]


prompt_template = """
### System:
You are an annotator that is tasked with summaizing texts.
You only respond with comma-separated keywords.
Please follow the users's instructions precisely.
### User:
Summarize the following piece of text with a couple of keywords:
```{X}```
### Assistant:
"""

print("Loading model")
beluga = Llama("generative_models/stablebeluga_7b.gguf", n_ctx=1024)


def extract_keywords(texts, prompt_template):
    for text in tqdm(texts):
        prompt = prompt_template.format(X=text)
        completion = beluga.create_completion(prompt, top_k=1, temperature=0.5)
        yield completion["choices"][0]["text"]


print("Preprocessing...")
preprocessed_corpus = list(extract_keywords(corpus, prompt_template))

print("Saving preprocessed data.")
with open("preprocessed.txt", "w") as f:
    f.write("\n\n".join(preprocessed_corpus))


with open("preprocessed.txt") as f:
    preprocessed_corpus = f.read().split("\n\n")

print("Fitting topic model...")
pipe = make_pipeline(CountVectorizer(stop_words="english"), NMF(10))
pipe.fit(preprocessed_corpus)

print("Saving model.")
out_dir = Path("topic_models")
out_dir.mkdir(exist_ok=True)
joblib.dump(pipe, out_dir.joinpath("beluga_nmf_10.joblib"))

print("Producing figures.")
figures_dir = Path("figures/llama_keyword_nmf")
figures_dir.mkdir(exist_ok=True, parents=True)
topic_map = figs.topic_map(preprocessed_corpus, pipeline=pipe)
topic_map.write_html(figures_dir.joinpath("topic_map.html"))
bars = figs.topic_barcharts(preprocessed_corpus, pipeline=pipe)
bars.write_html(figures_dir.joinpath("topic_barcharts.html"))
topic_wordclouds = figs.topic_wordclouds(preprocessed_corpus, pipeline=pipe)
topic_wordclouds.write_html(figures_dir.joinpath("topic_wordclouds.html"))
word_map = figs.word_map(preprocessed_corpus, pipeline=pipe)
word_map.write_html(figures_dir.joinpath("word_map.html"))
print("DONE")
