import random

from llama_cpp import Llama
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
from tweetopic import DMM

corpus = fetch_20newsgroups(
    remove=("headers", "footers", "quotes"),
    categories=["alt.atheism", "sci.space"],
).data
corpus = random.sample(corpus, 100)
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

beluga = Llama("generative_models/stablebeluga_7b.gguf", n_ctx=1024)


def extract_keywords(texts, prompt_template):
    for text in tqdm(texts):
        prompt = prompt_template.format(X=text)
        completion = beluga.create_completion(prompt, top_k=1, temperature=0.5)
        yield completion["choices"][0]["text"]


preprocessed_corpus = list(extract_keywords(corpus, prompt_template))

with open("preprocessed.txt", "w") as f:
    f.write("\n\n".join(preprocessed_corpus))


pipe = make_pipeline(CountVectorizer(), DMM(10))
pipe.fit(preprocessed_corpus)
