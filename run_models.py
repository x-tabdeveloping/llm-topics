import json
import pickle
import time

from utils.datasets import datasets
from utils.models import models

try:
    with open("results.json", "r") as in_file:
        prev_results = json.load(in_file)
        done = set()
        for run in prev_results:
            done.add((run["model"], run["dataset"], run["n_topics"]))
    # with open("results.pkl", "rb") as in_file:
    #     # prev_results = pickle.load(in_file)
    #     done = set()
    #     for run in prev_results:
    #         done.add((run["model"], run["dataset"], run["n_topics"]))
except FileNotFoundError:
    print("No previous results found starting from scratch.")
    done = set()
    # prev_results = []


results = []
for model_name, model in models.items():
    print(f"Model: {model_name}")
    for dataset_name, dataset in datasets.items():
        print(f"Dataset: {dataset_name}")
        for n_topics in [10, 20, 30, 40, 50]:
            print(f"N Topics: {n_topics}")
            if (model_name, dataset_name, n_topics) in done:
                print("Already Done Previously, continuing...")
                continue
            topic_model = model(n_topics)
            start_time = time.time()
            model_output = topic_model.train_model(dataset)
            end_time = time.time()
            results.append(
                dict(
                    model=model_name,
                    dataset=dataset_name,
                    model_output=model_output,
                    n_topics=n_topics,
                    duration=end_time - start_time,
                )
            )
            with open("results.pkl", "wb") as out_file:
                pickle.dump(results, out_file)
