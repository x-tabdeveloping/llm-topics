import json
import pickle
import time
from pathlib import Path

from utils.datasets import datasets
from utils.models import models

try:
    with open("done.json", "r") as done_file:
        done = json.load(done_file)
        done = set([tuple(entry) for entry in done])
except FileNotFoundError:
    print("No previous results found starting from scratch.")
    done = set()

timestamp = time.time_ns()

out_dir = Path("results")
out_dir.mkdir(exist_ok=True)

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
            with open(
                out_dir.joinpath(f"results_{timestamp}.pkl"), "wb"
            ) as out_file:
                pickle.dump(results, out_file)
