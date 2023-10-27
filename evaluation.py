import numpy as np
import pandas as pd
import tabulate
from octis.dataset.dataset import Dataset

from utils.metrics import metrics
from utils.models import models


def evaluate_model(model, dataset, metrics: dict):
    results = dict()
    for name in metrics.keys():
        results[name] = []
    for _ in range(3):
        model_output = model.train_model(dataset)
        for name, metric in metrics.items():
            results[name].append(metric(dataset).score(model_output))
    for name, runs in results.items():
        results[name] = np.mean(runs)
    return results


datasets = dict()
datasets["BBC News"] = Dataset()
datasets["BBC News"].fetch_dataset("BBC_News")

results = []
for model_name, model in models.items():
    print(f"Model: {model_name}")
    for dataset_name, dataset in datasets.items():
        print(f"Dataset: {dataset_name}")
        for n_topics in [10, 20, 30, 40, 50]:
            print(f"N Topics: {n_topics}")
            topic_model = model(n_topics)
            model_res = evaluate_model(topic_model, dataset, metrics)
            model_res["N Topics"] = n_topics
            model_res["Model"] = model_name
            results.append(model_res)

summary = pd.DataFrame.from_records(results)
summary.to_csv("evaluation.csv")

print(tabulate.tabulate(summary, headers="keys", tablefmt="psql"))
