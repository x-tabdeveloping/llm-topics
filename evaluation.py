import pickle

import pandas as pd
import tabulate

from utils.datasets import datasets
from utils.metrics import metrics


def evaluate_model(model_output, dataset, metrics: dict):
    results = dict()
    for name, metric in metrics.items():
        print(f" - {name}...")
        results[name] = metric(dataset).score(model_output)
    return results


with open("results.pkl", "b") as in_file:
    results = pickle.load(in_file)

records = []
for run in results:
    print("------------------------------")
    print("Model: ", run["model"])
    print("Dataset: ", run["dataset"])
    print("N topics: ", run["n_topics"])
    print("Evaluating on:")
    dataset = datasets[run["dataset"]]
    eval_res = evaluate_model(run["model_output"], dataset, metrics)
    print("------------------------------")
    run.pop("model_output")
    records.append({**run, **eval_res})

summary = pd.DataFrame.from_records(records)
summary.to_csv("evaluation.csv")

print(tabulate.tabulate(summary, headers="keys", tablefmt="psql"))
