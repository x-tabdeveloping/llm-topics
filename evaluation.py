import pickle

import pandas as pd
import tabulate

from utils.datasets import datasets
from utils.metrics import metrics

try:
    previous_records = pd.read_csv("evaluation.csv", index_col=0).to_dict(
        "records"
    )
except FileNotFoundError:
    previous_records = []

done = set()
for record in previous_records:
    done.add((record["model"], record["dataset"], record["n_topics"]))


def evaluate_model(model_output, dataset, metrics: dict):
    results = dict()
    for name, metric in metrics.items():
        print(f" - {name}...")
        results[name] = metric(dataset).score(model_output)
    return results


with open("results.pkl", "rb") as in_file:
    results = pickle.load(in_file)

records = previous_records
for run in results:
    print("------------------------------")
    print("Model: ", run["model"])
    print("Dataset: ", run["dataset"])
    print("N topics: ", run["n_topics"])
    print("Evaluating on:")
    if (run["model"], run["dataset"], run["n_topics"]) in done:
        print("Already Done Previously, continuing...")
        continue
    dataset = datasets[run["dataset"]]
    eval_res = evaluate_model(run["model_output"], dataset, metrics)
    print("------------------------------")
    run.pop("model_output")
    records.append({**run, **eval_res})
    summary = pd.DataFrame.from_records(records)
    summary.to_csv("evaluation.csv")

print(tabulate.tabulate(summary, headers="keys", tablefmt="psql"))
