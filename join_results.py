import json
import pickle
from glob import glob

from utils.datasets import datasets
from utils.models import models

files = glob("results/*.pkl")
done = set()
results = []


def is_valid(entry: dict) -> bool:
    return (entry["model"] in models) and (entry["dataset"] in datasets)


def identifier(entry: dict) -> tuple:
    return (entry["model"], entry["dataset"], entry["n_topics"])


for file in files:
    with open(file, "rb") as in_file:
        entries = pickle.load(in_file)
        for run in entries:
            if is_valid(run) and (identifier(run) not in done):
                done.add(identifier(run))
                results.append(run)

with open("joint_results.pkl", "wb") as out_file:
    pickle.dump(results, out_file)

with open("done.json", "w") as done_file:
    json.dump(list(done), done_file)
