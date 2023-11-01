import json
import pickle

with open("results.pkl", "rb") as in_file:
    results = pickle.load(in_file)

records = []
for run in results:
    run["model_output"] = {"topics": run["model_output"]["topics"]}
    records.append(run)

with open("results.json", "w") as out_file:
    json.dump(records, out_file)
