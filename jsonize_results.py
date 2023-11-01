import json
import pickle

with open("results.pkl", "rb") as in_file:
    results = pickle.load(in_file)

for run in results:
    run["model_output"].pop("topic-word-matrix")
    run["model_output"].pop("topic-document-matrix")

with open("results.json", "w") as out_file:
    json.dump(results, out_file)
