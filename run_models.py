import pickle

from octis.dataset.dataset import Dataset

from utils.models import models

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
            model_output = topic_model.train_model(dataset)
            results.append(
                dict(
                    model=model_name,
                    dataset=dataset_name,
                    model_output=model_output,
                    n_topics=n_topics,
                )
            )

with open("results.pkl", "wb") as out_file:
    pickle.dump(results, out_file)
