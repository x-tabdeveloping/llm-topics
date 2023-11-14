import pandas as pd
from tabulate import tabulate

dat = pd.read_csv("evaluation.csv", index_col=0)
dat = dat.drop(columns=["n_topics"])

for dataset, data in dat.groupby("dataset"):
    print()
    print("Dataset: ", dataset)
    print()
    print(
        tabulate(
            data.groupby("model")
            .mean(numeric_only=True)
            .reset_index()
            .sort_values("Diversity"),
            headers="keys",
            tablefmt="psql",
        )
    )
