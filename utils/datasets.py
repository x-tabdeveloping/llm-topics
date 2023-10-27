from octis.dataset.dataset import Dataset

datasets = dict()
datasets["BBC News"] = Dataset()
datasets["BBC News"].fetch_dataset("BBC_News")
