import pandas as pd

pd.set_option('display.max_columns', None)

df = pd.read_csv("../dataset_random_18k.csv")

df = df.drop("Unnamed: 0", axis=1)

df.to_csv("../Datasets/dataset_random_18k_clean.csv", index=False)
