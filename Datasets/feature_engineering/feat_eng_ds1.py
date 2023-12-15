import pandas as pd
import numpy as np


df = pd.read_csv("Datasets/dataset_random_18k_clean.csv")

df['vmvf'] = (df['Pitch'] * df['Pitch'] / (np.pi * df['FuelRadius'] * df['FuelRadius'])) - 1 - (0.12 / df['FuelRadius']) - (0.0036 / (df['FuelRadius'] * df['FuelRadius']))

df.to_csv("Datasets/dataset_random_18k_clean_fe.csv", index=False)
