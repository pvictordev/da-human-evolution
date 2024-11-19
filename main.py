import pandas as pd
import warnings

data = pd.read_csv('data/human-evolution_dataset.csv')

# Data Import, Analysis and Preprocessing
print(data.info())
print(data.duplicated().sum())
print(data["Genus_&_Specie"].unique())

