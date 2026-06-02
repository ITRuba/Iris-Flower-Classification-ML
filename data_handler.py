import pandas as pd
from sklearn.datasets import load_iris

class DataHandler:
 def __init__(self):
    self.df = None


 def load_data(self):
    iris = load_iris()
    self.df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    self.df['species'] = iris.target
    print("Data loaded and converted to DataFrame successfully.")
    return self.df

 def show_summary(self):
    print("\n--- First 5 rows of the dataset ---")
    print(self.df.head())
    print("\n--- Dataset Information ---")
    print(self.df.info())

 def clean_data(self):
    print("\n--- Data Cleaning Process ---")
    missing = self.df.isnull().sum()
    print(f"Missing values per column:\n{missing}")
    if missing.any():
        self.df.dropna(inplace=True)
        print("Missing values found and removed.")
    else:
        print("Data is clean. No missing values found.")
    return self.df
