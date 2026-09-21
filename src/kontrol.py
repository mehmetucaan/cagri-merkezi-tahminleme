import pandas as pd, numpy as np, sklearn, xgboost, tensorflow as tf

print("pandas      ", pd.__version__)
print("scikit-learn", sklearn.__version__)
print("xgboost     ", xgboost.__version__)
print("tensorflow  ", tf.__version__)

d = pd.read_csv("data/processed/halifax_hi_2saat.csv", parse_dates=["periyot"])
print("\nveri:", d.shape)
print("aralik:", d.periyot.min(), "->", d.periyot.max())
print("sutunlar:", list(d.columns))