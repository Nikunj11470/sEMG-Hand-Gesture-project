import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
print("shape:", df.shape)
print(df.head(3).to_string())
print("\ndtypes:\n", df.dtypes.to_string())
print("\nunique per column:\n", df.nunique().to_string())
print("\nNaNs:", int(df.isna().sum().sum()))
