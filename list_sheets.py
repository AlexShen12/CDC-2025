import pandas as pd

df = pd.read_excel("data/Business.xlsx", sheet_name="Table 1", header=None)
print(df.head(20).to_string())