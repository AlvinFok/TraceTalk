import pandas as pd
from tabulate import tabulate

df = pd.read_csv("./usedVideos_info.csv")
df.index = df.index + 1

# print(df)
print(tabulate(df, headers="keys", tablefmt="psql"))
