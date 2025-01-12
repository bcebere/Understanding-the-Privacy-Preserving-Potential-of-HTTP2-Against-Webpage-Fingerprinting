import numpy as np
import pandas as pd

df1 = pd.read_csv("reddit1.csv")
df2 = pd.read_csv("reddit2.csv")

df1 = df1[df1["url"].notna()]
df2 = df2[df2["url"].notna()]

print(df1)
print(df2)

urls = df1["url"].values.tolist() + df2["url"].values.tolist()
urls = np.unique(urls)

print(len(urls))

pd.Series(urls).to_csv("reddit_urls.csv", index=None)
