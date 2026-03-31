# stdlib
import time

# third party
from download_page import capture
import pandas as pd

urls = pd.read_csv("bbc_burmese_news.csv")["link"].drop_duplicates()
urls = urls.sample(frac=1, random_state=1).head(1024).values

for url in urls:
    print(url)
    capture(url)
    time.sleep(0.1)
