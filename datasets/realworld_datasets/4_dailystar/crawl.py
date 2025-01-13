# stdlib
import time

# third party
from download_page import capture
import pandas as pd

urls = pd.read_csv("dailystar_daily_news.csv")["urls"].drop_duplicates()
urls = urls.sample(frac=1, random_state=1).head(1024).values

for url in urls:
    print(url)
    capture(url)
    time.sleep(0.1)
