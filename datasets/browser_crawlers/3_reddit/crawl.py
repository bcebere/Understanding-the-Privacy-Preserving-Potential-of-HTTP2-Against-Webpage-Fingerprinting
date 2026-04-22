# stdlib
import time

import pandas as pd

# third party
from download_page import capture

urls = pd.read_csv("reddit_urls.csv")
urls = urls.sample(frac=1, random_state=1).head(1024).values

for url in urls:
    print(url[0])
    capture(url[0])
    time.sleep(0.1)
