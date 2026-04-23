# stdlib
import time

import pandas as pd

# third party
from download_page import capture

urls = pd.read_csv("url_list.csv")
urls = urls.sample(frac=1, random_state=0).head(2048).values

for url in urls:
    print(url[0])
    capture(url[0])
    time.sleep(0.1)
