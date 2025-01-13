# stdlib
import time

# third party
from download_page import capture
import pandas as pd

urls = pd.read_csv("wiki_dataset_10k.csv")
urls = urls.sample(frac=1, random_state=0).head(1024).values

for url in urls:
    print(url[0])
    capture(url[0])
    time.sleep(0.1)
