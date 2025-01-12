import json
import random
import time

import pandas as pd
from download_page import capture

urls = []
for batch in range(1, 5):
    data = pd.read_csv(f"udemy{batch}.csv", low_memory=False)
    urls += data["url"].sample(frac=1, random_state=0).head(1024).values.tolist()

print(len(urls))
urls = urls[:1024]
for url in urls:
    print(url)
    for retry in range(3):
        try:
            capture(url)
            break
        except BaseException:
            continue
    time.sleep(0.5)
