# Browser Crawlers

This section provides scripts for creating the resources  that can be replayed in the [experiments section](../experiments/example_replay).

First, install `playwright`
```bash
pip install -r requirements.txt
playwright install
sudo playwright install-deps
```

Run the crawling step
```bash
# Example using Wikipedia
cd 5_wiki
python crawl.py
```

This will generate a `data` which can be plugged into [experiments](../experiments/example_replay/) for replaying the website using various defenses.
