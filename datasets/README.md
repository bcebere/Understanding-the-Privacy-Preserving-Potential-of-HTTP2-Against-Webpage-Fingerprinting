# Datasets

This section provides scripts for creating the datasets in the paper.


## Browser Traces

[browser_crawlers](browser_crawlers) provides the scripts for recreating the browser traces.

The collected traces are available in the `browser_original_traces` folder in the [datasets repository](https://i62nextcloud.tm.kit.edu/public.php/dav/files/6ga8tgFyiXo4ZAf/?accept=zip).

## Replayed HTTP/2 Traces

The replayed traces using various HTTP/2 defenses are available in the `http2_replayed_traces` in the [datasets repository](https://i62nextcloud.tm.kit.edu/public.php/dav/files/6ga8tgFyiXo4ZAf/?accept=zip).

As a toy example, the code repository includes the undefended Amazon dataset in the [replays](replays) folder.
Further, the [experiments/example_benchmark](../experiments/example_benchmark) folder contains the scripts for evaluating this (or any other) replayed trace --- just replace the dataset.
