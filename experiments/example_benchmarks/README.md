## Dataset WF Audit Example

For the example audit, we provide an example dataset [here](workspace/data/). Extract the archive before running the scripts using

```bash
tar xvf undefended_1_amazon_traces.tar.zst
```

For other datasets, just replace the archive with other replayed dataset available in the [dataset repository](https://i62nextcloud.tm.kit.edu/public.php/dav/files/6ga8tgFyiXo4ZAf/?accept=zip), in the `http2_replayed_traces` folder.

For creating the ML datasets, run

```bash
python step1_create_datasets.py
```

For running the audit, run
```
python step2_audit.py
```

The script contains a simplified pool of evaluators; refer to [wfaudit docs](../../wfaudit/README.md) for the full collection of estimators.
