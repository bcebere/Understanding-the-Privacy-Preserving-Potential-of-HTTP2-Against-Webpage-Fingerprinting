"""Optuna hyper-parameter search for the wfaudit attack models.

from wfaudit.helpers_ml.tuning import tune, load_best_params

tune("robustfp", X, y, n_trials=25, per_class=150, epoch_budget=60)
best = load_best_params("robustfp", X=X, y=y)
evaluate_multiclass("robustfp", label="mydata_tuned", data=X, labels=y, **best)
"""

# future
from __future__ import annotations

# stdlib
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, Optional

# third party
import numpy as np
import optuna
from optuna.pruners import MedianPruner, NopPruner
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

NN_ARCHS = {"varcnn", "holmes", "robustfp", "df"}


# --------------------------------------------------------------------------- #
# Search spaces
# --------------------------------------------------------------------------- #
def _space_robustfp(trial) -> Dict[str, Any]:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_categorical(
            "weight_decay", [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        ),
        "dropout_conv": trial.suggest_float("dropout_conv", 0.1, 0.5),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 200, 256]),
    }


def _space_df(trial) -> Dict[str, Any]:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_categorical(
            "weight_decay", [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        ),
        "dropout_cls": trial.suggest_float("dropout_cls", 0.3, 0.7),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 200, 256]),
    }


def _space_varcnn(trial) -> Dict[str, Any]:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_categorical(
            "weight_decay", [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        ),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 200, 256]),
    }


def _space_holmes(trial) -> Dict[str, Any]:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_categorical(
            "weight_decay", [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        ),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 200, 256]),
    }


def _space_xgboost(trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "eta": trial.suggest_float("eta", 0.02, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }


def _space_rf(trial) -> Dict[str, Any]:
    # The default max_depth of 4 is far too shallow for a 100-class problem.
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_categorical("max_depth", [1, 2, 3, 4, None]),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }


def _space_kfp(trial) -> Dict[str, Any]:
    # k-FP probabilities are neighbour-vote fractions, so their resolution is
    # 1/n_neighbours. Small k makes probability-derived metrics degenerate.
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "n_neighbours": trial.suggest_int("n_neighbours", 2, 15),
    }


def _space_lr(trial) -> Dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
        "scaler": trial.suggest_categorical("scaler", ["minmax", "standard"]),
    }


SPACES: Dict[str, Callable] = {
    "robustfp": _space_robustfp,
    "df": _space_df,
    "varcnn": _space_varcnn,
    "holmes": _space_holmes,
    "xgboost": _space_xgboost,
    "rf": _space_rf,
    "kfp": _space_kfp,
    "lr": _space_lr,
}


# Default configurations, enqueued as the first trial of each study.
# --------------------------------------------------------------------------- #
# Dataset and protocol identity
# --------------------------------------------------------------------------- #
def _digest_array(a: np.ndarray, h) -> None:
    """Fold shape, dtype and contents of ``a`` into ``h`` without copying it."""
    a = np.ascontiguousarray(a)
    h.update(f"{a.shape}|{a.dtype.str}|".encode())
    view = memoryview(a.reshape(-1)).cast("B")
    chunk = 1 << 24
    for i in range(0, len(view), chunk):
        h.update(view[i : i + chunk])


def dataset_fingerprint(X: np.ndarray, y: np.ndarray) -> str:
    """Content hash of a dataset, used to keep studies from different datasets apart.

    Independent of the hash in ``evaluation.py``, which keys the benchmark cache
    and must keep producing its existing values.
    """
    h = hashlib.blake2b(digest_size=6)
    _digest_array(np.asarray(X), h)
    _digest_array(np.asarray(y), h)
    return h.hexdigest()


def dataset_id(X: np.ndarray, y: np.ndarray, dataset_tag: Optional[str] = None) -> str:
    """Identifier combining an optional readable tag with the content hash."""
    fp = dataset_fingerprint(X, y)
    return f"{dataset_tag}-{fp}" if dataset_tag else fp


def _protocol_id(protocol: Dict[str, Any]) -> str:
    """Hash of the settings that determine what a trial value means.

    Trials scored under different subsample sizes, splits or epoch budgets are
    not comparable, so they belong to different studies.
    """
    blob = json.dumps(protocol, sort_keys=True, default=str)
    return hashlib.blake2b(blob.encode(), digest_size=3).hexdigest()


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def resolve_per_class(y, fraction: float = 0.3, minimum: int = 50) -> int:
    """Traces per class to search on: a fraction of what is available, floored.

    A fixed count silently becomes a no-op on smaller datasets, and a bare
    fraction can drop below the point where the attack trains at all, which
    moves the regularisation optimum away from the one that applies at full
    size. Taking a fraction but never going below ``minimum`` avoids both.

    The fraction is taken over the median class count, so a single undersized
    class does not shrink the whole search.
    """
    counts = np.bincount(LabelEncoder().fit_transform(np.asarray(y)))
    available = int(np.median(counts))
    return int(min(available, max(minimum, round(fraction * available))))


def stratified_subsample(
    X: np.ndarray, y: np.ndarray, per_class: Optional[int], seed=0
):
    """Keep every class, cap the number of samples per class."""
    if per_class is None:
        return X, y
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        if len(idx) > per_class:
            idx = rng.choice(idx, per_class, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return X[keep], y[keep]


def _normalise_3d(X_tr: np.ndarray, X_va: np.ndarray):
    """Per-channel standardisation with statistics from the training split only.

    Matches the normalisation applied in ``evaluate_classifier`` so that the
    selected configuration transfers to the benchmark.
    """
    if X_tr.ndim != 3:
        return X_tr, X_va
    mu = X_tr.mean(axis=(0, 2), keepdims=True)
    sigma = X_tr.std(axis=(0, 2), keepdims=True) + 1e-8
    return (X_tr - mu) / sigma, (X_va - mu) / sigma


def _build_model(arch: str, **params):
    # Imported lazily to avoid a circular import with evaluation.py.
    # wfaudit absolute
    from wfaudit.helpers_ml.evaluation import _get_arch_mode

    return _get_arch_mode(arch, **params)


class _PinnedTrial:
    """Trial wrapper returning fixed values instead of sampling them.

    Overwriting a sampled value after the fact leaves the sampled value in
    ``trial.params``, so the recorded configuration is not the one that was
    trained. Intercepting the suggestion keeps the record honest.
    """

    def __init__(self, trial, pinned: Dict[str, Any]):
        self._trial = trial
        self._pinned = pinned

    def __getattr__(self, name):
        attr = getattr(self._trial, name)
        if not name.startswith("suggest_"):
            return attr

        def suggest(param_name, *args, **kwargs):
            if param_name in self._pinned:
                return self._pinned[param_name]
            return attr(param_name, *args, **kwargs)

        return suggest


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
def tune(
    arch: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_trials: int = 15,
    timeout: Optional[float] = None,
    per_class: Optional[int] = 150,
    val_frac: float = 0.2,
    epoch_budget: int = 60,
    patience: int = 8,
    seed: int = 0,
    workspace: Path = Path("workspace"),
    dataset_tag: Optional[str] = None,
    study_name: Optional[str] = None,
    use_pruning: bool = True,
    extra_fixed: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Search hyper-parameters for ``arch``, maximising macro-F1 on a held-out split.

    Args:
        per_class: samples retained per class for the search; None uses all.
        val_frac: fraction of the subsample held out to score each trial.
        epoch_budget: epoch cap applied to neural-network trials.
        timeout: wall-clock limit in seconds. Studies are resumable.
        extra_fixed: parameters pinned across all trials.
        dataset_tag: readable prefix for the study and output file names. The
            content hash of ``(X, y)`` is appended regardless, so two datasets
            can never share a study even if the tag is omitted or reused.

    Returns:
        The search summary, also written to
        ``best_params_{arch}_{dataset_id}.json``.
    """
    if arch not in SPACES:
        raise ValueError(f"no search space defined for {arch}")

    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X)
    y = LabelEncoder().fit_transform(np.asarray(y))

    did = dataset_id(X, y, dataset_tag)
    protocol = {
        "per_class": per_class,
        "val_frac": val_frac,
        "seed": seed,
        "epoch_budget": epoch_budget if arch in NN_ARCHS else None,
        "extra_fixed": extra_fixed or {},
    }
    proto = _protocol_id(protocol)

    Xs, ys = stratified_subsample(X, y, per_class, seed=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(
        Xs, ys, test_size=val_frac, stratify=ys, random_state=seed
    )
    X_tr, X_va = _normalise_3d(X_tr, X_va)
    kept = int(np.bincount(ys).max())
    total = int(np.bincount(y).max())
    print(
        f"[tune:{arch}] dataset={did} protocol={proto} :: "
        f"{X_tr.shape[0]} train / {X_va.shape[0]} val samples, "
        f"{len(np.unique(ys))} classes, {kept}/{total} traces per class"
        + (" (all available)" if kept >= total else "")
    )

    is_nn = arch in NN_ARCHS
    extra_fixed = dict(extra_fixed or {})

    def protocol_kwargs() -> Dict[str, Any]:
        """Training budget for the search, which is not a hyper-parameter."""
        if not is_nn:
            return {}
        return {
            "epochs": epoch_budget,
            "patience": patience,
            "verbose": False,
            "random_state": seed,
        }

    def score(params: Dict[str, Any]) -> float:
        try:
            model = _build_model(arch, **params)
        except TypeError as exc:
            raise TypeError(
                f"{arch} does not accept one of {sorted(params)}; its __init__ is "
                f"missing the corresponding keyword arguments. Original: {exc}"
            ) from exc
        try:
            model.fit(X_tr, y_tr)
            preds = np.asarray(model.predict(X_va)).ravel()
        finally:
            model = None
        return f1_score(y_va, preds, average="macro", zero_division=0)

    def objective(trial: optuna.Trial) -> float:
        params = SPACES[arch](_PinnedTrial(trial, extra_fixed))
        params.update(extra_fixed)
        for key, value in protocol_kwargs().items():
            params.setdefault(key, value)

        if is_nn and use_pruning:

            def _cb(epoch: int, metrics: Dict[str, float]) -> None:
                trial.report(metrics["val_acc"], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            params["on_epoch_end"] = _cb

        try:
            return score(params)
        except optuna.TrialPruned:
            raise
        except ValueError as exc:
            # Configurations that cannot train at this subsample size are not
            # errors in the search; discard them and continue.
            if "no complete batches" in str(exc):
                raise optuna.TrialPruned() from exc
            raise
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                # third party
                import torch

                torch.cuda.empty_cache()
                raise optuna.TrialPruned() from exc
            raise

    # The reference is the model constructed with no hyper-parameters at all,
    # measured on the same split as every trial. Nothing about it passes through
    # optuna, so it cannot be altered by what a search space happens to allow.
    baseline_value = score({**protocol_kwargs(), **extra_fixed})
    print(f"[tune:{arch}] defaults score {baseline_value:.4f}", flush=True)

    pruner = (
        MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
        if (use_pruning and is_nn)
        else NopPruner()
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        pruner=pruner,
        storage=f"sqlite:///{workspace / 'hpo.db'}",
        study_name=study_name or f"{arch}_{did}_{proto}",
        load_if_exists=True,
    )

    identity = {"arch": arch, "dataset_id": did, "protocol": protocol}
    stored = study.user_attrs.get("identity")
    if stored is None:
        for key, value in identity.items():
            study.set_user_attr(key, value)
        study.set_user_attr("identity", identity)
    elif stored != identity:
        raise ValueError(
            f"study '{study.study_name}' was created for {stored} but this call "
            f"describes {identity}. Trial values from different datasets or search "
            f"protocols are not comparable; use a distinct study_name or workspace."
        )

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True)
    elapsed = time.time() - t0

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
        raise RuntimeError(
            f"no trial completed for '{arch}': {n_pruned} of {len(study.trials)} "
            f"were pruned. A common cause is a sampled batch size larger than the "
            f"training split ({int(len(X_tr) * 0.9)} samples here), which leaves no "
            f"complete batches. Raise per_class, or pin a batch size with "
            f"extra_fixed={{'batch_size': ...}}."
        )

    # A search that does not beat the untouched model has not found anything, so
    # the defaults are what gets used. Ties go to the defaults, which are the
    # published settings and need no justification.
    improved = study.best_value > baseline_value
    selected = {**(study.best_params if improved else {}), **extra_fixed}

    out = {
        "arch": arch,
        "dataset_id": did,
        "dataset_tag": dataset_tag,
        "protocol_id": proto,
        "study_name": study.study_name,
        "best_params": selected,
        "selected": "search" if improved else "defaults",
        "searched_params": study.best_params,
        "search_value_macro_f1": study.best_value,
        "fixed_params": extra_fixed,
        "best_value_macro_f1": study.best_value if improved else baseline_value,
        "baseline_value_macro_f1": baseline_value,
        "n_trials_run": len(study.trials),
        "n_pruned": sum(
            t.state == optuna.trial.TrialState.PRUNED for t in study.trials
        ),
        "tuning_samples_per_class": per_class,
        "epoch_budget": epoch_budget,
        "seed": seed,
        "elapsed_seconds": round(elapsed, 1),
    }
    path = workspace / f"best_params_{arch}_{did}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(
        f"[tune:{arch}] defaults {baseline_value:.4f} | search {study.best_value:.4f} "
        f"-> using {'search' if improved else 'defaults'} -> {path}",
        flush=True,
    )
    return out


def _resolve_dataset_id(
    arch: str,
    workspace: Path,
    X=None,
    y=None,
    dataset_tag: Optional[str] = None,
    did: Optional[str] = None,
) -> str:
    """Work out which stored configuration is meant.

    Preference order: an explicit id, then a fingerprint recomputed from the
    data, then a readable tag, then the sole file on disk. Ambiguity raises
    rather than resolving to an arbitrary choice, since loading another
    dataset's configuration produces a plausible but wrong benchmark.
    """
    if did is not None:
        return did

    prefix = f"best_params_{arch}_"
    available = sorted(Path(workspace).glob(f"{prefix}*.json"))
    if not available:
        raise FileNotFoundError(
            f"no tuning results for '{arch}' in {workspace}; run tune() first"
        )
    ids = [f.stem[len(prefix) :] for f in available]

    if X is not None and y is not None:
        # The hash is the identity; any tag is only a readable prefix on it.
        fp = dataset_fingerprint(
            np.asarray(X), LabelEncoder().fit_transform(np.asarray(y))
        )
        matches = [i for i in ids if i.endswith(fp)]
        if not matches:
            raise FileNotFoundError(
                f"no tuning result for '{arch}' matches this dataset (fingerprint "
                f"{fp}); available: {ids}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"this dataset was tuned under several tags: {matches}. Pass did "
                f"to select one."
            )
        return matches[0]

    if dataset_tag is not None:
        matches = [i for i in ids if i.startswith(f"{dataset_tag}-")]
        if not matches:
            raise FileNotFoundError(
                f"no tuning result for '{arch}' tagged '{dataset_tag}'; "
                f"available: {ids}"
            )
        if len(matches) > 1:
            raise ValueError(f"tag '{dataset_tag}' matches several results: {matches}")
        return matches[0]

    if len(ids) > 1:
        raise ValueError(
            f"{len(ids)} tuning results exist for '{arch}': {ids}. Pass X and y, "
            f"dataset_tag or did to select one."
        )
    return ids[0]


def load_best_params(
    arch: str,
    workspace: Path = Path("workspace"),
    X=None,
    y=None,
    dataset_tag: Optional[str] = None,
    did: Optional[str] = None,
) -> Dict[str, Any]:
    """Load the configuration selected for a given dataset.

    Passing the same ``X`` and ``y`` used for the benchmark is the safest form,
    since the identifier is recomputed from the data rather than assumed.
    Search-only keyword arguments are removed.
    """
    workspace = Path(workspace)
    did = _resolve_dataset_id(arch, workspace, X, y, dataset_tag, did)
    record = json.loads((workspace / f"best_params_{arch}_{did}.json").read_text())
    params = dict(record["best_params"])
    for k in ("on_epoch_end", "verbose"):
        params.pop(k, None)
    return params


def top_trials(
    arch: str,
    k: int = 3,
    workspace: Path = Path("workspace"),
    X=None,
    y=None,
    dataset_tag: Optional[str] = None,
    did: Optional[str] = None,
    study_name: Optional[str] = None,
) -> list[Dict[str, Any]]:
    """Return the ``k`` best completed configurations from an existing study.

    Regularisation optima depend on training-set size, so a configuration
    selected on a subsample is not guaranteed to be optimal on the full data.
    Refitting a shortlist at full size and selecting among those is a cheaper
    alternative to searching at full size.
    """
    workspace = Path(workspace)
    if study_name is None:
        did = _resolve_dataset_id(arch, workspace, X, y, dataset_tag, did)
        record = json.loads((workspace / f"best_params_{arch}_{did}.json").read_text())
        study_name = record["study_name"]

    study = optuna.load_study(
        study_name=study_name, storage=f"sqlite:///{workspace / 'hpo.db'}"
    )
    completed = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    completed.sort(key=lambda t: t.value, reverse=True)
    return [
        {"rank": i, "search_macro_f1": t.value, "params": dict(t.params)}
        for i, t in enumerate(completed[:k], start=1)
    ]
