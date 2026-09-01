"""End-to-end checks for the tune -> evaluate -> report pipeline on toy data.

    pytest test_pipeline.py -v
    pytest test_pipeline.py -v -m "not slow"
"""

# stdlib
import ast
import importlib.util
import json
from pathlib import Path

# third party
import numpy as np
import pytest
from sklearn.datasets import make_classification

# wfaudit absolute
from wfaudit.helpers_ml.evaluation import (
    _array_hash,
    _get_arch_mode,
    _params_signature,
    classifier_metrics,
    evaluate_multiclass,
)
from wfaudit.helpers_ml.tuning import (
    SPACES,
    dataset_fingerprint,
    load_best_params,
    resolve_per_class,
    tune,
)

N_CLASSES = 8
N_PER_CLASS = 20

# k=10 and k=20 exceed the toy class count; sklearn warns and returns 1.0.
pytestmark = pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.UndefinedMetricWarning"
)


@pytest.fixture(scope="module")
def toy_tabular():
    """Learnable 2D feature matrix, the input shape kfp/xgboost/rf/lr expect."""
    X, y = make_classification(
        n_samples=N_CLASSES * N_PER_CLASS,
        n_features=24,
        n_informative=12,
        n_redundant=4,
        n_classes=N_CLASSES,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=0,
    )
    return X.astype(np.float32), y


@pytest.fixture(scope="module")
def toy_sequence():
    """Toy (N, 2, L) traces, the input shape the neural models expect."""
    rng = np.random.default_rng(0)
    n = N_CLASSES * N_PER_CLASS
    length = 128
    y = np.repeat(np.arange(N_CLASSES), N_PER_CLASS)
    t = np.linspace(0, 8 * np.pi, length)
    X = np.stack(
        [
            np.sin(t * (1 + y[:, None] * 0.35)) + rng.normal(0, 0.25, (n, length)),
            np.cos(t * (1 + y[:, None] * 0.35)) + rng.normal(0, 0.25, (n, length)),
        ],
        axis=1,
    )
    return X.astype(np.float32), y


def _tune_kwargs(tmp_path, **overrides):
    kwargs = dict(
        n_trials=3,
        per_class=12,
        val_frac=0.25,
        seed=0,
        workspace=tmp_path,
    )
    kwargs.update(overrides)
    return kwargs


# --------------------------------------------------------------------------- #
# Search spaces are consistent with the constructors they feed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arch", ["kfp", "xgboost", "rf", "lr"])
def test_defaults_build(arch):
    """The reference is the model with no arguments, so it must construct."""
    assert hasattr(_get_arch_mode(arch), "fit")


def test_sampled_params_are_accepted_by_constructor():
    """A sampled configuration must build, not just the hand-written baseline."""
    optuna = pytest.importorskip("optuna")
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    trial = study.ask()
    params = SPACES["kfp"](trial)
    model = _get_arch_mode("kfp", **params)
    assert model.n_neighbours == params["n_neighbours"]


# --------------------------------------------------------------------------- #
# Parameters must take effect, not merely be accepted
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "arch,param,value,probe",
    [
        ("rf", "n_estimators", 37, lambda m: m.model.n_estimators),
        ("rf", "max_depth", 9, lambda m: m.model.max_depth),
        ("lr", "class_weight", "balanced", lambda m: m.model.class_weight),
        ("lr", "scaler", "standard", lambda m: m.scaler_name),
        ("kfp", "n_estimators", 37, lambda m: m.forest.n_estimators),
        ("xgboost", "max_depth", 7, lambda m: m.model.max_depth),
    ],
)
def test_parameter_reaches_the_underlying_model(arch, param, value, probe):
    """A constructor that accepts a parameter and ignores it is worse than one
    that rejects it, because the search reports fold noise as improvement."""
    assert probe(_get_arch_mode(arch, **{param: value})) == value


def test_unknown_parameters_are_rejected_loudly():
    for arch in ("rf", "lr", "kfp"):
        with pytest.raises(TypeError):
            _get_arch_mode(arch, definitely_not_a_parameter=1)


# --------------------------------------------------------------------------- #
# Metric edge cases
# --------------------------------------------------------------------------- #
def test_top_k_survives_a_class_missing_from_the_fold():
    """Stratified folds keep every class at full size, but reduced-sample
    ablations and rare classes do not."""
    rng = np.random.default_rng(0)
    n_classes = 6
    y_test = np.array([0, 1, 2, 3, 4, 0, 1, 2])  # class 5 absent
    proba = rng.dirichlet(np.ones(n_classes), size=len(y_test))
    scores = classifier_metrics().score_proba(y_test, proba, list(range(n_classes)))
    assert 0.0 <= scores["acc_top5"] <= 1.0


def test_callable_parameters_are_refused_by_the_cache_key():
    """str() of a function embeds its address, which would silently disable the
    benchmark cache."""
    with pytest.raises(ValueError, match="callable"):
        _params_signature({"on_epoch_end": lambda e, m: None})


# --------------------------------------------------------------------------- #
# Every sampled parameter must be accepted by the constructor
# --------------------------------------------------------------------------- #
_ARCH_MODULES = {
    "robustfp": "wfaudit.helpers_ml.robustfp",
    "df": "wfaudit.helpers_ml.df",
    "varcnn": "wfaudit.helpers_ml.varcnn",
    "holmes": "wfaudit.helpers_ml.holmes",
    "xgboost": "wfaudit.helpers_ml.xgb",
    "rf": "wfaudit.helpers_ml.rf",
    "lr": "wfaudit.helpers_ml.lr",
    "kfp": "wfaudit.helpers_ml.kfpv2",
}


def _constructor_params(module_name):
    """Read the classifier's __init__ signature without importing the module.

    Parsing rather than importing keeps this check runnable where the deep
    learning dependencies are unavailable.
    """
    spec = importlib.util.find_spec(module_name)
    tree = ast.parse(Path(spec.origin).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.endswith("Classifier"):
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "__init__":
                    args = fn.args
                    names = {
                        a.arg for a in args.posonlyargs + args.args + args.kwonlyargs
                    } - {"self"}
                    return names, args.kwarg is not None
    raise AssertionError(f"no classifier __init__ found in {module_name}")


@pytest.mark.parametrize("arch", sorted(SPACES))
def test_search_space_matches_constructor(arch):
    """Catches a space and a model drifting apart without running either."""
    optuna = pytest.importorskip("optuna")
    accepted, accepts_kwargs = _constructor_params(_ARCH_MODULES[arch])

    produced = set()
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(30):
        produced |= set(SPACES[arch](study.ask()))

    missing = sorted(produced - accepted)
    assert not missing or accepts_kwargs, (
        f"{arch} search space produces {missing}, which its constructor neither "
        f"names nor forwards"
    )


# --------------------------------------------------------------------------- #
# Phase 1: search
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arch", ["kfp", "xgboost"])
def test_tune_writes_recoverable_params(arch, toy_tabular, tmp_path):
    X, y = toy_tabular
    result = tune(arch, X, y, **_tune_kwargs(tmp_path))

    written = tmp_path / f"best_params_{arch}_{result['dataset_id']}.json"
    assert written.exists()
    assert result["n_trials_run"] == 3
    assert 0.0 <= result["best_value_macro_f1"] <= 1.0

    best = load_best_params(arch, workspace=tmp_path, X=X, y=y)
    assert best == result["best_params"]
    assert set(best).issubset(set(json.loads(written.read_text())["best_params"]))

    # the recovered configuration must be usable by the benchmark
    assert hasattr(_get_arch_mode(arch, **best), "fit")


def test_tune_resumes_existing_study(toy_tabular, tmp_path):
    X, y = toy_tabular
    first = tune("kfp", X, y, **_tune_kwargs(tmp_path, n_trials=2))
    second = tune("kfp", X, y, **_tune_kwargs(tmp_path, n_trials=2))
    assert second["n_trials_run"] == first["n_trials_run"] + 2
    assert second["best_value_macro_f1"] >= first["best_value_macro_f1"]


def test_defaults_are_measured_as_the_reference(toy_tabular, tmp_path):
    """The reference is the untouched model, scored on the same split as trials."""
    X, y = toy_tabular
    result = tune("kfp", X, y, **_tune_kwargs(tmp_path, n_trials=2))
    assert result["baseline_value_macro_f1"] is not None
    assert 0.0 <= result["baseline_value_macro_f1"] <= 1.0


def test_resolve_per_class_scales_and_never_no_ops():
    """A fixed count silently uses everything on a smaller dataset; a bare
    fraction can fall below the size at which the attack trains."""

    def y_for(n):
        return np.repeat(np.arange(100), n)

    assert resolve_per_class(y_for(500), 0.3, 50) == 150  # fraction applies
    assert resolve_per_class(y_for(100), 0.3, 50) == 50  # floor applies
    assert resolve_per_class(y_for(40), 0.3, 50) == 40  # cap applies

    for n in (40, 100, 500):
        assert resolve_per_class(y_for(n), 0.3, 50) <= n

    # a single undersized class must not shrink the whole search
    lopsided = np.concatenate([np.repeat(np.arange(99), 100), np.repeat(99, 5)])
    assert resolve_per_class(lopsided, 0.3, 50) == 50


def test_tuning_keeps_every_class(toy_tabular):
    # wfaudit absolute
    from wfaudit.helpers_ml.tuning import stratified_subsample

    X, y = toy_tabular
    Xs, ys = stratified_subsample(X, y, per_class=5, seed=0)
    assert set(np.unique(ys)) == set(np.unique(y))
    assert np.bincount(ys).max() == 5


def test_pinned_parameters_are_not_sampled(toy_tabular, tmp_path):
    """The recorded configuration must be the one that was trained.

    Overwriting a suggested value after sampling leaves the sampled value in
    trial.params, so the stored result describes a configuration that was never
    evaluated.
    """
    X, y = toy_tabular
    result = tune(
        "kfp", X, y, **_tune_kwargs(tmp_path, extra_fixed={"n_neighbours": 7})
    )

    assert "n_neighbours" not in result["searched_params"]
    assert result["fixed_params"] == {"n_neighbours": 7}
    assert result["best_params"]["n_neighbours"] == 7

    loaded = load_best_params("kfp", tmp_path, X=X, y=y)
    assert loaded["n_neighbours"] == 7
    assert _get_arch_mode("kfp", **loaded).n_neighbours == 7


def test_tune_reports_when_every_trial_is_pruned(toy_tabular, tmp_path, monkeypatch):
    """An all-pruned study has no best trial; optuna's own error does not say why."""
    optuna = pytest.importorskip("optuna")
    # wfaudit absolute
    from wfaudit.helpers_ml import tuning

    X, y = toy_tabular

    def always_prune(trial):
        raise optuna.TrialPruned()

    monkeypatch.setitem(tuning.SPACES, "kfp", always_prune)
    with pytest.raises(RuntimeError, match="no trial completed"):
        tune("kfp", X, y, **_tune_kwargs(tmp_path, n_trials=2))


# --------------------------------------------------------------------------- #
# Phase 2: benchmark
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arch", ["kfp", "xgboost"])
def test_evaluate_with_tuned_params_reports_metrics(arch, toy_tabular, tmp_path):
    X, y = toy_tabular
    tune(arch, X, y, **_tune_kwargs(tmp_path))
    best = load_best_params(arch, workspace=tmp_path, X=X, y=y)

    score = evaluate_multiclass(
        arch, label="toy_tuned", data=X, labels=y, workspace=tmp_path, **best
    )

    for section in ("raw", "str"):
        assert "f1_score_macro" in score[section]
    mean, ci = score["raw"]["f1_score_macro"]
    assert 0.0 <= mean <= 1.0 and ci >= 0.0
    assert mean > 1.5 / N_CLASSES, "toy data should be learnable above chance"


def test_tuned_and_baseline_are_cached_separately(toy_tabular, tmp_path):
    """Regression test: the cache key must include the hyper-parameters."""
    X, y = toy_tabular
    common = dict(data=X, labels=y, workspace=tmp_path, label="toy")

    evaluate_multiclass("kfp", n_neighbours=2, **common)
    evaluate_multiclass("kfp", n_neighbours=11, **common)

    caches = sorted(p.name for p in tmp_path.glob("eval_*_kfp_toy_multiclass*.json"))
    assert len(caches) == 2, caches


def test_params_signature_is_stable_and_backwards_compatible():
    assert _params_signature({}) == ""
    assert _params_signature({"a": 1, "b": 2}) == _params_signature({"b": 2, "a": 1})
    assert _params_signature({"a": 1}) != _params_signature({"a": 2})


def test_sequence_cache_key_is_order_sensitive(toy_sequence):
    """Shuffling rows changes the cross-validation folds, so it must change the key."""
    X, y = toy_sequence
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    assert _array_hash(X, y) != _array_hash(X[perm], y[perm])

    nudged = X.copy()
    nudged[0, 0, 0] += 1e-3
    assert _array_hash(X, y) != _array_hash(nudged, y)
    assert _array_hash(X, y) == _array_hash(X.copy(), y.copy())


def test_cache_returns_identical_score_on_replay(toy_tabular, tmp_path):
    X, y = toy_tabular
    common = dict(data=X, labels=y, workspace=tmp_path, label="toy", max_depth=6)
    first = evaluate_multiclass("kfp", **common)
    second = evaluate_multiclass("kfp", **common)
    assert first["str"] == second["str"]


# --------------------------------------------------------------------------- #
# Study scoping
# --------------------------------------------------------------------------- #
def test_fingerprint_separates_datasets(toy_tabular):
    X, y = toy_tabular
    assert dataset_fingerprint(X, y) == dataset_fingerprint(X.copy(), y.copy())
    assert dataset_fingerprint(X, y) != dataset_fingerprint(X + 1.0, y)
    assert dataset_fingerprint(X, y) != dataset_fingerprint(X[:-8], y[:-8])
    assert dataset_fingerprint(X, y) != dataset_fingerprint(X, (y + 1) % N_CLASSES)


def test_two_datasets_do_not_share_a_study(toy_tabular, tmp_path):
    """Regression test: a shared study would report one dataset's score for another."""
    optuna = pytest.importorskip("optuna")
    X_easy, y_easy = toy_tabular
    rng = np.random.default_rng(1)
    X_hard = X_easy + rng.normal(0, 6.0, X_easy.shape).astype(np.float32)

    easy = tune("kfp", X_easy, y_easy, **_tune_kwargs(tmp_path, dataset_tag="easy"))
    hard = tune("kfp", X_hard, y_easy, **_tune_kwargs(tmp_path, dataset_tag="hard"))

    assert easy["dataset_id"] != hard["dataset_id"]
    assert easy["study_name"] != hard["study_name"]
    assert len(optuna.get_all_study_names(f"sqlite:///{tmp_path / 'hpo.db'}")) == 2
    assert hard["best_value_macro_f1"] < easy["best_value_macro_f1"]

    # each dataset resolves to its own record
    assert load_best_params("kfp", tmp_path, X=X_easy, y=y_easy) == easy["best_params"]
    assert load_best_params("kfp", tmp_path, X=X_hard, y=y_easy) == hard["best_params"]


def test_ambiguous_load_is_refused(toy_tabular, tmp_path):
    X, y = toy_tabular
    tune("kfp", X, y, **_tune_kwargs(tmp_path, dataset_tag="a"))
    tune("kfp", X + 5.0, y, **_tune_kwargs(tmp_path, dataset_tag="b"))
    with pytest.raises(ValueError, match="2 tuning results"):
        load_best_params("kfp", tmp_path)


def test_unknown_dataset_is_refused(toy_tabular, tmp_path):
    X, y = toy_tabular
    tune("kfp", X, y, **_tune_kwargs(tmp_path))
    with pytest.raises(FileNotFoundError):
        load_best_params("kfp", tmp_path, X=X * 3.0, y=y)


def test_protocol_change_does_not_reuse_a_study(toy_tabular, tmp_path):
    """Trial values scored under different protocols must not be pooled."""
    X, y = toy_tabular
    a = tune("kfp", X, y, **_tune_kwargs(tmp_path, per_class=12))
    b = tune("kfp", X, y, **_tune_kwargs(tmp_path, per_class=8))
    assert a["dataset_id"] == b["dataset_id"]
    assert a["protocol_id"] != b["protocol_id"]
    assert b["n_trials_run"] == 3, "must be a fresh study, not a resumed one"


def test_identity_mismatch_on_explicit_study_name_raises(toy_tabular, tmp_path):
    X, y = toy_tabular
    first = tune("kfp", X, y, **_tune_kwargs(tmp_path))
    with pytest.raises(ValueError, match="not comparable"):
        tune(
            "kfp",
            X,
            y,
            **_tune_kwargs(tmp_path, val_frac=0.4, study_name=first["study_name"]),
        )


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arch", ["kfp", "xgboost"])
def test_full_pipeline_reports_baseline_and_tuned(arch, toy_tabular, tmp_path):
    """The delta a reviewer asks for: same data, same folds, two configurations."""
    X, y = toy_tabular

    baseline = evaluate_multiclass(
        arch,
        label="toy_baseline",
        data=X,
        labels=y,
        workspace=tmp_path,
    )
    tune(arch, X, y, **_tune_kwargs(tmp_path, n_trials=4))
    tuned = evaluate_multiclass(
        arch,
        label="toy_tuned",
        data=X,
        labels=y,
        workspace=tmp_path,
        **load_best_params(arch, workspace=tmp_path, X=X, y=y),
    )

    report = {
        "arch": arch,
        "baseline_f1": baseline["raw"]["f1_score_macro"][0],
        "tuned_f1": tuned["raw"]["f1_score_macro"][0],
    }
    report["delta"] = report["tuned_f1"] - report["baseline_f1"]

    assert all(np.isfinite(v) for v in (report["baseline_f1"], report["tuned_f1"]))
    print(
        f"\n{arch}: baseline={report['baseline_f1']:.3f} "
        f"tuned={report['tuned_f1']:.3f} delta={report['delta']:+.3f}"
    )


# --------------------------------------------------------------------------- #
# Neural architectures
# --------------------------------------------------------------------------- #
NN_ARCHS = ["robustfp", "df", "varcnn", "holmes"]


@pytest.mark.slow
@pytest.mark.parametrize("arch", NN_ARCHS)
def test_neural_baseline_builds(arch):
    pytest.importorskip("torch")
    assert hasattr(_get_arch_mode(arch), "fit")


@pytest.mark.slow
@pytest.mark.parametrize("arch", NN_ARCHS)
def test_sampled_neural_config_trains(arch, toy_sequence, tmp_path):
    """Every point in the space must build, run a forward and backward pass, and
    predict. Shape errors from architecture parameters surface here."""
    optuna = pytest.importorskip("optuna")
    pytest.importorskip("torch")
    X, y = toy_sequence

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(4):
        params = SPACES[arch](study.ask())
        params.update(epochs=1, patience=1, verbose=False, batch_size=16)
        model = _get_arch_mode(arch, **params).fit(X, y)
        proba = model.predict_proba(X[:8])
        assert proba.shape == (8, N_CLASSES)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)


@pytest.mark.slow
def test_varcnn_dilation_shapes_match():
    """The residual branch must line up; a fixed padding breaks this when the
    dilation is greater than one."""
    torch = pytest.importorskip("torch")
    # wfaudit absolute
    from wfaudit.helpers_ml.varcnn import VARCNN

    x = torch.zeros(2, 2, 256)
    for dilated in (False, True):
        assert VARCNN(N_CLASSES, dilated=dilated)(x).shape == (2, N_CLASSES)


@pytest.mark.slow
@pytest.mark.parametrize(
    "arch,param,value,probe",
    [
        ("df", "dropout_cls", 0.42, lambda m: m.dropout_cls),
        ("varcnn", "dilated", True, lambda m: m.dilated),
        ("varcnn", "dropout", 0.37, lambda m: m.dropout),
        ("holmes", "conv_num_layers", 3, lambda m: m.conv_num_layers),
        ("robustfp", "dropout_conv", 0.44, lambda m: m.dropout_conv),
    ],
)
def test_neural_parameter_is_stored(arch, param, value, probe):
    pytest.importorskip("torch")
    assert probe(_get_arch_mode(arch, **{param: value})) == value


@pytest.mark.slow
def test_holmes_rejects_a_depth_that_collapses_the_trace(toy_sequence):
    pytest.importorskip("torch")
    X, y = toy_sequence
    short = X[:, :, :20]  # 20 samples cannot survive four poolings of 3
    with pytest.raises(ValueError, match="zero"):
        _get_arch_mode("holmes", conv_num_layers=5, epochs=1, verbose=False).fit(
            short, y
        )


# --------------------------------------------------------------------------- #
# Neural path
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_robustfp_pipeline(toy_sequence, tmp_path):
    pytest.importorskip("torch")
    X, y = toy_sequence

    result = tune(
        "robustfp",
        X,
        y,
        **_tune_kwargs(
            tmp_path,
            n_trials=2,
            epoch_budget=3,
            patience=2,
            extra_fixed={"batch_size": 16},
        ),
    )
    best = load_best_params("robustfp", workspace=tmp_path, X=X, y=y)
    assert "lr" in best and "dropout_conv" in best
    assert "on_epoch_end" not in best, "search-only kwargs must not reach the benchmark"
    assert 0.0 <= result["best_value_macro_f1"] <= 1.0

    score = evaluate_multiclass(
        "robustfp",
        label="toy_tuned",
        data=X,
        labels=y,
        workspace=tmp_path,
        epochs=3,
        patience=2,
        verbose=False,
        **best,
    )
    mean, _ = score["raw"]["f1_score_macro"]
    assert 0.0 <= mean <= 1.0


@pytest.mark.slow
def test_oversized_batch_raises_instead_of_skipping_training():
    """drop_last=True means batch_size > train split yields zero batches."""
    pytest.importorskip("torch")
    # third party
    import torch

    # wfaudit absolute
    from wfaudit.helpers_ml._core_nn import InvalidTrainingConfig, train_model

    X = np.zeros((40, 4), dtype=np.float32)
    y = np.repeat(np.arange(4), 10)
    net = torch.nn.Linear(4, 4)
    with pytest.raises(InvalidTrainingConfig, match="no complete batches"):
        train_model(net, X, y, epochs=1, batch_size=512, verbose=False)


@pytest.mark.slow
def test_train_model_defaults_are_unchanged():
    """The exposed hyper-parameters must not alter the previous behaviour."""
    torch = pytest.importorskip("torch")
    # wfaudit absolute
    from wfaudit.helpers_ml._core_nn import train_model

    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 4)).astype(np.float32)
    y = np.repeat(np.arange(4), 20)

    def make():
        torch.manual_seed(0)
        return torch.nn.Sequential(
            torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
        )

    a = train_model(make(), X, y, epochs=3, batch_size=8, verbose=False)
    b = train_model(
        make(),
        X,
        y,
        epochs=3,
        batch_size=8,
        verbose=False,
        lr=0.002,
        weight_decay=0.0,
        optimizer_name="adam",
        scheduler_name="none",
        label_smoothing=0.0,
    )
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa, pb)
