# Adapted from https://github.com/notem/reWeFDE
# stdlib
import copy
import hashlib
import os
from pathlib import Path

# third party
import dill
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessPool as Pool, cpu_count

# wfaudit absolute
from wfaudit.helpers_wefde.analysis.data_utils import WebsiteData, WebsiteData_v2
from wfaudit.helpers_wefde.analysis.fingerprint_modeler import WebsiteFingerprintModeler
from wfaudit.helpers_wefde.analysis.mi_analyzer import MutualInformationAnalyzer
import wfaudit.logger as log


def _individual_measure(modeler, pool, checkpoint):
    """
    Perform information leakage analysis for each feature one-by-one.

    The resulting leakages are saved in a plain-text ascii checkpoint file,
    which can be loaded in subsequent runs to avoid re-processing features.

    Parameters
    ----------
    modeler : WebsiteFingerprintModeler
        initialized fingerprinting engine
    pool : ProcessPool
        Pool to use for multiprocessing.
    checkpoint : str
        Path to ascii file to save individual leakage checkpoint information.

    Returns
    -------
    list
        list of leakages where the index of each leakage maps to the feature number

    """
    leakage_indiv = []

    # open a checkpoint file
    if checkpoint:
        lines = None
        if os.path.exists(checkpoint):
            with open(checkpoint, "r") as tmp_file:
                past_leaks = [float(line) for line in tmp_file]
                lines = len(past_leaks)
                leakage_indiv = past_leaks
        tmp_file = open(checkpoint, "a+")

    # if a pool has been provided, perform computation in parallel
    # otherwise do serial computation
    if checkpoint and lines:
        features = modeler.data.features[lines:]
    else:
        features = modeler.data.features
    if pool is None:
        proc_results = map(modeler, features)
    else:
        proc_results = pool.imap(modeler, features)
        pool.close()
    size = len(modeler.data.features)  # number of features

    log.info("Begin individual leakage measurements.")
    # measure information leakage
    # log current progress at twenty intervals
    for leakage in proc_results:
        leakage_indiv.append(leakage[0])
        if len(leakage_indiv) - 1 % int(size * 0.05) == 0:
            log.info(f"Progress InfoLeak: {len(leakage_indiv)}/{size}")
        if checkpoint:
            tmp_file.write(f"{str(leakage[0])}\n")
            tmp_file.flush()
    log.info("Progress: Done.")
    if pool is not None:
        pool.join()
        pool.restart()
    if checkpoint:
        tmp_file.close()
    return leakage_indiv


def print_leakage(
    features_range: dict,  # offsets for leakage types
    indiv_file: str,  # path to individual leakages,
    joint_file: str,  # path to joint leakages,
):
    # read independent leakage information from files
    with open(indiv_file, "rb") as fi:
        leakages = dill.load(fi)
    with open(joint_file, "rb") as fi:
        joint_leakages = dill.load(fi)

    summary = {}
    offset = 0
    if features_range is not None:
        for category in features_range:
            next_off = features_range[category]
            y = leakages[offset:next_off]
            summary[category] = [np.max(y)]
            offset = next_off

    summary["joint"] = [joint_leakages[0]]

    summary = pd.DataFrame(summary)
    return summary


def _evaluate_individual_leakage(
    feature_data,
    outdir: Path,
    n_procs: int = 0,
    topn=20,
    nmi_threshold=0.9,
    discrete_threshold=100000,
):
    # create process pool
    if n_procs > 1:
        pool = Pool(n_procs)
    elif n_procs == 0:
        pool = Pool(cpu_count() - 1)
    else:
        pool = None

    # initialize fingerprint modeler
    modeler = WebsiteFingerprintModeler(
        feature_data, discrete_threshold=discrete_threshold
    )
    modeler._pool = pool  # configure modeler to use the proc pool

    max_info_leakage = modeler.max_information_leakage()
    log.info(f"Maximum information leakage = {max_info_leakage}")
    with open(outdir / "max_entropy.pkl", "wb") as fi:
        dill.dump(max_info_leakage, fi)

    # load previous leakage measurements if possible
    indiv_path = outdir / "indiv.pkl"
    if indiv_path.exists():
        with open(indiv_path, "rb") as fi:
            log.info("Loading individual leakage measures from file.")
            leakage_indiv = dill.load(fi)
    else:
        # otherwise do individual measure
        log.info("Begin individual feature analysis.")

        # perform individual measure with checkpointing
        chk_path = outdir / "indiv_checkpoint.txt"
        leakage_indiv = _individual_measure(modeler, pool, str(chk_path))

        # save individual leakage to file
        log.info(f"Saving individual leakage to {indiv_path}.")
        with open(indiv_path, "wb") as fi:
            dill.dump(leakage_indiv, fi)

    # perform combined information leakage measurements
    # initialize MI analyzer
    analyzer = MutualInformationAnalyzer(feature_data, pool=pool)

    # sort the list of features by their individual leakage
    # we will process these features in the order of their importance during MI analysis
    log.info("Sorting features by individual leakage.")
    tuples = list(zip(feature_data.features, leakage_indiv))
    tuples = sorted(tuples, key=lambda x: (-x[1], x[0]))
    log.info(f"TopN:\t {tuples[:topn]}")
    sorted_features = list(list(zip(*tuples))[0])

    # process into list of non-redundant features
    cln_path = outdir / "cleaned.pkl"
    rdn_path = outdir / "redundant.pkl"
    chk_path = outdir / "prune_checkpoint.txt"
    if cln_path.exists():
        log.info("Loading top non-redundant features from file.")
        with open(cln_path, "rb") as fi:
            cleaned = dill.load(fi)
    else:
        log.info("Begin feature pruning.")
        try:
            cleaned, pruned = analyzer.prune(
                features=sorted_features,
                nmi_threshold=nmi_threshold,
                topn=topn,
                checkpoint=str(chk_path),
            )
            with open(cln_path, "wb") as fi:
                dill.dump(cleaned, fi)
            with open(rdn_path, "wb") as fi:
                dill.dump(pruned, fi)
        except BaseException:
            cleaned = sorted_features
            pruned = []

    top_feats = dict(tuples)
    relevant_feats = sorted(cleaned, key=lambda x: top_feats[x], reverse=True)

    return modeler, analyzer, relevant_feats, leakage_indiv


def _base_evaluate_info_leakage(
    feature_data,
    output_path: str,  # where to save the leakages
    features_range: dict,  # the offsets of each feature
    n_procs=0,
    n_samples=50000,
    topn=20,
    nmi_threshold=0.9,
    discrete_threshold=100000,
    max_instances=100,
):
    """
    Run the full information leakage analysis on a processed dataset.

    Parameters
    ----------
    features_data :
        Operating system file path to the directory containing processed feature files.
    output_path : str
        Operating system file path to the directory where analysis results should be saved.
    n_procs : int
        Number of processes to use for parallelism.
        If 0 is used, auto-detect based on number of system CPUs.
    n_samples : int
        Number of samples to use when performing monte-carlo estimation when running the fingerprint modeler.
    topn : int
        Top number of features to analyze during joint analysis.
    nmi_threshold : float
        Cut-off value for determining redundant features. Should be a percentage value.

    Returns
    -------
    float
        Combined feature leakage (in bits)
    """
    # prepare feature dataset
    log.info(f"Loaded {len(feature_data.sites)} sites.")
    log.info(f"Loaded {len(feature_data)} instances.")
    log.info(
        f"""Running analysis with params:
        *     n_procs = {n_procs}
        *     n_samples = {n_samples}
        *     topn = {topn}
        *     nmi_threshold = {nmi_threshold}
        *     max_instances = {max_instances}
             """
    )

    # directory to save results
    outdir = Path(output_path)
    outdir.mkdir(parents=True, exist_ok=True)

    if n_procs == 0:
        n_procs = cpu_count() - 1

    modeler, analyzer, relevant_feats, leakage_indiv = _evaluate_individual_leakage(
        feature_data,
        outdir=outdir,
        n_procs=n_procs,
        topn=topn,
        nmi_threshold=nmi_threshold,
    )

    indiv_path = outdir / "indiv.pkl"
    assert indiv_path.exists()

    log.info(f"Relevant features = {relevant_feats} total = {len(relevant_feats)}")
    # cluster non-redundant features
    cst_path = outdir / "clusters.pkl"
    chk_path = outdir / "prune_checkpoint.txt"
    if cst_path.exists():
        log.info("Loading clusters from file.")
        with open(cst_path, "rb") as fi:
            clusters = dill.load(fi)
    else:
        log.info("Begin feature clustering.")
        try:
            clusters, _ = analyzer.cluster(relevant_feats, checkpoint=str(chk_path))
            with open(cst_path, "wb") as fi:
                dill.dump(clusters, fi)
        except BaseException:
            clusters = [relevant_feats]

    # perform joint information leakage measurement
    log.info(f"Identified {len(clusters)} clusters.")
    log.info("Begin cluster leakage measurements.")

    def _eval_and_cache(clusters, joint_leakage: bool, out_file: str):
        out_path = outdir / out_file
        if os.path.exists(out_path):
            with open(out_path, "rb") as fi:
                results = dill.load(fi)
        else:
            results = modeler.information_leakage(
                clusters=clusters, sample_size=n_samples, joint_leakage=joint_leakage
            )
            assert len(results) != 0, clusters
            with open(out_path, "wb") as fi:
                dill.dump(results, fi)
        return results

    def _get_pretty_names(label, indexes):
        if features_range is not None:
            features_range_list = list(features_range.keys())
            pretty_names = [features_range_list[idx] for idx in indexes]
            # log.info(f"{label} : {pretty_names}")
            return pretty_names
        return None

    joint_leakage = _eval_and_cache(
        clusters=relevant_feats, joint_leakage=True, out_file="cleaned_leakage.pkl"
    )

    log.info(
        f"Non-redundant leakage results: {joint_leakage} bits. Source: {_get_pretty_names('Non-redundant', relevant_feats)}"
    )

    joint_path = outdir / "joint.pkl"
    if joint_path.exists():
        with open(joint_path, "rb") as fi:
            leakage_cluster_joint = dill.load(fi)
    else:
        leakage_cluster_joint = modeler.information_leakage(
            clusters=clusters, sample_size=n_samples, joint_leakage=True
        )
        with open(joint_path, "wb") as fi:
            dill.dump(leakage_cluster_joint, fi)

    log.info(
        f"Non-redundant clusters joint leakage results: {leakage_cluster_joint} bits. Clusters {len(clusters)}"
    )

    cluster_sep_results = Parallel(n_jobs=n_procs)(
        delayed(_eval_and_cache)(
            clusters=cluster, joint_leakage=True, out_file=f"cluster_{cidx}_leakage.pkl"
        )
        for cidx, cluster in enumerate(clusters)
    )

    log.info(f"Non-redundant clusters indiv leakage results: {cluster_sep_results}.")
    log.info("Finished execution.")
    return print_leakage(
        features_range=features_range,  # offsets for leakage types
        indiv_file=indiv_path,  # path to individual leakages,
        joint_file=joint_path,  # path to joint leakages,
    )


def evaluate_info_leakage(
    features_path: str,  # the folder with output of the feature extraction
    output_path: str,  # where to save the leakages
    features_range: dict,  # the offsets of each feature
    n_procs=0,
    n_samples=50000,
    topn=20,
    nmi_threshold=0.9,
    discrete_threshold=100000,
    max_instances=100,
):
    """
    Run the full information leakage analysis on a processed dataset.

    Parameters
    ----------
    features_path : str
        Operating system file path to the directory containing processed feature files.
    output_path : str
        Operating system file path to the directory where analysis results should be saved.
    n_procs : int
        Number of processes to use for parallelism.
        If 0 is used, auto-detect based on number of system CPUs.
    n_samples : int
        Number of samples to use when performing monte-carlo estimation when running the fingerprint modeler.
    topn : int
        Top number of features to analyze during joint analysis.
    nmi_threshold : float
        Cut-off value for determining redundant features. Should be a percentage value.

    Returns
    -------
    float
        Combined feature leakage (in bits)
    """
    # prepare feature dataset
    log.info("Loading dataset.")
    feature_data = WebsiteData(features_path, max_instances=max_instances)
    return _base_evaluate_info_leakage(
        feature_data=feature_data,
        output_path=output_path,
        features_range=features_range,
        n_procs=n_procs,
        n_samples=n_samples,
        topn=topn,
        nmi_threshold=nmi_threshold,
        discrete_threshold=discrete_threshold,
        max_instances=max_instances,
    )


def evaluate_info_leakage_v2(
    X: np.ndarray,
    y: np.ndarray,
    output_path: str,  # where to save the leakages
    features_range: dict,  # the offsets of each feature
    n_procs=0,
    n_samples=50000,
    topn=40,
    nmi_threshold=0.9,
    discrete_threshold=100000,
    max_instances=100,
):
    """
    Run the full information leakage analysis on a processed dataset.

    Parameters
    ----------
    X, y: dataset
    output_path : str
        Operating system file path to the directory where analysis results should be saved.
    n_procs : int
        Number of processes to use for parallelism.
        If 0 is used, auto-detect based on number of system CPUs.
    n_samples : int
        Number of samples to use when performing monte-carlo estimation when running the fingerprint modeler.
    topn : int
        Top number of features to analyze during joint analysis.
    nmi_threshold : float
        Cut-off value for determining redundant features. Should be a percentage value.

    Returns
    -------
    float
        Combined feature leakage (in bits)
    """
    # prepare feature dataset
    log.info("Loading dataset.")
    feature_data = WebsiteData_v2(X, y, max_instances=max_instances)
    return _base_evaluate_info_leakage(
        feature_data=feature_data,
        output_path=output_path,
        features_range=features_range,
        n_procs=n_procs,
        n_samples=n_samples,
        topn=topn,
        nmi_threshold=nmi_threshold,
        discrete_threshold=discrete_threshold,
        max_instances=max_instances,
    )


def exploratory_analysis(
    X,
    y,
    min_cluster_size: int,
    output_path: str,  # where to save the leakages
    features_range: dict,  # the offsets of each feature
    n_procs=0,
    n_samples=50000,
    topn=20,
    nmi_threshold=0.9,
    discrete_threshold=100000,
    max_instances=100,
):
    """
    Identify clusters of similar leakage.

    Parameters
    ----------
    X, y :
        features and labels
    output_path : str
        Operating system file path to the directory where analysis results should be saved.
    n_procs : int
        Number of processes to use for parallelism.
        If 0 is used, auto-detect based on number of system CPUs.
    n_samples : int
        Number of samples to use when performing monte-carlo estimation when running the fingerprint modeler.
    topn : int
        Top number of features to analyze during joint analysis.
    nmi_threshold : float
        Cut-off value for determining redundant features. Should be a percentage value.

    Returns
    -------
    float
        Combined feature leakage (in bits)
    """
    # prepare feature dataset
    feature_data = WebsiteData_v2(X, y, max_instances=max_instances)
    log.info(f"Loaded {len(feature_data.sites)} sites.")
    log.info(f"Loaded {len(feature_data)} instances.")

    # directory to save results
    outdir = Path(output_path)
    outdir.mkdir(parents=True, exist_ok=True)

    if n_procs == 0:
        n_procs = cpu_count() - 1

    modeler, analyzer, relevant_feats, leakage_indiv = _evaluate_individual_leakage(
        feature_data,
        outdir=outdir,
        n_procs=n_procs,
        topn=topn,
        nmi_threshold=nmi_threshold,
    )

    tuples = list(zip(feature_data.features, leakage_indiv))
    tuples = sorted(tuples, key=lambda x: (-x[1], x[0]))
    # sorted_features = list(list(zip(*tuples))[0])
    top_feats = dict(tuples)

    log.info(f"Relevant features = {relevant_feats} total = {len(relevant_feats)}")
    log.info(f"Leakage dict: {top_feats}")

    def _eval_and_cache(features, out_file: str = None):
        if out_file is None:
            cluster_hash = "_".join(map(str, sorted(features)))
            md5_hash = hashlib.md5()
            md5_hash.update(cluster_hash.encode("utf-8"))
            cluster_hash = md5_hash.hexdigest()
            out_file = f"cluster_len{len(features)}_{cluster_hash}.pkl"
            # log.debug(f'Evaluation output: {out_file}')

        out_path = outdir / out_file

        if os.path.exists(out_path):
            with open(out_path, "rb") as fi:
                results = dill.load(fi)
        else:
            results = modeler.information_leakage(
                clusters=features, sample_size=n_samples, joint_leakage=True
            )
            assert len(results) != 0, features
            with open(out_path, "wb") as fi:
                dill.dump(results, fi)
        return float(results[0])

    def _get_pretty_names(label, indexes):
        if features_range is not None:
            features_range_list = list(features_range.keys())
            pretty_names = [features_range_list[idx] for idx in indexes]

            # log.info(f"{label} : {pretty_names}")
            return pretty_names
        return None

    joint_leakage = _eval_and_cache(
        features=relevant_feats, out_file="cleaned_leakage.pkl"
    )

    log.info(
        f"Non-redudant leakage results: {joint_leakage} bits. Source: {_get_pretty_names('Non-redundant', relevant_feats)}"
    )
    top_features = copy.deepcopy(relevant_feats)
    relevant_feats = relevant_feats[:topn]

    max_cluster_size = min(int(joint_leakage) + 20, len(relevant_feats))

    candidates = []
    for test_size in range(min_cluster_size, max_cluster_size):
        log.info(f"Test clusters of size {test_size}/{len(relevant_feats)}")
        if test_size > len(relevant_feats):
            log.info(f"Less than {test_size} features remained!! Exiting...")
            break
        cluster_subsets = [
            relevant_feats[i : i + test_size] for i in range(0, len(relevant_feats), 1)
        ]

        cluster_size_leakages = Parallel(n_jobs=n_procs)(
            delayed(_eval_and_cache)(features=cluster)
            for cidx, cluster in enumerate(cluster_subsets)
        )

        cluster_leakages_sorted = sorted(
            enumerate(cluster_size_leakages), key=lambda x: x[1], reverse=True
        )
        rem_feats = set(relevant_feats)

        for tidx, leakage in cluster_leakages_sorted:
            log.debug(
                f"Leakage {leakage}/{joint_leakage} with cluster {cluster_subsets[tidx]}"
            )
            if joint_leakage <= leakage or abs(leakage - joint_leakage) <= 0.1:
                log.info(
                    f"Cluster {cluster_subsets[tidx]} is a candidate!. Leakage {leakage}/{joint_leakage}"
                )
                candidates.append((cluster_subsets[tidx], leakage))
                local_set = set(cluster_subsets[tidx])
                if local_set.issubset(rem_feats):
                    rem_feats.difference_update(local_set)
                    log.info(
                        f"Dropping features {cluster_subsets[tidx]} from benchmarks!!"
                    )

        rem_feats = list(rem_feats)
        relevant_feats = sorted(rem_feats, key=lambda x: top_feats[x], reverse=True)

    log.info(f"Redundant fingerprints =  {len(candidates)}")
    results = []
    for idx, (candidate, cand_leak) in enumerate(candidates):
        log.info(
            f" [{idx}] {cand_leak} bits >>> {_get_pretty_names('cluster', candidate)} leaks"
        )
        results.append((candidate, cand_leak))
    log.info("Finished exploration.")

    return top_features, results
