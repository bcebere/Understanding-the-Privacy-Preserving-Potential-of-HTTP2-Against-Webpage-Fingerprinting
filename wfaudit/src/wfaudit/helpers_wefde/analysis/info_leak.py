# Adapted from https://github.com/notem/reWeFDE
# stdlib
import os

# third party
import dill
import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessPool as Pool, cpu_count

# wfaudit absolute
from wfaudit.helpers_wefde.analysis.data_utils import WebsiteData
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
    for category in features_range:
        next_off = features_range[category]
        y = leakages[offset:next_off]
        summary[category] = [np.max(y)]
        offset = next_off

    summary["joint"] = joint_leakages[0]

    summary = pd.DataFrame(summary)
    return summary


def evaluate_info_leakage(
    features_path: str,  # the folder with output of the feature extraction
    output_path: str,  # where to save the leakages
    features_range: dict,  # the offsets of each feature
    n_procs=0,
    n_samples=50000,
    topn=100,
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
    log.info(f"Loaded {len(feature_data.sites)} sites.")
    log.info(f"Loaded {len(feature_data)} instances.")

    # create process pool
    if n_procs > 1:
        pool = Pool(n_procs)
    elif n_procs == 0:
        pool = Pool(cpu_count())
    else:
        pool = None

    # directory to save results
    outdir = output_path
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # initialize fingerprint modeler
    modeler = WebsiteFingerprintModeler(
        feature_data, discrete_threshold=discrete_threshold
    )

    # load previous leakage measurements if possible
    indiv_path = os.path.join(outdir, "indiv.pkl")
    if os.path.exists(indiv_path):
        with open(indiv_path, "rb") as fi:
            log.info("Loading individual leakage measures from file.")
            leakage_indiv = dill.load(fi)

    # otherwise do individual measure
    else:
        log.info("Begin individual feature analysis.")

        # perform individual measure with checkpointing
        chk_path = os.path.join(outdir, "indiv_checkpoint.txt")
        leakage_indiv = _individual_measure(modeler, pool, chk_path)

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
    log.info(f"Top 20:\t {tuples[:20]}")
    sorted_features = list(list(zip(*tuples))[0])

    # process into list of non-redundant features
    cln_path = os.path.join(outdir, "cleaned.pkl")
    rdn_path = os.path.join(outdir, "redundant.pkl")
    chk_path = os.path.join(outdir, "prune_checkpoint.txt")
    if os.path.exists(cln_path):
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
                checkpoint=chk_path,
            )
            with open(cln_path, "wb") as fi:
                dill.dump(cleaned, fi)
            with open(rdn_path, "wb") as fi:
                dill.dump(pruned, fi)
        except BaseException:
            cleaned = sorted_features
            pruned = []

    log.info(f"cleaned features = {cleaned} total = {len(cleaned)}")
    # cluster non-redundant features
    cst_path = os.path.join(outdir, "clusters.pkl")
    if os.path.exists(cst_path):
        log.info("Loading clusters from file.")
        with open(cst_path, "rb") as fi:
            clusters = dill.load(fi)
    else:
        log.info("Begin feature clustering.")
        try:
            clusters, _ = analyzer.cluster(cleaned, checkpoint=chk_path)
            with open(cst_path, "wb") as fi:
                dill.dump(clusters, fi)
        except BaseException:
            clusters = [cleaned]

    max_info_leakage = modeler.max_information_leakage()
    with open(os.path.join(outdir, "max_entropy.pkl"), "wb") as fi:
        dill.dump(max_info_leakage, fi)

    # perform joint information leakage measurement
    log.info(f"Identified {len(clusters)} clusters.")
    log.info("Begin cluster leakage measurements.")
    modeler._pool = pool  # configure modeler to use the proc pool

    def _eval_and_cache(clusters, joint_leakage: bool, out_file: str):
        out_path = os.path.join(outdir, out_file)
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

    selected_candidates = []
    leak_median = np.median(leakage_indiv)

    # [feat[0] for feat in clusters]
    for cluster in clusters:
        candidate_idx = np.argmax(np.asarray(leakage_indiv)[cluster])
        candidate = cluster[candidate_idx]
        candidate_leak = np.asarray(leakage_indiv)[candidate]

        if leak_median < candidate_leak:
            selected_candidates.append(candidate)

    top_leaks = _eval_and_cache(
        clusters=selected_candidates,
        joint_leakage=False,
        out_file="top_per_cluster_leaks.pkl",
    )
    log.info(f"Top-cluster leakage results: {top_leaks} bits")

    joint_leakage = _eval_and_cache(
        clusters=cleaned, joint_leakage=True, out_file="cleaned_leakage.pkl"
    )
    log.info(f"Cleaned feats leakage results: {joint_leakage} bits")

    joint_path = os.path.join(outdir, "joint.pkl")
    if os.path.exists(joint_path):
        with open(joint_path, "rb") as fi:
            leakage_joint = dill.load(fi)
    else:
        leakage_joint = modeler.information_leakage(
            clusters=clusters, sample_size=n_samples, joint_leakage=True
        )
        with open(joint_path, "wb") as fi:
            dill.dump(leakage_joint, fi)

    log.info("Finished execution.")
    return print_leakage(
        features_range=features_range,  # offsets for leakage types
        indiv_file=indiv_path,  # path to individual leakages,
        joint_file=joint_path,  # path to joint leakages,
    )
