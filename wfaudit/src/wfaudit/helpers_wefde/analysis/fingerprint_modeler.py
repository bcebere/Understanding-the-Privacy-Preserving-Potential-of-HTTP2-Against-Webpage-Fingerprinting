# Adapted from https://github.com/notem/reWeFDE

# future
from __future__ import annotations

# stdlib
import math

# third party
from joblib import Parallel, delayed
import numpy as np

# wfaudit absolute
from wfaudit.helpers_wefde.analysis.kde_wrapper import KDE
import wfaudit.logger as log


class WebsiteFingerprintModeler:
    """
    data : WebsiteData-like object with
           - data.sites = list of site indices
           - data.get_site(site, feature) = numpy array of values for that site/feature
           - data.get_feature(feature) = numpy array of values for that feature across all sites
    web_priors : Optional list of site priors, default uniform.
    """

    # ------------------------------------------------------------------
    def __init__(self, data, web_priors=None, *, discrete_threshold: int = 10000):
        self.data = data
        self.sites = data.sites
        self.website_priors = (
            web_priors
            if web_priors is not None
            else [1 / len(self.sites)] * len(self.sites)
        )
        self.discrete_threshold = discrete_threshold
        self._sample_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------
    def max_information_leakage(self) -> float:
        return -sum(p * math.log2(p) for p in self.website_priors if p > 0)

    # ------------------------------------------------------------------
    def _make_kde_for_cluster_and_site(self, cluster, site):
        cols = [self.data.get_site(site, f).reshape(-1, 1) for f in cluster]
        X = np.hstack(cols)
        return KDE(X, discrete_threshold=self.discrete_threshold)

    # ------------------------------------------------------------------
    def _draw_samples(self, kdes, sample_size: int):
        """Draw & cache samples for (dim, sample_size)."""
        d = kdes[0].n_features
        cache_key = (d, sample_size)
        if cache_key in self._sample_cache:
            return self._sample_cache[cache_key]

        pri = np.array(self.website_priors)
        counts = np.floor(pri * sample_size).astype(int)
        leftover = sample_size - counts.sum()
        if leftover > 0:
            counts[:leftover] += 1  # distribute residual

        X_all, idx_all = [], []
        for i, (kde, n_i) in enumerate(zip(kdes, counts)):
            if n_i:
                samp_i = kde.sample(int(n_i))
                X_all.append(samp_i)
                idx_all.append(np.full(len(samp_i), i, dtype=int))
        X_all = np.vstack(X_all)
        idx_all = np.concatenate(idx_all)
        self._sample_cache[cache_key] = (X_all, idx_all)
        return X_all, idx_all

    # ------------------------------------------------------------------
    def information_leakage(
        self, clusters, *, sample_size: int = 5000, n_procs: int = 2
    ):
        if not clusters:
            return []

        H_C = self.max_information_leakage()
        priors_log2 = np.log2(self.website_priors)[:, None]  # (sites,1)

        results = []
        for cluster in clusters:
            kdes = [self._make_kde_for_cluster_and_site(cluster, s) for s in self.sites]
            log.info(f"[Cluster {cluster}] Constructed KDEs : {len(kdes)}")

            X_samp, _ = self._draw_samples(kdes, sample_size)
            if X_samp.size == 0:
                results.append(0.0)
                continue

            # log p(x|site) in parallel
            def _logp(kde):
                with np.errstate(divide="ignore"):
                    return np.log2(kde.predict(X_samp))

            logp = np.array(
                Parallel(n_jobs=n_procs)(delayed(_logp)(k) for k in kdes)
            )  # (sites,N)
            lp = logp + priors_log2
            max_lp = lp.max(axis=0, keepdims=True)
            post = 2 ** (lp - max_lp)
            post /= post.sum(axis=0, keepdims=True)

            # logp = np.zeros_like(post)
            # np.log2(post, where=post > 0, out=logp)
            logp = np.log2(np.clip(post, 1e-300, 1.0))

            H_post = -np.sum(post * logp, axis=0)
            # H_post = -np.sum(post * np.log2(post, where=post > 0
            H_C_given_f = H_post.mean()

            if np.isnan(H_C_given_f):
                raise

            # 6. I(C; f) = H(C) - H(C|f).
            results.append(H_C - H_C_given_f)
        return results
