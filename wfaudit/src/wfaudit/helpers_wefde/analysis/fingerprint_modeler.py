# Adapted from https://github.com/notem/reWeFDE
# stdlib
import math

# third party
from joblib import Parallel, delayed
import numpy as np

# wfaudit absolute
from wfaudit.helpers_wefde.analysis.kde_wrapper import KDE
import wfaudit.logger as log


class WebsiteFingerprintModeler:
    def __init__(self, data, web_priors=None, discrete_threshold=10000):
        """
        data : WebsiteData-like object with
               - data.sites = list of site indices
               - data.get_site(site, feature) = numpy array of values for that site/feature
               - data.get_feature(feature) = numpy array of values for that feature across all sites
        web_priors : Optional list of site priors, default uniform.
        """
        self.data = data
        self.sites = data.sites
        self.website_priors = (
            web_priors
            if web_priors is not None
            else [1 / len(self.sites)] * len(self.sites)
        )
        self.discrete_threshold = discrete_threshold

    def max_information_leakage(self):
        """
        H(C) = - Σ p(c) log2 p(c)
        """
        return -sum(p * math.log2(p) for p in self.website_priors if p > 0)

    def _make_kde_for_cluster_and_site(self, cluster_features, site):
        """
        Build a *multi‐dimensional* KDE for 'cluster_features' restricted to 'site'.
        Returns a KDE object (see code below) fitted on shape=(n_samples, d) data,
        where d = len(cluster_features).
        """
        # collect the data columns
        X_list = []
        for feat in cluster_features:
            col = self.data.get_site(site, feat).reshape(-1, 1)
            X_list.append(col)
        X = np.hstack(
            X_list
        )  # shape = (num_samples_of_this_site, len(cluster_features))
        return KDE(X, discrete_threshold=self.discrete_threshold)

    def _sample_from_all_sites(self, kdes, sample_size):
        """
        Draw 'sample_size' total random samples, distributed among sites
        according to self.website_priors.  For site i, we draw
           n_i = int(sample_size * prior[i])    samples
        from the site‐specific KDE.  Then we label them all as "drawn from site i".
        Returns all samples stacked up, shape=(sample_size, d), plus a “site index” array.
        """
        site_indices = []
        X_all = []
        for i, kde in enumerate(kdes):
            num = int(sample_size * self.website_priors[i])
            if num > 0:
                samples_i = kde.sample(num)  # shape=(num, d)
                X_all.append(samples_i)
                site_indices.extend([i] * num)
        if not X_all:
            # corner case: e.g. if sample_size=0 or something
            return np.zeros((0, kdes[0].n_features)), []
        X_all = np.vstack(X_all)  # shape=(sample_size, d)
        site_indices = np.array(site_indices)
        return X_all, site_indices

    def information_leakage(self, clusters, sample_size=10000, n_procs=2):
        """
        Measures the (joint) information leakage I(C; f) for each cluster in 'clusters'.

        - 'clusters' is a list of lists, where each sub-list is a set of features to be combined.
        - If you want one big “joint leakage” across all features, pass just one sub-list
          containing them all: e.g. clusters=[[f1, f2, ..., fM]].
        """
        if not clusters:
            return []

        # Entropy of the class distribution: H(C)
        H_C = self.max_information_leakage()

        leakages = []
        for cluster in clusters:
            # 1. Build a multi‐dimensional KDE for each site using these 'cluster' features
            kdes = []
            for site in self.sites:
                kde_site = self._make_kde_for_cluster_and_site(cluster, site)
                kdes.append(kde_site)
            log.info(f"[Cluster {cluster}] Constructed KDEs : {len(kdes)}")

            # 2. Monte‐Carlo estimate H(C | f) by sampling from the mixture (with priors)
            X_samples, site_samples = self._sample_from_all_sites(kdes, sample_size)
            if len(X_samples) == 0:
                # corner case: no samples
                leakages.append(0.0)
                continue

            # 3. Evaluate p(x|site) for each site on these same sample points
            #    p(x, site) ~ p(site)*p(x|site).  We'll do everything in log domain.
            def _compute_log_probs(kde, X_samples):
                """
                Helper function to evaluate p(x) for the given KDE,
                and return log2(p(x)) for each row in X_samples.
                """
                pvals = kde.predict(X_samples)
                with np.errstate(divide="ignore"):
                    lpvals = np.log2(pvals)
                return lpvals

            log_probs = Parallel(n_jobs=n_procs)(
                delayed(_compute_log_probs)(kde, X_samples) for kde in kdes
            )
            log_probs = np.array(log_probs)  # shape: (n_sites, n_samples)
            assert len(log_probs) == len(self.sites)

            log.info(f"[Cluster {cluster}] Constructed probs : {len(log_probs)}")

            # 4. For each sample, the posterior p(site|x) ~ p(site) * p(x|site).
            #    So  log p(site|x) = log p(site) + log p(x|site) - log Σ(...)
            # We'll build a stable approach by normalizing across sites.
            priors_log2 = [math.log2(p) for p in self.website_priors]

            # Convert each sample’s log p(x|site) to a posterior distribution over site
            # for that same x.  Then compute the sample's "posterior site" entropy in bits.
            entropies_per_sample = []
            for s_i in range(len(X_samples)):  # each sample
                # log p(x_s_i | site j) + log p(site j)
                lvals = [
                    log_probs[j, s_i] + priors_log2[j] for j in range(len(self.sites))
                ]
                # shift for numerical stability
                max_l = max(lvals)
                shifted = [lv - max_l for lv in lvals]
                # exponentiate in base 2, sum them
                sum_2 = sum(2**lv for lv in shifted)
                # posterior probabilities for each site j
                post_probs = [(2**lv) / sum_2 for lv in shifted]

                # now entropy( posterior ) = - sum( p_j * log2(p_j) )
                H_post = 0.0
                for p_j in post_probs:
                    if p_j > 0:
                        H_post += -p_j * math.log2(p_j)
                entropies_per_sample.append(H_post)
            log.info(
                f"[Cluster {cluster}] Constructed entropies : {len(entropies_per_sample)}"
            )

            # 5. Average these entropies for all samples to get H(C|f).
            H_C_given_f = np.mean(entropies_per_sample)

            # 6. I(C; f) = H(C) - H(C|f).
            I_C_f = H_C - H_C_given_f
            leakages.append(I_C_f)

        return leakages
