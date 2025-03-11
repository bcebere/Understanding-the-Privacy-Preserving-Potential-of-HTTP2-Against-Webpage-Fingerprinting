# third party
import numpy as np
from scipy import stats
import statsmodels.api as sm

np.random.seed(42)


class KDE:
    """
    Kernel Density Estimator that:
      - Fits a univariate bandwidth for each feature via plugin or rule-of-thumb
      - Uses statsmodels.nonparametric.KDEMultivariate for 'predict'
      - Samples new points from a discrete mixture of the original data points
        plus a Gaussian offset scaled by the local bandwidth.
    """

    def __init__(self, data, weights=None, bw=None, discrete_threshold=1e10):
        """
        Parameters
        ----------
        data : ndarray of shape (n_samples, n_features)
            The raw data samples on which we build the KDE.
        weights : ndarray of shape (n_samples,), optional
            Mixture weights for each sample. Defaults to uniform.
        bw : ndarray, optional
            Either a single (n_features,)-shape array (bandwidth per feature)
            or a 2D shape=(n_features,n_kernels).  If None, bandwidths are
            computed automatically.
        discrete_threshold : int
            If any particular data point occurs >= this many times,
            we call it "discrete" and assign a small bandwidth to it.
        """
        self.points = data
        self.n_kernels, self.n_features = self.points.shape

        if weights is not None:
            assert weights.shape[0] == self.n_kernels, "weights must match data samples"
            # Ensure sum of weights=1
            self.weights = weights / np.sum(weights)
        else:
            self.weights = np.repeat(1.0 / self.n_kernels, self.n_kernels)

        # --- Determine or set bandwidths ---
        if bw is None:
            # We create a per-sample bandwidth array 'bw_array' first,
            # then consolidate to a single (n_features,) if needed.
            bw_array = np.empty((self.n_features, self.n_kernels))
            bw_array[:] = np.nan

            # Identify which samples are "discrete" (occur at least 'discrete_threshold' times)
            disc_vec = self._identify_discrete(self.points, discrete_threshold)

            # Discrete points get a very small bandwidth
            if np.any(disc_vec == 1):
                bw_array[:, disc_vec == 1] = 0.001

            # For continuous points, estimate bandwidth with Hall or fallback ROT
            if np.any(disc_vec == 0):
                try:
                    with np.errstate(all="raise"):
                        continuous_bw = self._ksizeHall(self.points[disc_vec == 0, :])
                except Exception:
                    continuous_bw = np.array([np.nan])

                if np.isnan(continuous_bw).any() or np.isinf(continuous_bw).any():
                    continuous_bw = self._ksizeROT(self.points[disc_vec == 0, :])

                # Fill those continuous columns in bw_array
                for i in range(self.n_features):
                    bw_array[i, disc_vec == 0] = continuous_bw[i]

            # Finally, pick the "mode" bandwidth across all kernels for each feature
            self.bw = np.zeros((self.n_features,))
            for i in range(self.n_features):
                # stats.mode(...) returns (mode, count); we want [0]
                self.bw[i] = stats.mode(bw_array[i, :])[0]
        else:
            # User-supplied bandwidth
            self.bw = bw

        # Replace zero bandwidths with a small positive number
        self.bw = self.bw + (self.bw == 0.0) * 0.001
        if np.any(self.bw <= 0):
            raise ValueError("All bandwidths must be > 0 after adjustments.")

        # Build statsmodels KDE: treat all features as continuous
        var_vector = "c" * self.n_features
        self._kde = sm.nonparametric.KDEMultivariate(
            data=self.points,
            var_type=var_vector,
            bw=self.bw,
        )

    def sample(self, n_samples):
        """
        Draw random samples from the fitted distribution.  We treat it
        as a mixture over the original data points plus Gaussian noise
        scaled by each dimension's bandwidth.  That is:
            1. Choose a kernel index i ~ 'weights'
            2. x = points[i, :] + bw[i, :] * Normal(0,1)
        except we do not differentiate bandwidth by kernel i here (only by dimension).
        We'll keep a uniform 'self.bw' across all kernels for each dimension.

        Returns
        -------
        points : ndarray of shape (n_samples, n_features)
        """
        # For statsmodels, we have only a single bandwidth per dimension
        # So each dimension's bandwidth is self.bw[d].  If you want a distinct
        # bandwidth for each kernel, you'd store a full 2D array. But here
        # we replicate self.bw for each kernel row, to match old code style.
        full_bw = np.tile(self.bw, (self.n_kernels, 1))  # shape=(n_kernels, n_features)

        # Discrete mixture: choose which kernel to sample from for each row
        chosen_kernels = np.random.choice(
            self.n_kernels, size=n_samples, p=self.weights
        )
        # Normal offsets
        randnums = np.random.normal(size=(n_samples, self.n_features))

        # Build output
        samples = np.zeros((n_samples, self.n_features))
        for i in range(n_samples):
            k_idx = chosen_kernels[i]
            # center + offset
            samples[i, :] = self.points[k_idx, :] + randnums[i, :] * full_bw[k_idx, :]

        return samples

    def predict(self, data):
        """
        Evaluate the KDE pdf at each row in 'data'.

        Parameters
        ----------
        data : ndarray of shape (n_samples, n_features)

        Returns
        -------
        probs : ndarray of shape (n_samples,)
        """
        return self._kde.pdf(data)

    def entropy(self, data=None):
        """
        Calculate an entropy estimate using the negative mean log-likelihood.
        If 'data' is None, compute "resubstitution" entropy with the original data.

        Parameters
        ----------
        data : ndarray of shape (n_samples, n_features), optional

        Returns
        -------
        float
            Estimated entropy (in natural log units).  If you want bits, divide by log(2).
            For pure bits, you can do:   bits = entropy(...)/np.log(2).
        """
        if data is None:
            data = self.points
            w = self.weights
        else:
            # If we have new data, we treat each sample as equally likely
            # i.e., uniform weighting
            w = np.ones(data.shape[0]) / data.shape[0]

        probs = self.predict(data)
        # If any predicted probability is zero => log(prob)= -inf => entire sum = inf
        if np.any(probs == 0.0):
            return -np.inf

        # Weighted average of -log(prob)
        return -np.sum(w * np.log(probs))

    @staticmethod
    def _ksizeROT(X):
        """
        Multivariate 'Rule of Thumb' bandwidth for X, shape=(n_samples,n_features).
        """
        # Transpose: shape => (n_features, n_samples)
        X = X.T
        dim = X.shape[0]
        N = X.shape[1]
        sig = np.std(X, axis=1)
        iqrSig = 0.7413 * stats.iqr(X, axis=1)
        iqrSig[iqrSig == 0] = sig[iqrSig == 0]  # fallback
        h = np.minimum(sig, iqrSig) * N ** (-1.0 / (4 + dim))
        return h

    @staticmethod
    def _ksizeHall(X):
        """
        "Plug-in" bandwidth method by Hall et al.
        May fail if variance is zero or data dimension is large.
        """
        # NOTE: This is the same as your original code, just lightly tidied
        X = X.T  # shape => (n_features, n_samples)
        n_features, n_samples = X.shape
        sig = np.std(X, axis=1)
        lamS = 0.7413 * stats.iqr(X, axis=1)
        lamS[lamS == 0] = sig[lamS == 0]

        # "Initial" plugin guess
        BW = 1.0592 * lamS * (n_samples ** (-1.0 / (4 + n_features)))
        # We replicate each dimension’s BW across all samples for next steps:
        BW_expand = np.tile(BW, (1, n_samples))

        # Build dX array
        t = np.transpose(X[:, :, None], (0, 2, 1))  # shape=(f,1,n_samples)
        dX = np.tile(t, (1, n_samples, 1))  # shape=(f,n_samples,n_samples)
        for i in range(n_samples):
            dX[:, :, i] = (dX[:, :, i] - X) / BW_expand
        # avoid self-distances
        for i in range(n_samples):
            dX[:, i, i] = 2e22
        dX = np.reshape(dX, (n_features, n_samples * n_samples))

        def h_findI2(n, dXa, alpha):
            # sum over axis=0 => each column is a data point
            t = np.exp(-0.5 * np.sum(dXa**2, axis=0))
            t = (
                (dXa**2 - 1)
                * (1 / np.sqrt(2 * np.pi))
                * np.tile(t, (dXa.shape[0], 1))
            )
            s = np.sum(t, axis=1)
            return s / (n * (n - 1) * (alpha**5))

        def h_findI3(n, dXb, beta):
            t = np.exp(-0.5 * np.sum(dXb**2, axis=0))
            t = (
                (dXb**3 - 3 * dXb)
                * (1 / np.sqrt(2 * np.pi))
                * np.tile(t, (dXb.shape[0], 1))
            )
            s = np.sum(t, axis=1)
            return -s / (n * (n - 1) * (beta**7))

        # Evaluate I2,I3 on dimension #0 as a reference
        I2 = h_findI2(n_samples, dX, BW_expand[:, 1])
        I3 = h_findI3(n_samples, dX, BW_expand[:, 1])

        # constants
        RK, mu2, mu4 = 0.282095, 1.0, 3.0
        J1 = (RK / (mu2**2)) * (1.0 / I2)
        J2 = (mu4 * I3) / (20 * mu2) * (1.0 / I2)

        h = (J1 / n_samples) ** (1 / 5) + J2 * ((J1 / n_samples) ** (3 / 5))
        h = h.real.astype(float)
        return h

    def _identify_discrete(self, data, threshold):
        """
        Mark a data sample as "discrete" if it occurs >= threshold times.
        data: shape=(n_samples, n_features)
        threshold: int
        """
        sampleNum = data.shape[0]
        isDiscVec = np.full(sampleNum, np.nan)

        for i in range(sampleNum):
            if not np.isnan(isDiscVec[i]):
                continue
            # find all rows j that are identical to row i
            match_list = []
            for j in range(sampleNum):
                if np.array_equal(data[i, :], data[j, :]):
                    match_list.append(j)

            if len(match_list) >= threshold:
                isDiscVec[match_list] = 1  # discrete
            else:
                isDiscVec[match_list] = 0  # continuous

        return isDiscVec
