# stdlib
import math

# third party
import numpy as np
from scipy.special import digamma
import torch


def _ch_lower_from_r1nn(r1, classes=None):
    """
    Cover–Hart lower bound on Bayes error from 1-NN error.
    If 'classes' is provided and >2, you can plug a multiclass generalization.
    The binary inversion below is conservative and never over-optimistic.
    """
    r1 = float(np.clip(r1, 0.0, 1.0))
    if r1 <= 0.5:
        return 0.5 * (1.0 - np.sqrt(max(0.0, 1.0 - 2.0 * r1)))
    else:
        return 0.5  # monotone extension


def knn_ber(d, y_train, y_test, k=5):
    """
    Return a *lower bound* on Bayes error (from 1-NN via Cover–Hart)
    and an *upper bound* on Bayes error (empirical k-NN test error).
    """
    M = d.shape[0]

    # 1-NN error (rows are test, cols are train)
    idx1 = np.argmin(d, axis=1)
    if y_train.ndim == 1:
        preds1 = y_train[idx1]
    else:
        preds1 = y_train[np.arange(M), idx1]
    r1 = 1.0 - np.mean((preds1 == y_test).astype(float))
    lb_err = _ch_lower_from_r1nn(
        r1, classes=np.unique(np.concatenate((y_train, y_test))).size
    )

    # k-NN error (majority vote); for k=1 this equals r1
    if k == 1:
        rk = r1
    else:
        ind = np.argpartition(d, k - 1, axis=1)[:, :k]  # M x k indices of nearest train
        if y_train.ndim == 1:
            lab = y_train[ind]  # M x k
        else:
            lab = y_train[np.arange(M)[:, None], ind]
        # majority vote
        vals, counts = np.unique(lab, axis=1, return_counts=True)  # slow for big M
        # stdlib
        from collections import Counter

        maj = np.array(
            [Counter(ly).most_common(1)[0][0] for ly in lab], dtype=y_test.dtype
        )
        rk = 1.0 - np.mean((maj == y_test).astype(float))

    ub_err = float(np.clip(rk, 0.0, 1.0))
    lb_err = float(np.clip(lb_err, 0.0, 1.0))

    assert lb_err <= ub_err + 1e-9, f"Inconsistent BER bounds: LB={lb_err}, UB={ub_err}"
    assert d.shape[0] == len(y_test) and d.shape[1] == len(y_train)

    return lb_err, ub_err


def compute_distance(x_train, x_test, measure="squared_l2"):
    """Calculates the distance matrix between test and train.

    Args:
      x_train: Matrix (NxD) where each row represents a training sample
      x_test: Matrix (MxD) where each row represents a test sample
      measure: Distance measure (not necessarly metric) to use
    Raises:
      NotImplementedError: When the measure is not implemented
    Returns:
      Matrix (MxN) where elemnt i,j is the distance between
      x_test_i and x_train_j.
    """
    if torch.cuda.is_available():
        x_train = torch.from_numpy(x_train).float().cuda()
        x_test = torch.from_numpy(x_test).float().cuda()
    else:
        if x_train.dtype != np.float32:
            x_train = np.float32(x_train)
        if x_test.dtype != np.float32:
            x_test = np.float32(x_test)

    if measure == "squared_l2":
        if torch.cuda.is_available():
            x_xt = torch.matmul(x_test, x_train.t()).cpu().numpy()

            x_train_2 = torch.sum(x_train**2, 1).cpu().numpy()
            x_test_2 = torch.sum(x_test**2, 1).cpu().numpy()
        else:
            x_xt = np.matmul(x_test, np.transpose(x_train))

            x_train_2 = np.sum(np.square(x_train), axis=1)
            x_test_2 = np.sum(np.square(x_test), axis=1)

        for i in range(np.shape(x_xt)[0]):
            x_xt[i, :] = np.multiply(x_xt[i, :], -2)
            x_xt[i, :] = np.add(x_xt[i, :], x_test_2[i])
            x_xt[i, :] = np.add(x_xt[i, :], x_train_2)

    elif measure == "cosine":
        # if len(tf.config.list_physical_devices("GPU")) > 0:
        if torch.cuda.is_available():
            x_xt = torch.matmul(x_test, x_train.t()).cpu().numpy()

            x_train_2 = torch.norm(x_train, dim=1).cpu().numpy()
            x_test_2 = torch.norm(x_test, dim=1).cpu().numpy()
        else:
            x_xt = np.matmul(x_test, np.transpose(x_train))

            x_train_2 = np.linalg.norm(x_train, axis=1)
            x_test_2 = np.linalg.norm(x_test, axis=1)

        outer = np.outer(x_test_2, x_train_2)
        x_xt = np.ones(np.shape(x_xt)) - np.divide(x_xt, outer)

    else:
        raise NotImplementedError(f"Method '{measure}' is not implemented")

    return x_xt


def knn_mi(d, y_train, y_test, k=5):
    """Calculate the Mutual Information based on knn method and on the precomputed distance matrix d.

    Args:
      d: Distance matrix (MxN) where elemnt i,j is the distance between
         x_test_i and x_train_j
      y_train: N label vector for the training samples
      y_test: M label vector for the test samples
      k: number of in-class neighbors for every test sample
    Returns:
      Mutual Information based on knn for the k provided
    """

    M, N = d.shape

    keys, counts = np.unique(y_train, return_counts=True)
    key_to_count = dict(zip(keys, counts))

    test_keys, test_class_counts = np.unique(y_test, return_counts=True)
    test_key_to_count = dict(zip(test_keys, test_class_counts))

    dg_N = digamma(N)
    dg_N_Ys = []
    dg_k = digamma(k)
    dg_m_y = []

    for c in keys:
        c = int(c)

        test_indices = np.nonzero(y_test == c)[0]
        train_indices = np.nonzero(y_train == c)[0]

        N_Y = key_to_count[c]
        dg_N_Ys.extend([digamma(N_Y)] * test_key_to_count[c])

        assert k < N_Y, "k should be smaller than then number of samples per class"

        # Get only the elements (train and test) with that class
        sub_d = d[test_indices, :][:, train_indices]

        # Get the k-1 nearest index of train samples w.r.t each test samples
        indices = np.argpartition(sub_d, k - 1, axis=1)

        # Get the distance for the index per test sample
        max_d = sub_d[np.arange(len(test_indices)), indices[:, k - 1]]

        # Filter the train samples based on the distance per test sample
        mask = np.less_equal(d[test_indices, :], np.tile(np.expand_dims(max_d, 1), (N)))

        # Count the number of samples with smaller, or equal distance
        m_y = np.count_nonzero(mask, axis=1)

        dg_m_y.extend(digamma(m_y))

    return max(0.0, dg_N - np.mean(dg_N_Ys) + dg_k - np.mean(dg_m_y)) * np.log2(math.e)
