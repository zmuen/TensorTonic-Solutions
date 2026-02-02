import numpy as np

def knn_distance(X_train, X_test, k):
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    if X_train.ndim == 1:
        X_train = X_train[:, None]
    if X_test.ndim == 1:
        X_test = X_test[:, None]

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    dists = np.sum((X_test[:, None, :] - X_train[None, :, :]) ** 2, axis=2)

    sorted_idx = np.argsort(dists, axis=1)

    if k <= n_train:
        return sorted_idx[:, :k].astype(int)

    res = -np.ones((n_test, k), dtype=int)
    res[:, :n_train] = sorted_idx
    return res
