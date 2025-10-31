import numpy as np

def normalize(X: np.ndarray) -> np.ndarray:
    """Column-wise z-score: (x - mean)/std. If std=0, leave that column at 0."""
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma[sigma == 0] = 1.0
    return (X - mu) / sigma

def minmax_scale(X: np.ndarray) -> np.ndarray:
    """Column-wise scale to [0,1]. If max==min, that column becomes 0."""
    X = np.asarray(X, dtype=float)
    mn = X.min(axis=0)
    mx = X.max(axis=0)
    rng = mx - mn
    rng[rng == 0] = 1.0
    return (X - mn) / rng

def train_test_split(X, y, test_size: float = 0.2, random_state: int = 42):
    """Simple deterministic split using a seeded shuffle."""
    X = np.asarray(X)
    y = np.asarray(y)
    assert 0 < test_size < 1, "test_size must be in (0,1)"
    n = len(X)
    n_test = int(np.ceil(n * test_size))
    rng = np.random.default_rng(seed=random_state)
    idx = np.arange(n)
    rng.shuffle(idx)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
